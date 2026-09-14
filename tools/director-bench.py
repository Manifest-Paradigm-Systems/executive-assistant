#!/usr/bin/env python3
"""Is Q8_0 actually better than Q5_K_M at the director's job?

The `runs` table stores the prompt AND the answer for every director call the Q5 lane ever
served. That makes the old arm free: replay the exact prompt text against the new lane and
the only thing that differs between the two answers is the weights.

Two kinds of evidence are collected, deliberately of different strengths:

* **Objective** — did the answer parse, how long did it take, how fast did it generate. These
  need no judge and cannot flatter either model.
* **Judged** — a blind pairwise verdict from the cloud adjudicator the project already trusts
  for exactly this call ("is this verdict real?"). Order is decided by a hash of the row id,
  not a coin, so the run is reproducible; the judge is never told which answer came from
  which build.

The judge can be wrong and the sample is small — eight prompts is a signal, not a trial. The
objective columns are printed beside every verdict so a disagreement between the numbers and
the judge is visible rather than smoothed over.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import sys
import time
import urllib.request

# Paths come from the environment with the live deployment as the default, so this runs
# unchanged on cerebro and can be pointed at a scratch copy anywhere else.
BRAIN = os.getenv("JARVIS_BRAIN_DIR", os.path.expanduser("~/jarvis/brain"))
sys.path.insert(0, BRAIN)
import llm  # noqa: E402

DB = os.getenv("JARVIS_DB", os.path.expanduser("~/jarvis/conversations.db"))
OUT = os.getenv("BENCH_OUT", "/tmp/dq8-bench/results.jsonl")
LANE = os.getenv("BENCH_LANE", "http://127.0.0.1:8081")

# Samples per task kind. Plan and falsify are rare in the corpus and slow, so they get one
# each; review and repair carry the volume and get three.
WANT = {"plan": 1, "review": 3, "repair": 3, "falsify": 1}

# The call sites' own settings, so the replay matches how the team really calls out.
# A repair consult is issued at 0.3 in the item loop (the dominant site) and 0.4 from the
# options/triage passes; the prompt text does not say which produced it, so 0.3 is used and
# this is the one knob that is approximate.
SETTINGS = {
    "plan":     {"max_tokens": 12000, "temperature": 0.4},
    "review":   {"max_tokens": 12000, "temperature": 0.2},
    "repair":   {"max_tokens": 8000,  "temperature": 0.3},
    "falsify":  {"max_tokens": 8000,  "temperature": 0.3},
    "triage":   {"max_tokens": 8000,  "temperature": 0.3},
}

KINDS = [
    (r"You are the DIRECTOR planning work", "plan"),
    (r"You are the DIRECTOR\. You are reviewing a plan", "review"),
    (r"You are the DIRECTOR\. A work item has failed", "repair"),
    (r"You are the DIRECTOR, and this pass is adversarial", "falsify"),
    (r"You are the DIRECTOR reading your own team", "triage"),
]


def kind_of(prompt: str) -> str:
    for pat, name in KINDS:
        if re.search(pat, prompt or ""):
            return name
    return "other"


def log(msg: str) -> None:
    print(msg, flush=True)


def lane_idle(timeout: int = 900) -> bool:
    """Wait for the director to finish anything real before adding to its queue.

    The lane has ONE slot now. A tick that consults the director while this bench is
    mid-generation would sit in the queue, and a consult that waits long enough dies as
    "director unavailable" — which blocks a work item for a reason that is entirely ours.
    So: never start a benchmark call while a real one is in flight.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{LANE}/slots", timeout=5) as r:
                slots = json.loads(r.read())
            if all(s.get("state") is None for s in slots):
                return True
        except Exception:
            pass
        time.sleep(20)
    return False


def call_lane(prompt: str, max_tokens: int, temperature: float) -> dict:
    payload = json.dumps({"model": "director",
                          "messages": [{"role": "user", "content": prompt}],
                          "max_tokens": max_tokens, "temperature": temperature}).encode()
    req = urllib.request.Request(f"{LANE}/v1/chat/completions", data=payload,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        body = json.loads(r.read())
    ms = int((time.time() - t0) * 1000)
    msg = body["choices"][0]["message"]
    text = llm.strip_think((msg.get("content") or "") + "\n" + (msg.get("reasoning_content") or ""))
    usage = body.get("usage") or {}
    gen = usage.get("completion_tokens") or 0
    return {"text": text, "ms": ms, "gen_tokens": gen,
            "tok_s": round(gen / (ms / 1000), 2) if ms and gen else None}


JUDGE = """You are comparing two answers to the SAME engineering task, produced by two builds
of the same model. You are not told which build produced which — judge only the answers.

THE TASK THE MODEL WAS GIVEN:
{prompt}

ANSWER A:
{a}

ANSWER B:
{b}

Which answer is more likely to lead to work that PASSES ITS VERIFICATION: concrete and
actionable, correct, and consistent with any design contract stated in the task. Penalise
vagueness, invented file names or signatures, restating the task instead of answering it, and
anything that would not run. A shorter answer that is correct beats a longer one that is not.

Reply with JSON only: {{"winner": "A" or "B" or "tie", "confidence": 0.0-1.0, "why": "<=30 words"}}"""


def judge(prompt: str, q5: str, q8: str, q8_is_b: bool) -> dict:
    a, b = (q5, q8) if q8_is_b else (q8, q5)
    try:
        raw = llm.cloud_chat(JUDGE.format(prompt=prompt[:6000], a=a[:6000], b=b[:6000]),
                             max_tokens=400, timeout=240)
        got = llm.extract_json(raw) or {}
    except Exception as e:                                   # noqa: BLE001
        return {"winner": "error", "why": f"{type(e).__name__}: {e}"[:120], "confidence": None}
    winner = str(got.get("winner", "")).strip().lower()
    if winner in ("a", "b"):
        winner = ("q8" if (winner == "b") == q8_is_b else "q5")
    elif winner == "tie":
        winner = "tie"
    else:
        winner = "unparsed"
    return {"winner": winner, "confidence": got.get("confidence"),
            "why": str(got.get("why", ""))[:200]}


def pick(conn) -> list[dict]:
    # db.log_run stores (prompt or "")[:8000] and (output or "")[:8000] — a row that is
    # exactly 8000 chars long is a CUT-OFF record, not a complete one. Replaying a truncated
    # prompt (or judging a truncated answer) would rig the comparison, so those rows are
    # excluded outright rather than used with a footnote. 16 of the 107 director prompts are
    # stored cut off (15%) — a real loss to the historical corpus, though not a majority.
    rows = list(conn.execute(
        "SELECT rowid, ts, plan_id, item_id, prompt, output, ms FROM runs "
        "WHERE role='director' AND output IS NOT NULL AND length(output) > 200 "
        "  AND length(prompt) < 8000 AND length(output) < 8000 "
        "ORDER BY ts DESC"))
    chosen, seen = [], {k: 0 for k in WANT}
    for rid, ts, plan_id, item_id, prompt, output, ms in rows:
        k = kind_of(prompt)
        if k not in WANT or seen[k] >= WANT[k]:
            continue
        # Prefer distinct plans/items so one pathological item cannot dominate the sample.
        if any(c["kind"] == k and c["item_id"] == item_id and item_id for c in chosen):
            continue
        seen[k] += 1
        chosen.append({"rowid": rid, "kind": k, "plan": plan_id, "item_id": item_id,
                       "prompt": prompt, "q5": llm.strip_think(output), "q5_ms": ms})
    return chosen


def main() -> int:
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    picks = pick(conn)
    log(f"selected {len(picks)} prompts: " + ", ".join(f"{p['kind']}#{p['rowid']}" for p in picks))

    tally = {"q5": 0, "q8": 0, "tie": 0, "unparsed": 0, "error": 0}
    for i, p in enumerate(picks, 1):
        cfg = SETTINGS.get(p["kind"], SETTINGS["repair"])
        if not lane_idle():
            log(f"[{i}] lane never went idle — skipping {p['kind']}#{p['rowid']}")
            continue
        log(f"[{i}/{len(picks)}] {p['kind']} #{p['rowid']} (plan={p['plan']} item={p['item_id']}) "
            f"— generating…")
        try:
            got = call_lane(p["prompt"], cfg["max_tokens"], cfg["temperature"])
        except Exception as e:                               # noqa: BLE001
            log(f"    lane error: {type(e).__name__}: {e}")
            continue
        q5_parse = llm.extract_json(p["q5"]) is not None
        q8_parse = llm.extract_json(got["text"]) is not None
        # Blind ordering: hash of the row id, so re-running reproduces the same A/B.
        q8_is_b = hashlib.sha256(str(p["rowid"]).encode()).digest()[0] % 2 == 0
        log(f"    Q8 {got['ms']/1000:.0f}s {got['tok_s']} t/s parse={q8_parse} | "
            f"Q5 {p['q5_ms']/1000:.0f}s parse={q5_parse}")
        verdict = judge(p["prompt"], p["q5"], got["text"], q8_is_b)
        tally[verdict["winner"]] = tally.get(verdict["winner"], 0) + 1
        log(f"    judge: {verdict['winner']} (conf {verdict['confidence']}) — {verdict['why']}")
        rec = {**{k: v for k, v in p.items() if k != "prompt"},
               "q5_chars": len(p["q5"]), "q5_parse": q5_parse, "q5_tok_s": None,
               "q8": got["text"], "q8_ms": got["ms"], "q8_tok_s": got["tok_s"],
               "q8_gen_tokens": got["gen_tokens"], "q8_parse": q8_parse,
               "q8_is_b": q8_is_b, "verdict": verdict}
        with open(OUT, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

    log("")
    log("=== tally ===")
    for k in ("q8", "q5", "tie", "unparsed", "error"):
        log(f"  {k}: {tally.get(k, 0)}")
    log(f"raw results: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
