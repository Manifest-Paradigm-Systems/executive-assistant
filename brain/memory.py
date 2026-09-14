"""Jarvis memory worker — turn raw transcripts into things worth remembering.

The problem this solves: `brain.py` logs every turn faithfully, which means the
session log holds "are you there?" and "we decided X" with exactly equal weight.
Searching that store returns noise.

This worker runs out-of-band on a timer:

    new turns ──► window ──► discriminator (a local lane) ──► facts + archive
                                                     │
                                             consolidation vs. the existing pool

Two decisions worth stating plainly, because they are the difference between a
memory that gets better with use and one that rots:

1. **Consolidate on a lexical heuristic, not on the model's judgement.** The model's
   job is extraction: "is there a durable fact here?" It never sees the existing pool,
   so it never gets to decide what to overwrite. Merging is done here, deterministically,
   with an explainable rule — and since a superseded fact is *retired, never deleted*,
   a wrong merge is visible and reversible.
2. **Never advance the cursor on a partial window.** A model timeout or a malformed
   reply leaves the cursor where it was; the next tick retries the same window. Nothing
   is silently dropped, and nothing is half-written.

Run:
    python3 memory.py --once            # one pass
    python3 memory.py --once --dry-run  # show what it would store
    python3 memory.py --stats
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request

import db as jarvis_db

LANE = os.getenv("JARVIS_MEMORY_LANE", "http://127.0.0.1:8082")
MODEL = os.getenv("JARVIS_MEMORY_MODEL", "coder")
WINDOW = int(os.getenv("JARVIS_MEMORY_WINDOW", "24"))       # turns per extraction call
MIN_CHARS = int(os.getenv("JARVIS_MEMORY_MIN_CHARS", "200"))  # skip trivial windows
CURSOR = "memory.turns"

# ------------------------------------------------------------------ discrimination

DISCRIMINATOR = """You are the Memory Subsystem of Jarvis, a private assistant running on the owner's own hardware.

Read the transcript segment and extract ONLY what is worth remembering long-term: \
durable facts, the owner's explicit preferences, system configurations, and decisions \
that were actually reached.

IGNORE AND FLUSH — never report these:
- conversational filler ("Are you there?", "Hello", "Thanks", "Can you hear me?")
- transient status checks and immediate one-off intents ("what time is it", "call the dentist now")
- speculation, brainstorming that reached no conclusion, and questions with no answer
- anything you are inferring rather than reading

For a decision, record the DECISION, not the debate that preceded it. If the segment \
revisits a decision and CHANGES it, record the new position — a later worker resolves \
which one stands.

Reply with ONE JSON object and nothing else. No markdown fence, no commentary.

{
  "extracted_facts": [
    {"statement": "<one declarative sentence, self-contained, no pronouns>",
     "entity": "<one of: cerebro_system, workhorse, user, rental_property, dave_care, cinematome, fleet, other>",
     "topic": "<short snake_case subject, e.g. memory_architecture>",
     "kind": "<fact|preference|config|decision>"}
  ],
  "archive": {
    "worth_keeping": <true if this segment contains reasoning or alternatives that would matter later>,
    "title": "<short title>",
    "summary": "<2-4 sentences: what was discussed, what was chosen, and what was rejected and why>"
  }
}

If nothing qualifies, return {"extracted_facts": [], "archive": {"worth_keeping": false}}.

TRANSCRIPT SEGMENT:
"""

_FILLER = re.compile(
    r"^\s*(are you (there|awake)|hello|hi|hey|thanks?|thank you|ok(ay)?|yes|no|"
    r"good (morning|evening|afternoon)|can you hear me|you there|testing)\b[\s.!?]*$",
    re.IGNORECASE)


def looks_like_filler(text: str) -> bool:
    return bool(_FILLER.match(text or "")) or len((text or "").strip()) < 12


def window_is_trivial(rows) -> bool:
    """Cheap pre-filter: skip windows that are nothing but greetings.

    Saves a model call on the overwhelmingly common case, and keeps the
    extractor from having to be trusted on input it should never have seen.
    """
    substance = [r["content"] for r in rows if not looks_like_filler(r["content"])]
    return sum(len(c) for c in substance) < MIN_CHARS


# ------------------------------------------------------------------ model call

def _extract_json(text: str) -> dict | None:
    """Pull a JSON object out of whatever the lane actually returned.

    Local lanes fence their JSON in markdown, prepend a sentence, or emit a
    trailing \x00. Being strict here would mean dropping real facts on the floor.
    """
    if not text:
        return None
    text = text.strip().replace("\x00", "")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except json.JSONDecodeError:
            pass
    # First {...} block, brace-matched.
    start = text.find("{")
    while start != -1:
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def call_lane(prompt: str, timeout: int = 300) -> str:
    payload = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": 1600, "temperature": 0.1}
    req = urllib.request.Request(f"{LANE}/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read())
    msg = body["choices"][0]["message"]
    return ((msg.get("content") or "") + "\n" + (msg.get("reasoning_content") or "")).strip()


# ------------------------------------------------------------------ window -> facts

def fetch_window(conn, after_id: int, limit: int):
    return list(conn.execute(
        "SELECT id, ts, session, model, role, content FROM turns WHERE id > ?"
        " ORDER BY id LIMIT ?", (after_id, limit)))


def transcript_of(rows) -> str:
    out = []
    for r in rows:
        who = {"user": "OWNER", "assistant": "JARVIS", "system": "SYSTEM"}.get(r["role"], r["role"])
        out.append(f"[{who}] {r['content']}")
    return "\n".join(out)


def consolidate(conn, fact: dict, *, session: str, dry: bool) -> tuple[str, int | None]:
    """Decide insert vs. duplicate vs. supersede. Returns (action, row_id).

    The rule, in order:
      - a near-identical statement is a duplicate  -> reinforce, don't duplicate rows
      - a similar subject with different content    -> a revision, so supersede
      - otherwise                                   -> a new belief
    Lexical, so it is auditable and cheap; and it errs toward *not* merging.
    """
    statement = (fact.get("statement") or "").strip()
    if not statement:
        return "empty", None
    entity, topic = fact.get("entity"), fact.get("topic")
    match = jarvis_db.find_similar_fact(conn, statement, entity=entity, topic=topic)
    if match is None:
        if dry:
            return "insert", None
        fid = jarvis_db.add_fact(conn, statement, entity=entity, topic=topic,
                                 kind=fact.get("kind", "fact"), source_session=session)
        return "insert", fid

    a, b = jarvis_db._words(match["statement"]), jarvis_db._words(statement)
    overlap = len(set(a) & set(b)) / max(1, len(set(a) | set(b)))
    if overlap >= 0.9:
        if not dry:
            with conn:
                conn.execute("UPDATE facts SET confidence=MIN(2.0, confidence+0.1), updated=?"
                             " WHERE id=?", (time.time(), match["id"]))
        return "reinforce", match["id"]
    if not dry:
        fid = jarvis_db.add_fact(conn, statement, entity=entity, topic=topic,
                                 kind=fact.get("kind", "fact"), supersedes=match["id"],
                                 source_session=session)
        return "supersede", fid
    return "supersede", match["id"]


def process_window(conn, rows, *, dry: bool) -> dict:
    result = {"facts": 0, "insert": 0, "reinforce": 0, "supersede": 0, "archived": 0,
              "actions": [], "skipped": None}

    if window_is_trivial(rows):
        result["skipped"] = "filler-only window"
        return result

    reply = call_lane(DISCRIMINATOR + transcript_of(rows))
    data = _extract_json(reply)
    if data is None:
        # No parse -> no cursor advance -> the next tick retries this window.
        result["skipped"] = "unparseable model reply"
        return result

    session = rows[0]["session"]
    for fact in (data.get("extracted_facts") or []):
        if not isinstance(fact, dict):
            continue
        action, fid = consolidate(conn, fact, session=session, dry=dry)
        if action == "empty":
            continue
        result["facts"] += 1
        result[action] = result.get(action, 0) + 1
        result["actions"].append(f"{action}: {(fact.get('statement') or '')[:90]}")

    arch = data.get("archive") or {}
    if arch.get("worth_keeping") and (arch.get("summary") or "").strip():
        if not dry:
            jarvis_db.add_chunk(conn, arch["summary"], title=arch.get("title") or "conversation",
                                session=session, ts_from=rows[0]["ts"], ts_to=rows[-1]["ts"])
        result["archived"] = 1

    return result


# ------------------------------------------------------------------ driver

def run_once(dry: bool = False, max_windows: int = 5, verbose: bool = True) -> dict:
    conn = jarvis_db.open_db()
    start = int(jarvis_db.get_cursor(conn, CURSOR, "0"))
    totals = {"windows": 0, "facts": 0, "insert": 0, "reinforce": 0, "supersede": 0,
              "archived": 0, "skipped": 0, "last_id": start}

    for _ in range(max_windows):
        rows = fetch_window(conn, totals["last_id"], WINDOW)
        if not rows:
            break
        if verbose:
            print(f"  window #{totals['windows'] + 1}: turns {rows[0]['id']}..{rows[-1]['id']} "
                  f"({len(rows)} turns)")
        res = process_window(conn, rows, dry=dry)
        totals["windows"] += 1
        totals["facts"] += res["facts"]
        totals["insert"] += res["insert"]
        totals["reinforce"] += res["reinforce"]
        totals["supersede"] += res["supersede"]
        totals["archived"] += res["archived"]
        if res["skipped"]:
            totals["skipped"] += 1
            if verbose:
                print(f"    skipped: {res['skipped']}")
        for line in res["actions"]:
            if verbose:
                print(f"    {line}")
        totals["last_id"] = rows[-1]["id"]
        # A skipped window still advances: retrying filler forever would wedge the
        # queue behind it. Only an unparseable reply is worth a retry, and that
        # leaves the cursor alone.
        if res["skipped"] == "unparseable model reply":
            totals["last_id"] = rows[0]["id"] - 1 if rows[0]["id"] > start else start
            break
        if not dry:
            jarvis_db.set_cursor(conn, CURSOR, totals["last_id"])

    if dry and verbose:
        print("  (dry run — cursor not advanced)")
    return totals


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--once", action="store_true", help="one pass (default)")
    ap.add_argument("--dry-run", action="store_true", help="show, do not write")
    ap.add_argument("--max-windows", type=int, default=5)
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    conn = jarvis_db.open_db()
    if a.stats:
        c = int(jarvis_db.get_cursor(conn, CURSOR, "0"))
        total = conn.execute("SELECT COALESCE(MAX(id),0) m FROM turns").fetchone()["m"]
        n_all = conn.execute("SELECT count(*) n FROM facts").fetchone()["n"]
        n_active = conn.execute("SELECT count(*) n FROM facts WHERE status='active'").fetchone()["n"]
        print(f"cursor at turn {c} of {total} ({max(0, total - c)} behind)")
        print(f"facts: {n_all} ({n_active} active, {n_all - n_active} retired)")
        for r in conn.execute("SELECT entity, topic, statement FROM facts WHERE status='active'"
                              " ORDER BY updated DESC LIMIT 15"):
            print(f"  {r['entity'] or '-'}/{r['topic'] or '-'}: {r['statement'][:100]}")
        return 0

    t0 = time.time()
    totals = run_once(dry=a.dry_run, max_windows=a.max_windows, verbose=not a.quiet)
    print(f"memory: {totals['windows']} window(s), {totals['facts']} fact(s) "
          f"({totals['insert']} new, {totals['reinforce']} reinforced, "
          f"{totals['supersede']} superseded), {totals['archived']} archived, "
          f"{totals['skipped']} skipped — {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
