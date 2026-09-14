"""jarvis brain — the fleet's assistant, served as OpenAI-compatible models.

Two brains behind one endpoint, because conversation and planning have very
different clocks:

  jarvis            Kunou-14B (:8083).  The CONVERSATIONALIST. Fast enough to
                    talk. It does not do the heavy work itself — it delegates:
                    background jobs to the coder or the director, then keeps
                    talking until the results come back.
  jarvis-director   R1-32B (:8081).  The PLANNER. Slow (30-40 s a turn), so it
                    is not the voice; it is who the conversationalist asks when
                    a task needs a real plan. Keeps the approval gate: it
                    proposes, the human approves, then it hands tasks to the
                    coder one at a time.

Why an OpenAI endpoint rather than an Open WebUI plugin: the assistant then
appears in the model switcher like any other model, no OWUI-internal code, and
it streams like any other model.

Surface:
  GET  /health                  liveness + job/session counts
  GET  /v1/models               jarvis, jarvis-director
  POST /v1/chat/completions     streaming or whole
  GET  /events                  SSE status feed (docks panel + face bus)
  POST /approve                 programmatic gate for the panel

Delegation is asynchronous on purpose: a 14B conversationalist answering in ~2 s
can dispatch a 30 s coder job, reply "I've set that going, sir" and still be
responsive when the result lands. Jobs are in-memory; a restart forgets them.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import queue
import re
import sqlite3
import threading
import time
import uuid

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ag2 import Agent
from ag2.config import OpenAIConfig

DIRECTOR_URL = os.getenv("AG2_DIRECTOR_URL", "http://127.0.0.1:8081")
CODER_URL = os.getenv("AG2_CODER_URL", "http://127.0.0.1:8082")
CONVERSATIONAL_URL = os.getenv("AG2_CONVERSATIONAL_URL", "http://127.0.0.1:8083")
CONVERSATIONAL_MODEL = os.getenv("AG2_CONVERSATIONAL_MODEL", "actor")
CODER_FAILURE_LIMIT = int(os.getenv("AG2_CODER_FAILURE_LIMIT", "3"))
MODEL_NAME = os.getenv("JARVIS_MODEL_NAME", "jarvis")
DIRECTOR_MODEL_NAME = "jarvis-director"
# Answers from live sources, but the *model* stays local: SearxNG (self-hosted,
# no API key) fetches the pages, the conversationalist reads them and answers
# with citations. Only the search leaves the house.
WEB_MODEL_NAME = "jarvis-web"
NEWS_MODEL_NAME = "jarvis-news"
WIKI_MODEL_NAME = "jarvis-wiki"
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://127.0.0.1:8888/search")
WIKI_URL = os.getenv("JARVIS_WIKI_URL", "http://127.0.0.1:8090")
SEARCH_RESULTS = int(os.getenv("JARVIS_SEARCH_RESULTS", "6"))
PERSONA_FILE = os.getenv("JARVIS_PERSONA_FILE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "persona.txt"))
SESSION_TTL = float(os.getenv("JARVIS_SESSION_TTL", "43200"))  # 12 h

# ---------------------------------------------------------------- persona

DEFAULT_PERSONA = (
    "You are Jarvis, a personal assistant: unflappable, impeccably organised and "
    "quietly amused. Formal British cadence, measured and precise, never hurried; "
    "address the user as 'sir'. Dry wit welcome, sycophancy not. Offer the sensible "
    "next step rather than a menu of options."
)


def load_persona() -> str:
    try:
        with open(PERSONA_FILE, encoding="utf-8") as fh:
            text = fh.read().strip()
        if text:
            return text
    except OSError:
        pass
    return DEFAULT_PERSONA


# ---------------------------------------------------------------- state

STATE_LOCK = threading.Lock()
SESSIONS: dict[str, dict] = {}
JOBS: dict[str, dict] = {}
EVENT_SINKS: set[queue.Queue] = set()


DB_PATH = os.getenv("JARVIS_DB", "/var/home/admin/jarvis/conversations.db")


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("""CREATE TABLE IF NOT EXISTS turns (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts REAL, session TEXT, model TEXT, role TEXT, content TEXT)""")
    # FTS5 from the start: the point of keeping conversations is being able to
    # find them again, and a LIKE scan over years of chat is not a plan.
    conn.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5(
        content, session UNINDEXED, ts UNINDEXED, role UNINDEXED)""")
    return conn


def record(session: str, model: str, role: str, content: str, ephemeral: bool = False) -> None:
    """Persist one turn. Incognito conversations write nothing at all."""
    if ephemeral or not (content or "").strip():
        return
    try:
        with _db() as c:
            cur = c.execute("INSERT INTO turns (ts, session, model, role, content) VALUES (?,?,?,?,?)",
                            (time.time(), session, model, role, content))
            c.execute("INSERT INTO turns_fts (rowid, content, session, ts, role) VALUES (?,?,?,?,?)",
                      (cur.lastrowid, content, session, time.time(), role))
    except sqlite3.Error as exc:      # never lose a reply because the log failed
        print(f"[jarvis-brain] conversation log failed: {exc}", flush=True)


def recall(query: str, limit: int = 8) -> list[dict]:
    """Search past conversations (FTS5). Used by the recall tool and /recall."""
    terms = " OR ".join(re.findall(r"[A-Za-z0-9_]{3,}", query or "")) or "''"
    try:
        with _db() as c:
            rows = c.execute(
                "SELECT t.ts, t.session, t.model, t.role, t.content FROM turns_fts f "
                "JOIN turns t ON t.id = f.rowid WHERE turns_fts MATCH ? "
                "ORDER BY rank LIMIT ?", (terms, limit)).fetchall()
    except sqlite3.Error:
        return []
    return [{"ts": r[0], "session": r[1], "model": r[2], "role": r[3], "content": r[4][:1200]}
            for r in rows]


def emit(kind: str, **fields) -> None:
    event = {"ts": time.time(), "kind": kind, **fields}
    with STATE_LOCK:
        sinks = list(EVENT_SINKS)
    for sink in sinks:
        try:
            sink.put_nowait(event)
        except queue.Full:
            pass


def session_for(messages: list[dict]) -> dict:
    first_user = next((m.get("content") or "" for m in messages if m.get("role") == "user"), "")
    system = next((m.get("content") or "" for m in messages if m.get("role") == "system"), "")
    key = hashlib.sha1(f"{system}\x00{first_user}".encode()).hexdigest()[:16]
    now = time.time()
    with STATE_LOCK:
        for k, s in list(SESSIONS.items()):
            if now - s["touched"] > SESSION_TTL:
                SESSIONS.pop(k, None)
        sess = SESSIONS.get(key)
        if sess is None:
            sess = {"approved": False, "plan": "", "failures": 0, "log": [], "key": key,
                    "created": now, "touched": now}
            SESSIONS[key] = sess
    sess["touched"] = now
    return sess


APPROVAL_WORDS = re.compile(
    r"^\s*(y|yes|yeah|yep|ok|okay|approve[d]?|go ahead|proceed|do it|sounds good|"
    r"carry on|make it so|affirmative)\b", re.IGNORECASE)


def looks_like_approval(text: str) -> bool:
    return bool(APPROVAL_WORDS.match(text or ""))


# ---------------------------------------------------------------- helpers

def _strip_think(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def _chat(base_url: str, model: str, system: str, user: str, max_tokens: int = 2000,
          temperature: float | None = None) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "max_tokens": max_tokens,
        "temperature": temperature if temperature is not None else (0.6 if model == "director" else 0.7),
    }
    with httpx.Client(timeout=900.0) as c:
        r = c.post(f"{base_url}/v1/chat/completions", json=payload)
        r.raise_for_status()
        body = r.json()
    msg = body["choices"][0]["message"]
    return ((msg.get("content") or "") + "\n" + (msg.get("reasoning_content") or "")).strip()


# ---------------------------------------------------------------- background jobs

CODER_SYSTEM = ("You are the coder. Execute the task concretely and report exactly what you did; "
                "if something failed, say FAILED and why.")
DIRECTOR_SYSTEM = ("You are the director. Produce a concrete, ordered plan for the task: the steps, "
                   "the order, the risks. Do not execute anything — plan only.")


def start_job(target: str, task: str) -> str:
    """Dispatch work in the background and return immediately."""
    job_id = uuid.uuid4().hex[:8]
    url = CODER_URL if target == "coder" else DIRECTOR_URL
    model = "coder" if target == "coder" else "director"
    system = CODER_SYSTEM if target == "coder" else DIRECTOR_SYSTEM
    with STATE_LOCK:
        JOBS[job_id] = {"id": job_id, "target": target, "task": task, "status": "running",
                        "result": "", "started": time.time(), "delivered": False}
    emit("job_start", job=job_id, target=target, task=task[:400])

    def run():
        try:
            reply = _strip_think(_chat(url, model, system, task, max_tokens=2000))
            status = "done"
        except Exception as exc:  # noqa: BLE001
            reply, status = f"{type(exc).__name__}: {exc}", "failed"
        with STATE_LOCK:
            job = JOBS.get(job_id)
            if job:
                job.update(status=status, result=reply[:6000], finished=time.time())
        emit("job_done", job=job_id, target=target, ok=status == "done",
             task=task[:200], result=reply[:600])

    threading.Thread(target=run, daemon=True).start()
    return job_id


def job_digest() -> str:
    """What the conversationalist should know about work in flight."""
    with STATE_LOCK:
        jobs = list(JOBS.values())
    running = [j for j in jobs if j["status"] == "running"]
    fresh = [j for j in jobs if j["status"] != "running" and not j["delivered"]]
    lines = []
    for j in running:
        lines.append(f"[running {j['id']}] {j['target']} job: {j['task'][:180]}")
    for j in fresh:
        j["delivered"] = True
        lines.append(f"[finished {j['id']}] {j['target']} job: {j['task'][:120]}\n"
                     f"result: {j['result'][:1200]}")
    return "\n".join(lines)


# ---------------------------------------------------------------- conversational tools

def send_to_coder(task: str) -> str:
    """Hand a concrete, well-specified task to the coder in the background.
    Returns immediately — do NOT wait for it, and do NOT invent its result."""
    job = start_job("coder", task)
    return (f"Dispatched to the coder as job {job}. It runs in the background; carry on talking. "
            f"You will be given the result when it lands — do not guess at it.")


def send_to_director(task: str) -> str:
    """Ask the director for a detailed plan in the background. Use when the task
    needs real thought about approach, not just execution."""
    job = start_job("director", task)
    return (f"Dispatched to the director as job {job}. It runs in the background; carry on talking. "
            f"You will be given the plan when it lands — do not guess at it.")


def check_background_work() -> str:
    """Check on delegated work. Returns anything that has finished since you last
    looked, and what is still running."""
    digest = job_digest()
    return digest or "Nothing in flight."


DELEGATE_RE = re.compile(r"^\s*@@\s*DELEGATE\s+(coder|director)\s*:\s*(.+?)\s*$",
                         re.IGNORECASE | re.MULTILINE)
PLAN_RE = re.compile(r"@@PLAN\s*\n(.*?)\n\s*@@END", re.DOTALL)
HANDOFF_RE = re.compile(r"^\s*@@\s*HANDOFF\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)

# AG2 1.0.4 registers tools but never executes them (llama.cpp returns a valid
# tool_calls block; AG2 answers with the model's raw text and no ToolResult).
# Delegation therefore rides an explicit text protocol the brain parses — less
# elegant, but it works with every model here and is trivial to debug.
CHOOSE_RE = re.compile(r"^\s*@@\s*CHOOSE\s+(\S+)\s+(\d+)\s*$",
                       re.IGNORECASE | re.MULTILINE)
VOICE_RE = re.compile(r"^\s*@@\s*VOICE\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
PLAY_RE = re.compile(r"^\s*@@\s*PLAY\s+(foley|music|sfx)\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
# The adapter on the Workhorse owns both libraries (audio stays on that box).
MEDIA_URL = os.getenv("JARVIS_MEDIA_URL", "http://192.168.0.62:7863")
VOICES_URL = os.getenv("JARVIS_VOICES_URL", "http://192.168.0.62:7863/v1/audio/voices")
_VOICE_CACHE: dict = {"at": 0.0, "voices": []}


def find_media(kind: str, description: str) -> dict | None:
    """One sound or one track for a description. Foley goes through CLAP (prose
    -> sound); music matches the descriptive filenames of the packs."""
    endpoint = "foley/search" if kind in ("foley", "sfx") else "music/search"
    try:
        with httpx.Client(timeout=180.0) as c:
            data = c.get(f"{MEDIA_URL}/media/{endpoint}",
                         params={"q": description, "k": 3}).json()
    except (httpx.HTTPError, ValueError) as exc:
        emit("media_error", kind=kind, error=str(exc)[:200])
        return None
    results = data.get("results") or []
    if not results:
        return None
    top = results[0]
    return {"kind": "foley" if endpoint.startswith("foley") else "music",
            "name": top.get("name"), "url": top.get("url"),
            "score": top.get("score")}


def voice_list() -> list[str]:
    """The Higgs seed library (cached — it is 900+ names and rarely changes)."""
    if _VOICE_CACHE["voices"] and time.time() - _VOICE_CACHE["at"] < 300:
        return _VOICE_CACHE["voices"]
    try:
        with httpx.Client(timeout=10.0) as c:
            voices = c.get(VOICES_URL).json().get("voices", [])
        if voices:
            _VOICE_CACHE.update(at=time.time(), voices=voices)
    except (httpx.HTTPError, ValueError):
        pass
    return _VOICE_CACHE["voices"]


# When a description says nothing about gender, several seeds tie on tokens
# ("a scottish dwarf" matches both archetype_dwarf_male_scottish and
# archetype_dwarf_female_scottish). Without a tie-break the alphabet decides,
# which picked the female voice every time. Neutral > male > female is an
# arbitrary but *deliberate* default, and saying "female dwarf" still wins.
_GENDER_ORDER = {"neutral": 0, "male": 1, "female": 2}


def _gender_rank(name: str) -> int:
    parts = name.lower().split("_")
    for part, rank in _GENDER_ORDER.items():
        if part in parts:
            return rank
    return 1  # unmarked names sit with "male" rather than being deprioritised


# Filler words must not count as matches: "something that does not exist" once
# resolved to seremet_neutral_A purely because the article "a" matched.
_STOPWORDS = {"the", "and", "for", "with", "that", "this", "does", "not", "you", "your",
              "voice", "speak", "sound", "like", "please", "change", "use", "make",
              "more", "less", "kind", "of", "in", "on", "at", "to", "a", "an", "it",
              "its", "he", "she", "they", "some", "any", "one", "new", "now", "from"}


def clap_voice(description: str) -> str | None:
    """Semantic voice search over the seed banks (899 CLAP vectors across the
    recorded LibriVox/LibriTTS/VCTK/palette banks). This is the good path: the
    banks are real recorded speakers, where the legacy 900+ seeds are the set
    the owner is only keeping around. Returns a seed name Higgs can clone."""
    try:
        with httpx.Client(timeout=200.0) as c:
            data = c.get(f"{MEDIA_URL}/media/voice/search",
                         params={"q": description, "k": 5}).json()
    except (httpx.HTTPError, ValueError):
        return None
    for hit in data.get("results") or []:
        if hit.get("has_seed") and hit.get("seed_path"):
            name = os.path.basename(hit["seed_path"])
            return name[:-4] if name.endswith(".wav") else name
    return None


def resolve_voice(description: str) -> str | None:
    """Best seed for a spoken description ("a scottish dwarf" ->
    archetype_dwarf_male_scottish).

    Tries the CLAP seed-bank search first — "a warm elderly british gentleman"
    is not a filename, and matching words against names cannot answer it. Falls
    back to ranking by how many query words match, then by how *little else*
    the name says, then gender and name for determinism. Nothing sensible ->
    None, and the voice stays put."""
    semantic = clap_voice(description)
    if semantic:
        return semantic
    voices = voice_list()
    query = (description or "").lower().strip()
    tokens = {t for t in re.findall(r"[a-z0-9]+", query)
              if len(t) >= 3 and t not in _STOPWORDS}
    if not voices or not tokens:
        return None
    slug = query.replace(" ", "_")
    ranked = []
    for v in voices:
        vt = set(re.findall(r"[a-z0-9]+", v.lower()))
        matched = len(tokens & vt)
        if not matched:
            continue
        bonus = 3 if (slug and slug in v.lower()) else (3 if v.lower() in query else 0)
        ranked.append((-(matched + bonus), len(vt - tokens), _gender_rank(v), v))
    if not ranked:
        return None
    ranked.sort()
    return ranked[0][3]


PROTOCOL = (
    "\n\nDELEGATION PROTOCOL (use it exactly; it is parsed by the house system)\n"
    "When a request needs real work — code, research, anything slow — do NOT do it yourself and do "
    "NOT pretend to. Emit a line on its own:\n"
    "    @@DELEGATE coder: <one concrete task>\n"
    "or  @@DELEGATE director: <a task needing a considered plan>\n"
    "You may emit more than one. Say in one short sentence that it is underway, then carry on. The "
    "system removes the @@ line before the human sees your reply, and it will show you the result "
    "when it lands. Never invent a result you have not been given. If a result is in the background "
    "state, report it naturally in speech.\n"
    "When the human asks you to speak in a different voice, emit:\n"
    "    @@VOICE: <a description of the voice, e.g. a scottish dwarf, a calm narrator>\n"
    "The system picks the closest voice from the library and switches to it. Confirm briefly; the "
    "line itself is hidden.\n"
    "When the human picks one of the dev-team options you offered them, emit:\n"
    "    @@CHOOSE <item-id> <option-number>\n"
    "The system applies it. Confirm in one short line afterwards.\n"
    "When the human asks to HEAR something — a sound effect or some music — emit:\n"
    "    @@PLAY foley: <what it should sound like, e.g. a sword being drawn, thunder in the distance>\n"
    "    @@PLAY music: <what it should be like, e.g. a cosy tavern, dark tense combat>\n"
    "The sound library is searched by description and played straight away. Say one short line "
    "about it; the directive is hidden. Do not describe the sound at length — they are about to "
    "hear it."
)


BOARD_URL = os.getenv("JARVIS_BOARD_URL", "http://127.0.0.1:8094")


def team_digest(question: str = "") -> str:
    """A compact line about the offline dev team, or "" when there is nothing to say.

    Read from the board service rather than from the database directly: the board
    already owns that view, and the brain should not grow a second opinion about what
    the team is doing.

    Injected only when it is relevant — the human asked about the team, or something
    is genuinely waiting on a decision. Never raises: a missing board must not break
    a reply.
    """
    wants_team = bool(re.search(r"\b(dev ?team|team|architect|plans?|work ?items?|"
                                r"coder|offline team|progress|working on)\b",
                                question or "", re.IGNORECASE))
    try:
        with httpx.Client(timeout=4.0) as c:
            r = c.get(f"{BOARD_URL}/api/board")
            r.raise_for_status()
            d = r.json()
    except Exception:  # noqa: BLE001
        return ""
    needs = d.get("needs") or []
    if not wants_team and not needs:
        return ""
    t = d.get("totals") or {}
    active = d.get("active") or []
    bits = [f"{t.get('verified', 0)} of {t.get('items', 0)} items verified "
            f"across {t.get('plans', 0)} plans"]
    if active:
        bits.append("currently working on "
                    + "; ".join(f"{i['id']} ({i['title']})" for i in active[:2]))
    elif d.get("busy"):
        bits.append("currently working")
    else:
        bits.append("idle")
    line = "Dev team state: " + ", ".join(bits) + "."
    stuck = d.get("options_for") or []
    if stuck:
        line += ("\nSTUCK ITEMS WITH OPTIONS (if the human asks about one, read the "
                 "options out, say which you recommend and why, and offer to apply it. "
                 "If they pick one, emit @@CHOOSE <item> <n> on its own line):")
        for s in stuck[:3]:
            opts = "; ".join(f"{n}. {o.get('label')}"
                             + (" (you recommend this)" if o.get("recommended") else "")
                             for n, o in enumerate(s.get("options", []), 1))
            line += f"\n  {s['id']} — {s['title']}: {opts}"
    if needs:
        line += (" WAITING ON THE HUMAN: "
                 + "; ".join(n.get("what", "") for n in needs[:3])
                 + ". Mention this once, briefly, if it fits the moment — do not nag, "
                   "and never read this line aloud verbatim.")
    return line


def build_conversationalist(sess: dict) -> Agent:
    return Agent(
        name="jarvis",
        prompt=conversational_prompt(),
        config=OpenAIConfig(model=CONVERSATIONAL_MODEL, api_key="local",
                            base_url=f"{CONVERSATIONAL_URL}/v1"),
    )


def conversational_prompt() -> str:
    return (
        f"{load_persona()}\n\n"
        "You are the VOICE — the one the human actually talks to. You are quick, and you stay "
        "quick: you never sit in silence while heavy work happens. Give as much detail as the "
        "question deserves — several paragraphs when that genuinely helps, a sentence when it "
        "does not. This is speech, so keep the STRUCTURE of speech: no headings, no bullet "
        "lists, no markdown; just well-organised prose that a person would say aloud."
        f"{PROTOCOL}"
    )


def stream_conversational(messages: list[dict], sess: dict):
    """Yield the reply as it is written, straight from the actor lane.

    This exists to overlap thinking with speaking: the caller can start talking
    about sentence one while sentence three is still being generated. AG2 is not
    in this path — its tools never executed (see apply_directives), so all it
    contributed here was assembling a prompt, and it cannot stream anyway.

    @@ directive lines are held back rather than spoken: they are instructions
    to the house, not things to say. They are applied once the reply finishes.
    """
    convo = [{"role": "system", "content": conversational_prompt()}]
    convo += to_ag2_messages(messages, sess, conversational=True)
    payload = {"model": CONVERSATIONAL_MODEL, "messages": convo, "stream": True,
               "max_tokens": int(os.getenv("JARVIS_MAX_REPLY_TOKENS", "1200")),
               "temperature": 0.7}
    held: list[str] = []
    buf = ""
    with httpx.Client(timeout=httpx.Timeout(900.0, connect=8.0)) as c:
        with c.stream("POST", f"{CONVERSATIONAL_URL}/v1/chat/completions", json=payload) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0].get("delta", {})
                except (ValueError, KeyError, IndexError):
                    continue
                piece = delta.get("content") or delta.get("reasoning_content") or ""
                if not piece:
                    continue
                buf += piece
                # Emit whole lines only, so a directive can never be half-spoken.
                while "\n" in buf:
                    ln, buf = buf.split("\n", 1)
                    if ln.strip().startswith("@@"):
                        held.append(ln)
                    else:
                        yield ln + "\n"
    tail = buf.rstrip()
    if tail:
        if tail.lstrip().startswith("@@"):
            held.append(tail)
        else:
            yield tail
    if held:
        apply_directives("\n".join(held), sess)


_STREAM_DONE = object()


async def _aiter_stream(messages: list[dict], sess: dict):
    """Run the blocking stream_conversational generator off the event loop and
    yield its pieces as they arrive.

    httpx's sync streaming API is a plain generator, so a worker thread feeds a
    queue and this drains it. Polling every 30 ms is deliberate: it is far below
    the granularity of speech, and it avoids holding a thread-pool slot per open
    stream the way `await asyncio.to_thread(q.get)` would.
    """
    q: queue.Queue = queue.Queue()

    def run():
        try:
            for piece in stream_conversational(messages, sess):
                q.put(piece)
        except Exception as exc:  # noqa: BLE001
            q.put(("__error__", exc))
        finally:
            q.put(_STREAM_DONE)

    threading.Thread(target=run, daemon=True).start()
    while True:
        try:
            item = q.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.03)
            continue
        if item is _STREAM_DONE:
            return
        if isinstance(item, tuple) and item and item[0] == "__error__":
            raise item[1]
        yield item


DIRECTOR_PROTOCOL = (
    "\n\nPROTOCOL (parsed by the house system; use it exactly)\n"
    "To put a plan in front of the human, end your reply with:\n"
    "    @@PLAN\n    <the plan, as numbered steps>\n    @@END\n"
    "That registers the plan and arms the approval gate — do not hand anything to the coder until "
    "the human approves it in their next message. Once approved you will be told so.\n"
    "To hand ONE approved task to the coder, emit a line on its own:\n"
    "    @@HANDOFF: <one concrete task>\n"
    "The system runs it and shows you the report inline. One task at a time; adapt on the result."
)


def build_director(sess: dict) -> Agent:
    prompt = (
        f"{load_persona()}\n\n"
        "You are the DIRECTOR: you shape plans with the human, then — only after an approved plan — "
        "hand tasks to the coder ONE AT A TIME, adapting as results come back. Keep <think> "
        "reasoning internal."
        f"{DIRECTOR_PROTOCOL}"
    )
    return Agent(
        name="director",
        prompt=prompt,
        config=OpenAIConfig(model="director", api_key="local", base_url=f"{DIRECTOR_URL}/v1"),
    )


def apply_directives(text: str, sess: dict) -> str:
    """Act on the conversational @@ directives in a finished reply.

    Separate from apply_protocol because the streaming path collects directive
    lines while the reply is still being written (it must not speak them) and
    only gets to act on them at the end."""
    delegations = DELEGATE_RE.findall(text)
    text = DELEGATE_RE.sub("", text).strip()
    for target, task in delegations:
        start_job(target.lower(), task.strip())
        if not text:
            text = (f"Of course, sir — I've put that in front of the "
                    f"{target.lower()} and will report back shortly.")
    for kind, description in PLAY_RE.findall(text):
        media = find_media(kind.lower(), description.strip())
        text = PLAY_RE.sub("", text).strip()
        if media:
            sess["media"] = media
            emit("media", name=media["name"], url=media["url"],
                 score=media.get("score"), source=media["kind"])
        else:
            emit("media_error", kind=kind, query=description[:120])
            text = f"{text}\n\n_[nothing in the {kind} library matched “{description}”]_".strip()
    choose = CHOOSE_RE.search(text)
    if choose:
        text = CHOOSE_RE.sub("", text).strip()
        pick_item, pick_n = choose.group(1), int(choose.group(2))
        try:
            import devteam
            out: list[str] = []
            result = devteam.apply_option(
                pick_item, pick_n, log=lambda m="": out.append(str(m)))
            note = {"reopened": "that is applied and the coder is retrying now",
                    "respecified": "that is applied and the item is back in the queue",
                    "recorded": "noted — that one needs you to do it by hand",
                    "failed": "the change did not apply cleanly"}.get(result, result)
            text = (text + f"\n\n_[{pick_item} option {pick_n} → {note}]_").strip()
            emit("choose", item=pick_item, option=pick_n, result=result)
        except Exception as exc:  # noqa: BLE001
            text = (text + f"\n\n_[could not apply option {pick_n} to {pick_item}: "
                           f"{exc}]_").strip()
    voice_match = VOICE_RE.search(text)
    if voice_match:
        wanted = voice_match.group(1).strip()
        chosen = resolve_voice(wanted)
        text = VOICE_RE.sub("", text).strip()
        if chosen:
            sess["voice"] = chosen
            text = f"{text}\n\n_[voice → {chosen}]_".strip()
        else:
            text = f"{text}\n\n_[no voice matches “{wanted}”]_".strip()
    return re.sub(r"\s*@@\s*$", "", text, flags=re.MULTILINE).strip()


def apply_protocol(reply: str, sess: dict, conversational: bool) -> str:
    """Turn the model's @@ directives into real work, and return what the human
    sees (directives stripped, handoff results inlined)."""
    if conversational:
        return apply_directives(reply, sess) or "Of course, sir — here it is."
    plan_match = PLAN_RE.search(reply)
    if plan_match:
        plan = plan_match.group(1).strip()
        sess["plan"] = plan
        sess["approved"] = False
        sess["failures"] = 0
        sess["log"] = []
        emit("plan", plan=plan)
        reply = PLAN_RE.sub("", reply).strip()

    def do_handoff(match):
        if not sess.get("approved"):
            emit("coder_refused", reason="no approved plan")
            return "[refused: no approved plan — present it with @@PLAN and wait for the human]"
        task = _strip_think(match.group(1))
        if not task:
            return "[refused: empty task]"
        emit("coder_start", task=task[:400])
        try:
            result = _chat(CODER_URL, "coder", CODER_SYSTEM, task, max_tokens=2000)
            ok = "FAILED" not in result.upper()[:400]
        except Exception as exc:  # noqa: BLE001
            result, ok = f"transport error: {exc}", False
        sess["log"].append({"task": task[:400], "ok": ok, "reply": result[:800], "at": time.time()})
        emit("coder_done", task=task[:200], ok=ok, reply=result[:600])
        if ok:
            sess["failures"] = 0
            return f"[coder OK]\n{result}"
        sess["failures"] += 1
        if sess["failures"] > CODER_FAILURE_LIMIT:
            sess["failures"] = 0
            log = "\n".join(f"- {e['task'][:120]} | {e['reply'][:200]}" for e in sess["log"][-6:])
            return (f"[INTERVENE: the coder exceeded the failure limit — stop and re-plan with the "
                    f"human. Log:\n{log}]")
        return (f"[coder FAILED (attempt {sess['failures']}/{CODER_FAILURE_LIMIT}) — retry, refine, "
                f"or continue]\n{result}")

    return HANDOFF_RE.sub(do_handoff, reply).strip()


def memory_block(question: str, max_facts: int = 4) -> str:
    """Curated memory for the reply prompt: verified facts first, then the archive,
    and only then the raw turn log.

    Facts cost one short line each, so this is both cheaper and sharper than the
    turn excerpts it replaces: a fact is what the house decided, whereas a turn
    excerpt is merely something that was once said.

    Never raises — this sits on the reply path, and a memory miss must degrade to
    "Jarvis remembers nothing" rather than "Jarvis is down".
    """
    lines: list[str] = []
    try:
        import db as _mem  # same directory; stdlib-only module
        conn = _mem.open_db()
        facts = _mem.search_facts(conn, question, limit=max_facts)
        chunks = _mem.search_archive(conn, question, limit=1) if len(facts) < max_facts else []
    except Exception as exc:  # noqa: BLE001
        print(f"[jarvis-brain] memory lookup failed: {exc}", flush=True)
        facts, chunks = [], []

    if facts:
        lines.append("What you already know (your verified memory — use only what is relevant, "
                     "do not read this list aloud):")
        for f in facts:
            tag = "/".join(p for p in (f["entity"], f["topic"]) if p)
            lines.append(f"- {f['statement']}" + (f"  [{tag}]" if tag else ""))
    if chunks:
        lines.append("Relevant past discussion:")
        for c in chunks:
            lines.append(f"- {c['title']}: {(c['text'] or '')[:220]}")

    if not lines:
        # Nothing curated matched — fall back to the old raw-log behaviour so
        # long-running conversations do not lose their only thread of continuity.
        hits = recall(question, limit=3)
        if hits:
            lines.append("Things said before that may bear on this (use only what is "
                         "genuinely relevant; do not read this list aloud):")
            for h in hits:
                lines.append(f"- [{time.strftime('%Y-%m-%d', time.localtime(h['ts']))} "
                             f"{h['role']}] {h['content'][:200]}")
    return "\n".join(lines)


def to_ag2_messages(messages: list[dict], sess: dict, conversational: bool) -> list[dict]:
    out = [m for m in messages if m.get("role") in ("user", "assistant", "system") and m.get("content")]
    if conversational:
        digest = job_digest()
        if digest:
            out.append({"role": "system",
                        "content": f"Background work state (do not read this aloud; use it):\n{digest}"})
        # Memory, retrieved rather than held: Kunou's window is 8k, so recall
        # what is relevant to *this* turn instead of carrying the past around.
        # Terms are picked from the actual question, so quiet turns pull
        # nothing and noisy turns pull only a few excerpts.
        question = next((m.get("content") or "" for m in reversed(out)
                         if m.get("role") == "user"), "")
        if question and len(question) > 12:
            # Kept small on purpose: every token here is prompt the model must
            # read before it can say a word, and it is read BEFORE the reply
            # starts. Measured: a 4x400-char memory block cost ~2.3s of
            # time-to-first-token. Curated facts are one short line each, so this
            # is cheaper than the excerpts it replaces.
            block = memory_block(question)
            if block:
                out.append({"role": "system", "content": block})
        team = team_digest(question)
        if team:
            out.append({"role": "system", "content": team})
        return out
    if sess.get("approved") and sess.get("plan"):
        out.append({"role": "system",
                    "content": f"The human APPROVED your plan. You may now hand tasks to the coder, "
                               f"one at a time.\nPlan:\n{sess['plan']}"})
    elif sess.get("plan"):
        out.append({"role": "system",
                    "content": "A plan is awaiting the human's verdict. If their last message approves "
                               "it, proceed; otherwise treat it as a revision request."})
    return out


def searx(query: str, category: str = "general", limit: int = None) -> list[dict]:
    """Search via the self-hosted SearxNG. Returns [{title, url, snippet}]."""
    limit = limit or SEARCH_RESULTS
    try:
        with httpx.Client(timeout=45.0) as c:
            r = c.get(SEARXNG_URL, params={"q": query, "format": "json",
                                           "categories": category, "language": "en"})
            r.raise_for_status()
            results = r.json().get("results", [])
    except (httpx.HTTPError, ValueError) as exc:
        emit("web_error", query=query[:120], error=str(exc)[:200])
        return []
    out = []
    for item in results[:limit]:
        if not item.get("url"):
            continue
        out.append({"title": (item.get("title") or item["url"])[:140],
                    "url": item["url"],
                    "snippet": (item.get("content") or "")[:400]})
    return out


def wiki_lookup(term: str) -> tuple[str, list[dict]]:
    """The local wiki service on cerebro (7M abstracts) — the house's own
    encyclopedia tier, consulted before the open web for reference material."""
    try:
        with httpx.Client(timeout=20.0) as c:
            d = c.get(f"{WIKI_URL}/lookup", params={"title": term}).json()
    except (httpx.HTTPError, ValueError):
        return "", []
    text = (d.get("extract") or d.get("abstract") or d.get("text") or "").strip()
    if not text:
        return "", []
    return text, [{"title": f"Wiki: {d.get('title', term)}", "url": f"wiki://{d.get('title', term)}"}]


def _local_answer(question: str, context: str, sources: list[dict]) -> str:
    """Read the fetched material with the LOCAL conversationalist."""
    system = (f"{load_persona()}\n\nYou are answering from material just fetched for you. "
              "Use it rather than your memory; say plainly if it is thin or contradictory. "
              "Cite the sources inline as [1], [2] … matching the numbered list. "
              "Keep it to a few sentences.")
    numbered = "\n".join(f"[{i+1}] {s['title']} — {s['url']}" for i, s in enumerate(sources))
    body = f"Question: {question}\n\nSources:\n{numbered}\n\nFetched content:\n{context[:6000]}"
    return _strip_think(_chat(CONVERSATIONAL_URL, CONVERSATIONAL_MODEL, system, body, max_tokens=900))


def run_web(messages: list[dict], category: str = "general"):
    """Search, then answer locally. Returns (reply, sources, timings)."""
    question = next((m.get("content") or "" for m in reversed(messages)
                     if m.get("role") == "user"), "").strip()
    if not question:
        return "I did not catch the question, sir.", [], {}
    emit("web_start", query=question[:200], where="news" if category == "news" else "web")

    t0 = time.time()
    hits = searx(question, category=category)
    search_s = time.time() - t0
    if not hits:
        return ("I could not reach the search engines just now, sir — SearxNG returned nothing.",
                [], {"search": round(search_s, 2)})
    context = "\n\n".join(f"{h['title']}\n{h['url']}\n{h['snippet']}" for h in hits)
    t1 = time.time()
    text = _local_answer(question, context, hits)
    answer_s = time.time() - t1
    timings = {"search": round(search_s, 2), "answer": round(answer_s, 2),
               "total": round(search_s + answer_s, 2), "results": len(hits)}
    emit("web_done", query=question[:120], sources=hits[:6], timings=timings)
    return text, hits[:6], timings


def run_wiki(messages: list[dict]):
    """Answer from the local wiki service. Returns (reply, sources, timings)."""
    question = next((m.get("content") or "" for m in reversed(messages)
                     if m.get("role") == "user"), "").strip()
    emit("web_start", query=question[:200], where="wiki")
    t0 = time.time()
    text, sources = wiki_lookup(question)
    wiki_s = time.time() - t0
    if not text:
        return run_web(messages)          # fall through to the open web
    t1 = time.time()
    answer = _local_answer(question, text, sources)
    timings = {"wiki": round(wiki_s, 2), "answer": round(time.time() - t1, 2)}
    emit("web_done", query=question[:120], sources=sources, timings=timings)
    return answer, sources, timings


async def run_agent(messages: list[dict], sess: dict, conversational: bool) -> str:
    agent = build_conversationalist(sess) if conversational else build_director(sess)
    reply = await agent.ask(to_ag2_messages(messages, sess, conversational))
    text = getattr(reply, "content", None)
    if callable(text):
        text = await text()
    return apply_protocol(_strip_think(str(text if text is not None else reply)), sess, conversational)


# ---------------------------------------------------------------- app

app = FastAPI(title="jarvis brain (AG2 as OpenAI models)")


@app.get("/health")
async def health():
    with STATE_LOCK:
        sessions, jobs = len(SESSIONS), len(JOBS)
        running = sum(1 for j in JOBS.values() if j["status"] == "running")
    return {"status": "ok", "models": [MODEL_NAME, DIRECTOR_MODEL_NAME], "sessions": sessions,
            "jobs": {"total": jobs, "running": running},
            "backends": {"conversational": CONVERSATIONAL_URL, "director": DIRECTOR_URL,
                         "coder": CODER_URL}}


@app.get("/v1/models")
async def models():
    now = int(time.time())
    return {"object": "list", "data": [
        {"id": MODEL_NAME, "object": "model", "created": now, "owned_by": "jarvis"},
        {"id": DIRECTOR_MODEL_NAME, "object": "model", "created": now, "owned_by": "jarvis"},
        {"id": WEB_MODEL_NAME, "object": "model", "created": now, "owned_by": "jarvis"},
        {"id": NEWS_MODEL_NAME, "object": "model", "created": now, "owned_by": "jarvis"},
        {"id": WIKI_MODEL_NAME, "object": "model", "created": now, "owned_by": "jarvis"},
    ]}


@app.get("/recall")
async def recall_endpoint(q: str, limit: int = 8):
    """Search everything ever said here. The seed of Jarvis's memory: rather
    than holding the past in the window, the past is looked up on demand."""
    hits = recall(q, limit)
    return {"query": q, "hits": len(hits), "results": hits}


@app.get("/events")
async def events(request: Request):
    sink: queue.Queue = queue.Queue(maxsize=256)
    with STATE_LOCK:
        EVENT_SINKS.add(sink)

    async def gen():
        try:
            yield f"data: {json.dumps({'kind': 'hello', 'ts': time.time()})}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = sink.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.25)
                    continue
                yield f"data: {json.dumps(event)}\n\n"
        finally:
            with STATE_LOCK:
                EVENT_SINKS.discard(sink)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/approve")
async def approve(payload: dict):
    key = payload.get("session")
    approved = bool(payload.get("approved"))
    with STATE_LOCK:
        if key and key in SESSIONS:
            SESSIONS[key]["approved"] = approved
            plan = SESSIONS[key]["plan"]
            if not approved:
                SESSIONS[key]["plan"] = ""
        else:
            pending = [s for s in SESSIONS.values() if s["plan"] and not s["approved"]]
            if not pending:
                return JSONResponse({"status": "no pending plan"}, status_code=404)
            sess = max(pending, key=lambda s: s["touched"])
            sess["approved"] = approved
            plan = sess["plan"]
            if not approved:
                sess["plan"] = ""
    emit("approval", approved=approved, plan=plan[:400])
    return {"status": "approved" if approved else "rejected", "plan": plan[:400]}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages") or []
    stream = bool(body.get("stream"))
    requested = (body.get("model") or MODEL_NAME).lower()
    web = WEB_MODEL_NAME in requested
    news = NEWS_MODEL_NAME in requested
    wiki = WIKI_MODEL_NAME in requested
    fetching = web or news or wiki
    conversational = (DIRECTOR_MODEL_NAME not in requested) and not fetching
    sess = session_for(messages)
    # epub = "keep nothing". Set by the panel's incognito toggle; OWUI callers
    # simply never send it, so their conversations are kept by default.
    ephemeral = bool(body.get("ephemeral"))

    last_user = next((m.get("content") or "" for m in reversed(messages) if m.get("role") == "user"), "")
    if not conversational and not fetching and sess.get("plan") and not sess.get("approved") \
            and looks_like_approval(last_user):
        sess["approved"] = True
        emit("approval", approved=True, plan=sess["plan"][:400])

    def fetch_kind():
        if news:
            return run_web(messages, category="news")
        if wiki:
            return run_wiki(messages)
        return run_web(messages)

    if not stream:
        sources: list[dict] = []
        timings: dict = {}
        if fetching:
            text, sources, timings = await asyncio.to_thread(fetch_kind)
        else:
            text = await run_agent(messages, sess, conversational)
        body_out = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}", "object": "chat.completion",
            "created": int(time.time()), "model": requested,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": text}}],
        }
        # Non-standard fields: clients that know about them (the docks/live
        # pages) follow a voice change and show sources; everyone else ignores.
        voice = sess.pop("voice", None)
        if voice:
            body_out["voice"] = voice
        media = sess.pop("media", None)
        if media:
            body_out["media"] = media
        if sources:
            body_out["sources"] = sources
        if timings:
            body_out["timings"] = timings
        # Keep the exchange unless this conversation is incognito.
        record(sess.get("key", ""), requested, "user", last_user, ephemeral)
        record(sess.get("key", ""), requested, "assistant", text, ephemeral)
        return JSONResponse(body_out)

    async def gen():
        cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        def frame_of(delta: dict, finish=None) -> str:
            payload = {"id": cid, "object": "chat.completion.chunk", "created": created,
                       "model": requested,
                       "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            return f"data: {json.dumps(payload)}\n\n"

        # The conversationalist streams for real: the client hears the first
        # sentence while the rest is still being written. This is the whole
        # point of the split — speech overlaps generation instead of waiting
        # for it.
        if conversational and not fetching:
            yield frame_of({"role": "assistant", "content": ""})
            said = []
            try:
                async for piece in _aiter_stream(messages, sess):
                    said.append(piece)
                    yield frame_of({"content": piece})
            except Exception as exc:  # noqa: BLE001
                yield frame_of({"content": f"\n\n_[trouble: {type(exc).__name__}: {exc}]_"})
            text = "".join(said).strip()
            if not text:
                text = "Of course, sir — here it is."
                yield frame_of({"content": text})
            record(sess.get("key", ""), requested, "user", last_user, ephemeral)
            record(sess.get("key", ""), requested, "assistant", text, ephemeral)
            media = sess.pop("media", None)
            if media:
                yield "data: " + json.dumps({"media": media}) + "\n\n"
            voice = sess.pop("voice", None)
            if voice:
                yield "data: " + json.dumps({"voice": voice}) + "\n\n"
            yield frame_of({}, finish="stop")
            yield "data: [DONE]\n\n"
            return

        out: queue.Queue = queue.Queue()

        def worker():
            try:
                if fetching:
                    text, srcs, tm = fetch_kind()
                    if srcs:                      # sources first: the stream ends at "final"
                        out.put(("sources", srcs))
                    if tm:
                        out.put(("timings", tm))
                    out.put(("final", text))
                else:
                    out.put(("final", asyncio.run(run_agent(messages, sess, conversational))))
            except Exception as exc:  # noqa: BLE001
                out.put(("error", f"{type(exc).__name__}: {exc}"))

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        def frame(delta: dict, finish=None) -> str:
            payload = {"id": cid, "object": "chat.completion.chunk", "created": created,
                       "model": requested,
                       "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            return f"data: {json.dumps(payload)}\n\n"

        yield frame({"role": "assistant", "content": ""})
        if sess.get("media"):
            yield "data: " + json.dumps({"media": sess.pop("media")}) + "\n\n"
        started = time.time()
        while True:
            try:
                kind, payload = out.get_nowait()
            except queue.Empty:
                if not thread.is_alive():
                    break
                await asyncio.sleep(0.2)
                if time.time() - started > 20:
                    started = time.time()
                    yield ": keepalive\n\n"
                continue
            if kind == "final":
                record(sess.get("key", ""), requested, "user", last_user, ephemeral)
                record(sess.get("key", ""), requested, "assistant", payload, ephemeral)
                for word in re.findall(r"\S+\s*", payload):
                    yield frame({"content": word})
                    await asyncio.sleep(0.012)
                break
            if kind == "sources":
                yield "data: " + json.dumps({"model": requested, "sources": payload}) + "\n\n"
                continue
            if kind == "timings":
                yield "data: " + json.dumps({"model": requested, "timings": payload}) + "\n\n"
                continue
            yield frame({"content": f"\n\n_[{payload}]_\n\n"})
        yield frame({}, finish="stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8092")))
