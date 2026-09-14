"""The Jarvis database — one SQLite file, four tiers.

Design notes live in
`~/manifest-paradigm/executive-assisstant/2-jarvis-database-and-devteam-plan.md`.

    turns           raw session log (owned by brain.py; kept as-is)
    sessions        one row per conversation
    facts           the epistemic pool: decisions, preferences, configs
    archive_chunks  chunked debate archive (rejected options stay findable)
    plans           a plan + its approval state
    work_items      the unit of devteam work
    runs            every model call the devteam makes
    cursors         high-water marks (memory extractor)

Two rules this module exists to enforce:

1. **Supersede, never silently overwrite.** A fact that revises another leaves the old
   row in place, marked `superseded`, with `supersedes` pointing forward from the new
   row. "What did we believe in August?" stays answerable.
2. **Status is earned.** A work item reaches `verified` because a command exited 0,
   not because a model said it was done.

Used by brain.py (read path), memory.py (extraction) and devteam.py (execution).
stdlib only — this has to run under the plain system python as well as the venv.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time

DB_PATH = os.getenv("JARVIS_DB", "/var/home/admin/jarvis/conversations.db")

# Work-item status vocabulary — kept identical to the AI-devteam harness contract
# (~/AI-devteam/AGENTS.md) so the two can read each other's items.
STATUSES = ("pending", "in-progress", "implemented", "verified", "complete",
            "blocked", "rejected")

SCHEMA = """
-- ---------------------------------------------------------------- sessions
CREATE TABLE IF NOT EXISTS sessions (
    key       TEXT PRIMARY KEY,
    started   REAL,
    touched   REAL,
    model     TEXT,
    title     TEXT,
    turns     INTEGER DEFAULT 0,
    archived  INTEGER DEFAULT 0
);

-- ---------------------------------------------------------------- facts
CREATE TABLE IF NOT EXISTS facts (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    created        REAL NOT NULL,
    updated        REAL NOT NULL,
    entity         TEXT,                -- cerebro_system | user | rental_prop | ...
    topic          TEXT,                -- memory_architecture | gmail_oauth | ...
    statement      TEXT NOT NULL,
    kind           TEXT,                -- fact | preference | config | decision
    status         TEXT NOT NULL DEFAULT 'active',   -- active | superseded | rejected
    confidence     REAL NOT NULL DEFAULT 1.0,
    supersedes     INTEGER,             -- id of the fact this one replaces
    source_session TEXT,
    source_turn    INTEGER,
    source_url     TEXT
);
CREATE INDEX IF NOT EXISTS facts_status  ON facts(status);
CREATE INDEX IF NOT EXISTS facts_entity  ON facts(entity, topic);

CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
    statement, entity, topic, content=facts, content_rowid=id);

CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
    INSERT INTO facts_fts(rowid, statement, entity, topic)
    VALUES (new.id, new.statement, new.entity, new.topic);
END;
CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, statement, entity, topic)
    VALUES ('delete', old.id, old.statement, old.entity, old.topic);
END;
CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, statement, entity, topic)
    VALUES ('delete', old.id, old.statement, old.entity, old.topic);
    INSERT INTO facts_fts(rowid, statement, entity, topic)
    VALUES (new.id, new.statement, new.entity, new.topic);
END;

-- ---------------------------------------------------------------- archive
CREATE TABLE IF NOT EXISTS archive_chunks (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    created    REAL NOT NULL,
    session    TEXT,
    ts_from    REAL,
    ts_to      REAL,
    title      TEXT,
    text       TEXT NOT NULL,
    source_url TEXT
);
CREATE VIRTUAL TABLE IF NOT EXISTS archive_fts USING fts5(
    title, text, content=archive_chunks, content_rowid=id);

CREATE TRIGGER IF NOT EXISTS archive_ai AFTER INSERT ON archive_chunks BEGIN
    INSERT INTO archive_fts(rowid, title, text) VALUES (new.id, new.title, new.text);
END;
CREATE TRIGGER IF NOT EXISTS archive_ad AFTER DELETE ON archive_chunks BEGIN
    INSERT INTO archive_fts(archive_fts, rowid, title, text)
    VALUES ('delete', old.id, old.title, old.text);
END;
CREATE TRIGGER IF NOT EXISTS archive_au AFTER UPDATE ON archive_chunks BEGIN
    INSERT INTO archive_fts(archive_fts, rowid, title, text)
    VALUES ('delete', old.id, old.title, old.text);
    INSERT INTO archive_fts(rowid, title, text) VALUES (new.id, new.title, new.text);
END;

-- ---------------------------------------------------------------- plans
CREATE TABLE IF NOT EXISTS plans (
    id          TEXT PRIMARY KEY,
    created     REAL NOT NULL,
    updated     REAL,
    title       TEXT,
    brief       TEXT,
    body        TEXT,
    status      TEXT NOT NULL DEFAULT 'draft',   -- draft | proposed | approved | done | abandoned
    approved    INTEGER NOT NULL DEFAULT 0,
    approved_at REAL,
    decided_md  TEXT
);

-- ---------------------------------------------------------------- work items
CREATE TABLE IF NOT EXISTS work_items (
    id         TEXT PRIMARY KEY,      -- e.g. JV-001
    plan_id    TEXT,
    created    REAL NOT NULL,
    updated    REAL,
    ordinal    INTEGER DEFAULT 0,
    title      TEXT NOT NULL,
    detail     TEXT,
    owner      TEXT,                  -- coder | director
    status     TEXT NOT NULL DEFAULT 'pending',
    depends_on TEXT,                  -- comma-separated item ids
    workspace  TEXT,                  -- path the work happens in
    verify     TEXT,                  -- shell command; exit 0 == verified
    artifact   TEXT,                  -- commit/ref produced
    evidence   TEXT,                  -- last verify output (trimmed)
    notes      TEXT
);
CREATE INDEX IF NOT EXISTS work_items_status ON work_items(status);

-- ---------------------------------------------------------------- runs
CREATE TABLE IF NOT EXISTS runs (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      REAL NOT NULL,
    plan_id TEXT,
    item_id TEXT,
    role    TEXT,                     -- director | coder | verifier
    engine  TEXT,                     -- lane URL / model
    prompt  TEXT,
    output  TEXT,
    ok      INTEGER,
    ms      INTEGER
);
CREATE INDEX IF NOT EXISTS runs_item ON runs(item_id);

-- ---------------------------------------------------------------- cursors
CREATE TABLE IF NOT EXISTS cursors (
    name  TEXT PRIMARY KEY,
    value TEXT,
    ts    REAL
);
"""


def connect(path: str | None = None) -> sqlite3.Connection:
    """Open the DB. WAL + a busy timeout because brain.py writes concurrently."""
    conn = sqlite3.connect(path or DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=15000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# Columns added after the first release. CREATE TABLE IF NOT EXISTS will not add a
# column to a table that already exists, so they are applied explicitly.
MIGRATIONS = (
    ("plans", "design", "TEXT"),        # the canonical design the whole plan must honour
    ("plans", "reviewed_at", "REAL"),   # when the architect last sanity-checked it
    ("plans", "consistent", "INTEGER"), # did that review find contradictions?
    # The architect's options for resolving a stuck item, as JSON — kept structured
    # rather than as prose so the panels can render one button per option.
    ("work_items", "options", "TEXT"),
)


def _migrate(conn: sqlite3.Connection) -> None:
    for table, column, coltype in MIGRATIONS:
        have = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in have:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")
    conn.commit()


def init(conn: sqlite3.Connection) -> sqlite3.Connection:
    """Create every table. Idempotent — safe to call on every start."""
    conn.executescript(SCHEMA)
    _migrate(conn)
    conn.commit()
    return conn


def open_db(path: str | None = None) -> sqlite3.Connection:
    return init(connect(path))


# ------------------------------------------------------------------ facts

def add_fact(conn: sqlite3.Connection, statement: str, *, entity: str | None = None,
             topic: str | None = None, kind: str = "fact", confidence: float = 1.0,
             supersedes: int | None = None, source_session: str | None = None,
             source_turn: int | None = None, source_url: str | None = None,
             status: str = "active") -> int:
    """Insert a fact; if it supersedes another, retire that one in the same transaction."""
    now = time.time()
    with conn:
        cur = conn.execute(
            "INSERT INTO facts (created, updated, entity, topic, statement, kind, status,"
            " confidence, supersedes, source_session, source_turn, source_url)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (now, now, entity, topic, statement.strip(), kind, status, confidence,
             supersedes, source_session, source_turn, source_url))
        if supersedes:
            conn.execute("UPDATE facts SET status='superseded', updated=? WHERE id=?",
                         (now, supersedes))
    return int(cur.lastrowid)


def active_facts(conn: sqlite3.Connection, *, entity: str | None = None,
                 topic: str | None = None, limit: int = 50) -> list[sqlite3.Row]:
    q = "SELECT * FROM facts WHERE status='active'"
    args: list = []
    if entity:
        q += " AND entity=?"
        args.append(entity)
    if topic:
        q += " AND topic=?"
        args.append(topic)
    q += " ORDER BY updated DESC LIMIT ?"
    args.append(limit)
    return list(conn.execute(q, args))


def search_facts(conn: sqlite3.Connection, query: str, limit: int = 8,
                 status: str = "active") -> list[sqlite3.Row]:
    """FTS over the fact pool, precision-filtered. Returns [] on a malformed query
    rather than raising — callers are on the live reply path."""
    terms = content_terms(query)
    if not terms:
        return []
    try:
        rows = conn.execute(
            "SELECT f.* FROM facts_fts x JOIN facts f ON f.id = x.rowid"
            " WHERE facts_fts MATCH ? AND f.status=? ORDER BY rank LIMIT ?",
            (" OR ".join(terms), status, limit * 4)).fetchall()
    except sqlite3.Error:
        return []
    return _rank(rows, terms, limit, ("statement", "entity", "topic"))


def find_similar_fact(conn: sqlite3.Connection, statement: str, entity: str | None = None,
                      topic: str | None = None, limit: int = 3,
                      threshold: float = 0.6) -> sqlite3.Row | None:
    """Best active match for consolidation.

    Deliberately lexical (token overlap), not semantic: this runs on the write path
    for every candidate fact, and a wrong merge is worse than a duplicate that a
    later pass can catch. Entity/topic must agree when supplied.
    """
    want = set(_words(statement))
    if not want:
        return None
    best, best_score = None, 0.0
    q = "SELECT * FROM facts WHERE status='active'"
    args: list = []
    if entity:
        q += " AND entity=?"
        args.append(entity)
    if topic:
        q += " AND topic=?"
        args.append(topic)
    for row in conn.execute(q + " LIMIT 500", args):
        have = set(_words(row["statement"]))
        if not have:
            continue
        score = len(want & have) / len(want | have)
        if score > best_score:
            best, best_score = row, score
    return best if best_score >= threshold else None


def _words(text: str) -> list[str]:
    import re
    return [w.lower() for w in re.findall(r"[A-Za-z0-9_]{3,}", text or "")]


STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "does", "did", "not", "you", "your",
    "are", "was", "were", "has", "have", "had", "but", "can", "could", "would", "should",
    "what", "when", "where", "which", "who", "why", "how", "about", "into", "from",
    "they", "them", "then", "than", "there", "here", "his", "her", "its", "our", "their",
    "will", "shall", "may", "might", "must", "been", "being", "get", "got", "make", "made",
    "use", "used", "using", "any", "all", "some", "more", "most", "other", "such", "only",
    "own", "same", "too", "very", "just", "now", "also", "out", "off", "over", "under",
    "again", "once", "tell", "say", "said", "know", "want", "need", "like", "well", "yes",
}


def content_terms(text: str) -> list[str]:
    """Query words that actually carry meaning.

    Without this an OR'd FTS match on a natural-language question matches almost
    every row in the pool ('the', 'what', 'did'), which is worse than returning
    nothing: the model is handed confident-looking, irrelevant "memory".
    """
    return [w for w in _words(text) if w not in STOPWORDS]


def _rank(rows, terms: list[str], limit: int, keys: tuple[str, ...]):
    """Keep only rows sharing a content term with the query, best overlap first.

    FTS5's own ranking rewards documents matching many of the OR'd terms, but it
    will still hand back a row that matched only one common word. This is the
    precision gate that keeps 'memory backend' from returning music-generator
    facts just because both contain 'the'.
    """
    scored = []
    for i, r in enumerate(rows):
        hay = " ".join(str(r[k] or "") for k in keys if k in r.keys()).lower()
        score = sum(1 for t in terms if t in hay)
        if score:
            scored.append((score, -i, r))       # -i keeps FTS order on ties
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [r for _, _, r in scored[:limit]]


# ------------------------------------------------------------------ archive

def add_chunk(conn: sqlite3.Connection, text: str, *, title: str | None = None,
              session: str | None = None, ts_from: float | None = None,
              ts_to: float | None = None, source_url: str | None = None) -> int:
    with conn:
        cur = conn.execute(
            "INSERT INTO archive_chunks (created, session, ts_from, ts_to, title, text, source_url)"
            " VALUES (?,?,?,?,?,?,?)",
            (time.time(), session, ts_from, ts_to, title, text.strip(), source_url))
    return int(cur.lastrowid)


def search_archive(conn: sqlite3.Connection, query: str, limit: int = 5) -> list[sqlite3.Row]:
    terms = content_terms(query)
    if not terms:
        return []
    try:
        rows = conn.execute(
            "SELECT a.* FROM archive_fts x JOIN archive_chunks a ON a.id = x.rowid"
            " WHERE archive_fts MATCH ? ORDER BY rank LIMIT ?",
            (" OR ".join(terms), limit * 4)).fetchall()
    except sqlite3.Error:
        return []
    return _rank(rows, terms, limit, ("title", "text"))


# ------------------------------------------------------------------ plans

def create_plan(conn: sqlite3.Connection, plan_id: str, title: str, brief: str,
                body: str = "") -> str:
    now = time.time()
    with conn:
        # Not INSERT OR REPLACE: re-proposing a plan must not silently discard an
        # approval the owner already gave.
        conn.execute(
            "INSERT INTO plans (id, created, updated, title, brief, body, status)"
            " VALUES (?,?,?,?,?,?,'proposed')"
            " ON CONFLICT(id) DO UPDATE SET updated=excluded.updated, title=excluded.title,"
            "  brief=excluded.brief, body=excluded.body",
            (plan_id, now, now, title, brief, body))
    return plan_id


def approve_plan(conn: sqlite3.Connection, plan_id: str, decided_md: str | None = None) -> None:
    with conn:
        conn.execute("UPDATE plans SET approved=1, approved_at=?, status='approved', updated=?"
                     " WHERE id=?", (time.time(), time.time(), plan_id))


def get_plan(conn: sqlite3.Connection, plan_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM plans WHERE id=?", (plan_id,)).fetchone()


def set_plan_review(conn: sqlite3.Connection, plan_id: str, design: str,
                    consistent: bool) -> None:
    """Record the architect's consistency review and the canonical design.

    The design string is the contract every work item is implemented against — it is
    what stops early items and late items drifting into two incompatible systems.
    """
    with conn:
        conn.execute("UPDATE plans SET design=?, consistent=?, reviewed_at=?, updated=?"
                     " WHERE id=?",
                     (design, 1 if consistent else 0, time.time(), time.time(), plan_id))


# ------------------------------------------------------------------ work items

def add_item(conn: sqlite3.Connection, item: dict) -> str:
    """Add or update a work item. `id`, `title` required."""
    now = time.time()
    item = dict(item)
    item.setdefault("status", "pending")
    if item["status"] not in STATUSES:
        raise ValueError(f"bad status {item['status']!r}; expected one of {STATUSES}")
    deps = item.get("depends_on")
    if isinstance(deps, (list, tuple)):
        deps = ",".join(str(d) for d in deps)

    # Item ids are a global primary key but planners number their items locally
    # (JV-001, JV-002...), so a second plan would silently overwrite the first.
    # Loud failure beats quiet data loss.
    existing = conn.execute("SELECT plan_id FROM work_items WHERE id=?",
                            (item["id"],)).fetchone()
    if existing is not None and existing["plan_id"] != item.get("plan_id"):
        raise ValueError(
            f"item id {item['id']!r} already belongs to plan {existing['plan_id']!r}; "
            f"refusing to reassign it to {item.get('plan_id')!r}")

    with conn:
        conn.execute(
            "INSERT INTO work_items (id, plan_id, created, updated, ordinal, title, detail,"
            " owner, status, depends_on, workspace, verify, artifact, evidence, notes)"
            " VALUES (:id,:plan_id,:created,:updated,:ordinal,:title,:detail,:owner,:status,"
            ":depends_on,:workspace,:verify,:artifact,:evidence,:notes)"
            " ON CONFLICT(id) DO UPDATE SET"
            "  updated=excluded.updated, ordinal=excluded.ordinal, title=excluded.title,"
            "  detail=excluded.detail, owner=excluded.owner, status=excluded.status,"
            "  depends_on=excluded.depends_on, workspace=excluded.workspace,"
            "  verify=excluded.verify, artifact=excluded.artifact,"
            "  evidence=excluded.evidence, notes=excluded.notes",
            {"id": item["id"], "plan_id": item.get("plan_id"), "created": now, "updated": now,
             "ordinal": item.get("ordinal", 0), "title": item["title"],
             "detail": item.get("detail"), "owner": item.get("owner"),
             "status": item["status"], "depends_on": deps, "workspace": item.get("workspace"),
             "verify": item.get("verify"), "artifact": item.get("artifact"),
             "evidence": item.get("evidence"), "notes": item.get("notes")})
    return item["id"]


def list_items(conn: sqlite3.Connection, plan_id: str | None = None,
               status: str | None = None) -> list[sqlite3.Row]:
    q, args = "SELECT * FROM work_items WHERE 1=1", []
    if plan_id:
        q += " AND plan_id=?"
        args.append(plan_id)
    if status:
        q += " AND status=?"
        args.append(status)
    return list(conn.execute(q + " ORDER BY ordinal, id", args))


def _deps_satisfied(conn: sqlite3.Connection, row: sqlite3.Row) -> bool:
    deps = [d.strip() for d in (row["depends_on"] or "").split(",") if d.strip()]
    for dep in deps:
        other = conn.execute("SELECT status FROM work_items WHERE id=?", (dep,)).fetchone()
        # A rejected dependency blocks its dependents; nothing else counts.
        if other is None or other["status"] not in ("verified", "complete"):
            return False
    return True


def next_ready_item(conn: sqlite3.Connection, plan_id: str | None = None) -> sqlite3.Row | None:
    """The next pending item whose dependencies are all verified/complete."""
    q, args = "SELECT * FROM work_items WHERE status='pending'", []
    if plan_id:
        q += " AND plan_id=?"
        args.append(plan_id)
    for row in conn.execute(q + " ORDER BY ordinal, id", args):
        if _deps_satisfied(conn, row):
            return row
    return None


def set_item(conn: sqlite3.Connection, item_id: str, *, status: str | None = None,
             evidence: str | None = None, artifact: str | None = None,
             notes: str | None = None, options: str | None = None) -> None:
    if status is not None and status not in STATUSES:
        raise ValueError(f"bad status {status!r}")
    sets, args = ["updated=?"], [time.time()]
    for col, val in (("status", status), ("evidence", evidence),
                     ("artifact", artifact), ("notes", notes), ("options", options)):
        if val is not None:
            sets.append(f"{col}=?")
            args.append(val)
    args.append(item_id)
    with conn:
        conn.execute(f"UPDATE work_items SET {', '.join(sets)} WHERE id=?", args)


# ------------------------------------------------------------------ runs

def log_run(conn: sqlite3.Connection, *, role: str, prompt: str, output: str, ok: bool,
            ms: int, plan_id: str | None = None, item_id: str | None = None,
            engine: str | None = None) -> None:
    with conn:
        conn.execute(
            "INSERT INTO runs (ts, plan_id, item_id, role, engine, prompt, output, ok, ms)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (time.time(), plan_id, item_id, role, engine, (prompt or "")[:8000],
             (output or "")[:8000], 1 if ok else 0, ms))


# ------------------------------------------------------------------ cursors / sessions

def get_cursor(conn: sqlite3.Connection, name: str, default: str = "0") -> str:
    row = conn.execute("SELECT value FROM cursors WHERE name=?", (name,)).fetchone()
    return row["value"] if row else default


def set_cursor(conn: sqlite3.Connection, name: str, value: str) -> None:
    with conn:
        conn.execute("INSERT INTO cursors (name, value, ts) VALUES (?,?,?)"
                     " ON CONFLICT(name) DO UPDATE SET value=excluded.value, ts=excluded.ts",
                     (name, str(value), time.time()))


def touch_session(conn: sqlite3.Connection, key: str, model: str | None = None,
                  title: str | None = None) -> None:
    now = time.time()
    with conn:
        conn.execute(
            "INSERT INTO sessions (key, started, touched, model, title, turns) VALUES (?,?,?,?,?,1)"
            " ON CONFLICT(key) DO UPDATE SET touched=excluded.touched, turns=turns+1,"
            "  model=COALESCE(excluded.model, model), title=COALESCE(excluded.title, title)",
            (key, now, now, model, title))


# ------------------------------------------------------------------ cli

def _main() -> None:
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "init"
    conn = open_db()
    if cmd == "init":
        tables = [r["name"] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        print(f"initialised {DB_PATH}")
        print("tables:", ", ".join(tables))
    elif cmd == "stats":
        for t in ("turns", "sessions", "facts", "archive_chunks", "plans", "work_items", "runs"):
            try:
                n = conn.execute(f"SELECT count(*) c FROM {t}").fetchone()["c"]
            except sqlite3.Error:
                n = "-"
            print(f"{t:>16}: {n}")
        print("\nactive facts:")
        for r in active_facts(conn, limit=20):
            print(f"  [{r['id']}] {r['entity'] or '-'}/{r['topic'] or '-'}: {r['statement'][:100]}")
    elif cmd == "items":
        pid = sys.argv[2] if len(sys.argv) > 2 else None
        for r in list_items(conn, pid):
            print(f"  {r['id']:<10} {r['status']:<12} {r['title'][:70]}")
    elif cmd == "facts":
        for r in active_facts(conn, limit=100):
            print(f"  [{r['id']}] {r['entity'] or '-'}/{r['topic'] or '-'}: {r['statement']}")
    elif cmd == "search":
        for r in search_facts(conn, " ".join(sys.argv[2:])):
            print(f"  [{r['id']}] {r['statement']}")
    else:
        print(__doc__)
        raise SystemExit(2)


if __name__ == "__main__":
    _main()
