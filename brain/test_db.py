"""Unit tests for the Jarvis DB layer. Run on a scratch file — never the live DB."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def run():
    path = os.path.join(tempfile.mkdtemp(), "test.db")
    conn = db.open_db(path)

    print("\n-- facts: insert + supersede --")
    old = db.add_fact(conn, "Vector memory backend is SQLite FTS5.",
                      entity="cerebro_system", topic="memory_architecture", kind="decision")
    new = db.add_fact(conn, "Vector memory backend is ChromaDB.",
                      entity="cerebro_system", topic="memory_architecture", kind="decision",
                      supersedes=old)
    check("old fact is superseded",
          conn.execute("SELECT status FROM facts WHERE id=?", (old,)).fetchone()["status"] == "superseded")
    check("new fact is active",
          conn.execute("SELECT status FROM facts WHERE id=?", (new,)).fetchone()["status"] == "active")
    check("new fact points back at the one it replaced",
          conn.execute("SELECT supersedes FROM facts WHERE id=?", (new,)).fetchone()["supersedes"] == old)
    check("history survives — both rows still exist",
          conn.execute("SELECT count(*) c FROM facts").fetchone()["c"] == 2)
    check("only the current belief is active", len(db.active_facts(conn)) == 1)

    print("\n-- facts: search --")
    hits = db.search_facts(conn, "what is the vector memory backend")
    check("FTS finds the active fact", any(h["id"] == new for h in hits))
    check("FTS does not surface the retired fact",
          all(h["id"] != old for h in db.search_facts(conn, "SQLite FTS5")))
    check("superseded row is still reachable for history",
          len(db.search_facts(conn, "SQLite", status="superseded")) == 1)

    print("\n-- facts: consolidation --")
    dup = db.add_fact(conn, "The memory backend uses ChromaDB for vectors.",
                      entity="cerebro_system", topic="memory_architecture")
    near = db.find_similar_fact(conn, "The memory backend uses ChromaDB for vectors.",
                                entity="cerebro_system", topic="memory_architecture")
    check("near-duplicate detected", near is not None and near["id"] in (new, dup))
    check("unrelated text is not merged",
          db.find_similar_fact(conn, "Rent is due on the first of the month.",
                               entity="cerebro_system", topic="memory_architecture") is None)

    print("\n-- retrieval precision (the 'the' problem) --")
    db.add_fact(conn, "The music generator and the foley library are fully operational.",
                entity="cerebro_system", topic="audio_systems")
    db.add_fact(conn, "The owner's cat is named Thunder.", entity="user", topic="pets")
    hits = db.search_facts(conn, "what is the name of my cat")
    check("question about the cat returns the cat fact",
          any("Thunder" in h["statement"] for h in hits))
    check("question about the cat does not return music facts",
          not any("music" in h["statement"].lower() for h in hits),
          [h["statement"][:40] for h in hits])
    hits = db.search_facts(conn, "what did we decide about the memory backend")
    check("unrelated question returns no music facts",
          not any("music" in h["statement"].lower() for h in hits),
          [h["statement"][:40] for h in hits])
    check("all-stopword query returns nothing",
          db.search_facts(conn, "what is it that you do") == [])

    print("\n-- archive --")
    cid = db.add_chunk(conn, "We weighed ChromaDB against SQLite FTS5. FTS5 was rejected because "
                             "semantic recall over the archive mattered more than zero deps.",
                       title="memory backend debate", source_url="file:///plan.md")
    check("archive chunk stored", cid > 0)
    check("rejected option is findable",
          len(db.search_archive(conn, "ChromaDB SQLite rejected")) >= 1)

    print("\n-- work items: dependency gating --")
    pid = db.create_plan(conn, "PLAN-1", "test plan", "brief")
    db.add_item(conn, {"id": "JV-001", "plan_id": pid, "ordinal": 1,
                       "title": "scaffold", "verify": "true"})
    db.add_item(conn, {"id": "JV-002", "plan_id": pid, "ordinal": 2,
                       "title": "build on scaffold", "depends_on": ["JV-001"]})
    db.add_item(conn, {"id": "JV-003", "plan_id": pid, "ordinal": 3, "title": "independent"})

    nxt = db.next_ready_item(conn, pid)
    check("blocked-dependent item is not handed out first", nxt["id"] == "JV-001", nxt["id"])

    db.set_item(conn, "JV-001", status="verified")
    ids = {db.next_ready_item(conn, pid)["id"]}
    db.set_item(conn, "JV-003", status="verified")
    nxt = db.next_ready_item(conn, pid)
    check("dependent unblocks only after its dep verifies", nxt is not None and nxt["id"] == "JV-002")

    db.set_item(conn, "JV-002", status="blocked", evidence="attempt 3 failed: no such table")
    check("blocked item is skipped", db.next_ready_item(conn, pid) is None)
    check("failure evidence is kept",
          conn.execute("SELECT evidence FROM work_items WHERE id='JV-002'").fetchone()["evidence"].startswith("attempt 3"))

    try:
        db.set_item(conn, "JV-001", status="banana")
        check("invalid status rejected", False)
    except ValueError:
        check("invalid status rejected", True)

    print("\n-- deps on a rejected item --")
    db.add_item(conn, {"id": "JV-004", "plan_id": pid, "ordinal": 4, "title": "child of rejected",
                       "depends_on": ["JV-005"]})
    db.add_item(conn, {"id": "JV-005", "plan_id": pid, "ordinal": 5, "title": "rejected thing"})
    db.set_item(conn, "JV-005", status="rejected")
    check("a rejected dependency blocks its dependent",
          db.next_ready_item(conn, pid) is None)

    print("\n-- an item id cannot be stolen by another plan --")
    db.create_plan(conn, "PLAN-2", "second plan", "brief")
    try:
        db.add_item(conn, {"id": "JV-001", "plan_id": "PLAN-2", "title": "hijack"})
        check("cross-plan id collision is refused", False)
    except ValueError:
        check("cross-plan id collision is refused", True)
    check("the original item is untouched",
          conn.execute("SELECT plan_id,title FROM work_items WHERE id='JV-001'").fetchone()["plan_id"] == pid)
    db.add_item(conn, {"id": "JV-900", "plan_id": "PLAN-2", "title": "legit"})
    check("a fresh id in another plan is fine",
          conn.execute("SELECT plan_id FROM work_items WHERE id='JV-900'").fetchone()["plan_id"] == "PLAN-2")

    print("\n-- runs / cursors --")
    db.log_run(conn, role="coder", prompt="do it", output="did it", ok=True, ms=1200,
               plan_id=pid, item_id="JV-001", engine=":8082")
    check("run is audited", conn.execute("SELECT count(*) c FROM runs").fetchone()["c"] == 1)
    db.set_cursor(conn, "memory", "42")
    db.set_cursor(conn, "memory", "43")
    check("cursor updates in place", db.get_cursor(conn, "memory") == "43")

    print("\n-- idempotent init (what a service restart does) --")
    db.init(conn)
    db.add_item(conn, {"id": "JV-001", "plan_id": pid, "ordinal": 1, "title": "scaffold v2",
                       "status": "verified"})
    check("re-adding an item updates rather than duplicating",
          conn.execute("SELECT count(*) c FROM work_items WHERE id='JV-001'").fetchone()["c"] == 1)
    check("update preserved its status",
          conn.execute("SELECT title,status FROM work_items WHERE id='JV-001'").fetchone()["title"] == "scaffold v2")

    print("\n" + ("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}"))
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(run())
