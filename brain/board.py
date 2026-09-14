"""The devteam panels — silent by default, loud only when a human decision is needed.

Three views over the same work_items/plans/runs tables the worker writes, so they cannot
drift from reality. Read-only by construction: this service never writes, so it is always
safe to leave open.

    /           SILENT   — is the team working, on what, and does it need you. Nothing else.
    /architect  ARCHITECT— the plans, the canonical design contract, the review findings:
                           the reasoning, not just the status.
    /tasks      TASKS    — every work item, its evidence, its errors, what ran when.
    /api/board  the whole thing as JSON
    /api/needs  only the things awaiting a human decision

The "needs you" set is deliberately narrow. It fires for exactly four things, all of
which are genuinely blocked on a person: a plan nobody has reviewed, a plan the review
found contradictory, a reviewed plan waiting for approval, and an item that failed past
its retry budget. Everything else — progress, retries, successful items — stays silent.
"""
from __future__ import annotations

import html
import json
import os
import sqlite3
import time
from datetime import datetime

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

DB_PATH = os.getenv("JARVIS_DB", "/var/home/admin/jarvis/conversations.db")
REFRESH = int(os.getenv("BOARD_REFRESH", "10"))
SILENT_REFRESH = int(os.getenv("BOARD_SILENT_REFRESH", "30"))
STALE_AFTER = float(os.getenv("BOARD_STALE_AFTER", "900"))
# Where "back to Jarvis" points. Default assumes the proxied path (/team/ under the
# docks), where ".." is the dock and "../live" is the conversation surface.
UI_BASE = os.getenv("JARVIS_UI_BASE", "..")

app = FastAPI(title="Jarvis devteam panels")

STATUS_LABEL = {
    "in-progress": "working", "blocked": "stuck", "pending": "queued",
    "verified": "verified", "complete": "done", "rejected": "rejected",
}
ORDER = ["in-progress", "blocked", "pending", "verified", "complete", "rejected"]


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def collect() -> dict:
    conn = _db()
    try:
        plans = [dict(r) for r in conn.execute(
            "SELECT id, title, brief, body, design, approved, consistent, reviewed_at,"
            " created FROM plans ORDER BY created DESC")]
        items = [dict(r) for r in conn.execute(
            "SELECT id, plan_id, title, detail, status, updated, created, verify,"
            " evidence, depends_on, notes, artifact FROM work_items")]
        runs = [dict(r) for r in conn.execute(
            "SELECT ts, item_id, role, ok, ms, plan_id FROM runs ORDER BY id DESC LIMIT 40")]
    finally:
        conn.close()

    by_plan: dict[str, list] = {}
    for it in items:
        by_plan.setdefault(it["plan_id"] or "", []).append(it)

    now = time.time()
    for plan in plans:
        mine = by_plan.get(plan["id"], [])
        plan["items"] = sorted(mine, key=lambda i: (ORDER.index(i["status"])
                                                    if i["status"] in ORDER else 9, i["id"]))
        plan["total"] = len(mine)
        plan["done"] = sum(1 for i in mine if i["status"] in ("verified", "complete"))
        plan["blocked"] = sum(1 for i in mine if i["status"] == "blocked")
        plan["working"] = sum(1 for i in mine if i["status"] == "in-progress")
        plan["pct"] = round(100 * plan["done"] / plan["total"]) if plan["total"] else 0
        plan["reviewed"] = plan["reviewed_at"] is not None
        if not plan["total"]:
            plan["state"] = "empty"
        elif plan["done"] == plan["total"]:
            plan["state"] = "finished"
        elif plan["blocked"]:
            plan["state"] = "needs attention"
        elif not plan["reviewed"]:
            plan["state"] = "awaiting review"
        elif plan["consistent"] == 0:
            plan["state"] = "contradictory"
        elif not plan["approved"]:
            plan["state"] = "awaiting approval"
        elif plan["working"]:
            plan["state"] = "working"
        else:
            plan["state"] = "idle"

    last_run = runs[0]["ts"] if runs else None
    busy = bool([i for i in items if i["status"] == "in-progress"]) and \
        last_run is not None and (now - last_run) < STALE_AFTER
    data = {
        "now": now, "plans": [p for p in plans if p["total"]], "items": items,
        "runs": runs, "active": [i for i in items if i["status"] == "in-progress"],
        "last_run": last_run, "busy": busy,
    }
    data["needs"] = needs_human(data)
    data["totals"] = {
        "plans": len(data["plans"]), "items": len(items),
        "verified": sum(1 for i in items if i["status"] in ("verified", "complete")),
        "blocked": sum(1 for i in items if i["status"] == "blocked"),
        "pending": sum(1 for i in items if i["status"] == "pending"),
    }
    return data


def needs_human(data: dict) -> list[dict]:
    """The only things worth interrupting a person for."""
    out = []
    for p in data["plans"]:
        if p["state"] == "finished":
            continue
        # Stuck items are reported individually below, with their real error — a
        # plan-level "N failed" rollup on top of that is the same news twice.
        if p["state"] == "contradictory":
            out.append({"severity": "decision", "plan": p["id"],
                        "what": f"“{p['title']}” failed its consistency review — "
                                f"the coder will not start",
                        "action": f"devteam.py review {p['id']}"})
        elif not p["reviewed"]:
            out.append({"severity": "decision", "plan": p["id"],
                        "what": f"“{p['title']}” has not been sanity-checked by the architect",
                        "action": f"devteam.py review {p['id']}"})
        elif not p["approved"]:
            out.append({"severity": "decision", "plan": p["id"],
                        "what": f"“{p['title']}” passed review and is waiting for your approval",
                        "action": f"devteam.py approve {p['id']}"})
    for it in data["items"]:
        if it["status"] == "blocked":
            tail = [l for l in (it["evidence"] or "").strip().splitlines() if l.strip()]
            out.append({"severity": "stuck", "item": it["id"], "plan": it["plan_id"],
                        "what": f"{it['id']} is stuck: {it['title']}",
                        "detail": tail[-1][:300] if tail else "",
                        "action": f"devteam.py respec {it['id']}"})
    return out


def _esc(t) -> str:
    return html.escape(str(t if t is not None else ""))


def _ago(sec: float) -> str:
    sec = max(0, sec)
    if sec < 60:
        return f"{int(sec)}s ago"
    if sec < 3600:
        return f"{int(sec // 60)}m ago"
    if sec < 86400:
        return f"{int(sec // 3600)}h ago"
    return f"{int(sec // 86400)}d ago"


CSS = """
:root { --bg:#05070a; --panel:#0b1016; --line:#16202b; --text:#cfe3f2; --dim:#6b8497;
        --accent:#39d0ff; --ok:#4ade80; --warn:#ffb454; --bad:#ff6b6b; --queued:#41566a; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text);
       font:13.5px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace; }
a { color:var(--accent); text-decoration:none; }
a:hover { text-decoration:underline; }
.wrap { max-width:1240px; margin:0 auto; padding:22px 18px 70px; }
header { display:flex; align-items:center; gap:10px; flex-wrap:wrap;
         border-bottom:1px solid var(--line); padding-bottom:13px; margin-bottom:22px; }
header h1 { font-size:15px; margin:0; letter-spacing:2.5px; font-weight:600;
            color:var(--accent); }
header .spacer { flex:1; }
nav a { padding:5px 12px; border:1px solid var(--line); border-radius:6px;
        margin-left:7px; font-size:12px; }
nav a.on { border-color:var(--accent); color:var(--accent); }
.totals { display:flex; gap:9px; flex-wrap:wrap; margin-bottom:22px; }
.tot { background:var(--panel); border:1px solid var(--line); border-radius:8px;
       padding:8px 13px; min-width:96px; }
.tot b { display:block; font-size:19px; font-weight:600; }
.tot span { color:var(--dim); font-size:10.5px; text-transform:uppercase;
            letter-spacing:.7px; }
.plan { background:var(--panel); border:1px solid var(--line); border-radius:10px;
        padding:15px 17px; margin-bottom:14px; }
.plan h2 { font-size:14px; margin:0; font-weight:600; }
.head { display:flex; justify-content:space-between; align-items:baseline; gap:12px;
        flex-wrap:wrap; }
.badge { font-size:10.5px; padding:2px 9px; border-radius:20px;
         border:1px solid var(--line); color:var(--dim); white-space:nowrap; }
.badge.working { color:var(--warn); border-color:#4a3a12; }
.badge.finished, .badge.verified { color:var(--ok); border-color:#1c3a22; }
.badge.blocked, .badge.needs { color:var(--bad); border-color:#4a1f1d; }
.badge.contradictory, .badge.decision { color:var(--warn); border-color:#4a3a12; }
.bar { height:4px; background:#141c25; border-radius:3px; margin:11px 0 13px;
       overflow:hidden; }
.bar i { display:block; height:100%; background:var(--ok); }
.item { display:flex; gap:10px; padding:5px 0; align-items:baseline;
        border-bottom:1px solid #101821; }
.item:last-child { border-bottom:0; }
.item .id { color:var(--dim); font-size:11.5px; min-width:132px; }
.item .st { font-size:10.5px; min-width:70px; }
.st.verified, .st.complete { color:var(--ok); }
.st.in-progress { color:var(--warn); }
.st.blocked, .st.rejected { color:var(--bad); }
.st.pending { color:var(--queued); }
.item .ti { flex:1; }
.item .ago { color:var(--dim); font-size:11px; white-space:nowrap; }
.note { color:var(--dim); font-size:11.5px; padding-left:142px; margin:1px 0 5px;
        white-space:pre-wrap; word-break:break-word; }
.note.err { color:var(--bad); }
.section { margin-top:26px; }
.section h3 { font-size:11px; color:var(--dim); text-transform:uppercase;
              letter-spacing:1px; margin:0 0 9px; font-weight:600; }
.quiet { color:var(--dim); padding:16px 0; }
pre { background:#080d13; border:1px solid var(--line); border-radius:8px;
      padding:13px 15px; overflow-x:auto; white-space:pre-wrap; word-break:break-word;
      font-size:12.5px; margin:8px 0 0; }
details { margin-top:7px; }
summary { cursor:pointer; color:var(--dim); font-size:12px; padding:3px 0; }
summary:hover { color:var(--text); }
.needs { background:#1a1206; border:1px solid #4a3a12; border-radius:10px;
         padding:14px 17px; margin-bottom:20px; }
.needs h3 { margin:0 0 9px; font-size:12px; color:var(--warn); letter-spacing:1px;
            text-transform:uppercase; }
.needs .row { padding:5px 0; border-top:1px solid #2a2010; }
.needs .row:first-of-type { border-top:0; }
.needs code { color:var(--dim); font-size:12px; }
.ok-banner { background:#0b1710; border:1px solid #1c3a22; border-radius:10px;
             padding:13px 17px; margin-bottom:20px; color:var(--ok); font-size:12.5px; }
.silent { text-align:center; padding:60px 16px 40px; }
.silent .big { font-size:26px; letter-spacing:1px; margin-bottom:10px; }
.silent .big.on { color:var(--warn); }
.silent .big.ok { color:var(--ok); }
.silent .big.need { color:var(--warn); }
.silent .sub { color:var(--dim); font-size:13px; }
.silent .dot { display:inline-block; width:8px; height:8px; border-radius:50%;
               background:var(--queued); margin-right:10px; vertical-align:middle; }
.silent .dot.on { background:var(--warn); animation:pulse 1.8s infinite; }
.silent .dot.need { background:var(--bad); animation:pulse 1.2s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }
a.back { border:1px solid var(--line); border-radius:6px; padding:5px 11px;
         font-size:12px; margin-right:14px; white-space:nowrap; }
a.back:hover { border-color:var(--accent); text-decoration:none; }
.opts { margin:9px 0 4px 0; border-left:2px solid #2a2010; padding-left:13px; }
.opt { display:flex; gap:11px; align-items:flex-start; padding:7px 0;
       border-bottom:1px solid #12161d; }
.opt:last-child { border-bottom:0; }
.opt button { flex:0 0 auto; margin-top:2px; background:#141c25; color:var(--text);
              border:1px solid var(--line); border-radius:6px; padding:5px 12px;
              cursor:pointer; font:12px ui-monospace,Menlo,monospace; }
.opt button:hover { border-color:var(--accent); color:var(--accent); }
.opt button:disabled { opacity:.45; cursor:default; }
.opt .body { flex:1; }
.opt .lbl { font-weight:600; }
.opt .lbl .rec { color:var(--ok); font-size:11px; margin-left:7px; }
.opt .meta { color:var(--dim); font-size:11.5px; margin-top:2px; }
.opt.done .lbl { color:var(--ok); }
#toast { position:fixed; left:50%; bottom:22px; transform:translateX(-50%);
         background:#0b1016; border:1px solid var(--line); border-radius:8px;
         padding:11px 20px; font-size:13px; display:none; z-index:99; max-width:80vw; }
#toast.err { border-color:#4a1f1d; color:var(--bad); }
#toast.ok { border-color:#1c3a22; color:var(--ok); }
"""


def _shell(title: str, active: str, data: dict, body: str, refresh: int) -> str:
    # Relative hrefs on purpose: the same markup has to work standalone on :8094 and
    # proxied under the docks panel at /team/, where "/" would escape to the panel.
    nav = "".join(
        f'<a href="{href}" class="{"on" if key == active else ""}">{label}</a>'
        for key, href, label in (("silent", "./", "Silent"), ("architect", "architect", "Architect"),
                                 ("tasks", "tasks", "Tasks")))
    # The way back. `..` is the docks when proxied; JARVIS_UI_BASE overrides it. Both
    # links are same-origin so they work inside the Android WebView, where target="_blank"
    # does nothing.
    back = ('<a class="back" href="../live">‹ back to Jarvis</a>'
            if UI_BASE == ".." else f'<a class="back" href="{UI_BASE}/live">‹ back to Jarvis</a>')
    return (f'<!doctype html><html><head><meta charset="utf-8"><title>{_esc(title)}</title>'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<meta http-equiv="refresh" content="{refresh}">'
            f'<style>{CSS}</style></head><body><div class="wrap">'
            f'<header>{back}<h1>DEV TEAM</h1><span class="spacer"></span>'
            f'<nav>{nav}</nav></header>{body}'
            f'<div id="toast"></div>'
            f'<script>{JS}</script></div></body></html>')


JS = """
function toast(msg, cls) {
  var t = document.getElementById('toast');
  t.textContent = msg; t.className = cls || ''; t.style.display = 'block';
  clearTimeout(t._h); t._h = setTimeout(function(){ t.style.display = 'none'; }, 6000);
}
function choose(item, index, btn, label) {
  if (!confirm('Apply "' + label + '" to ' + item + '?')) return;
  btn.disabled = true; btn.textContent = '…';
  fetch('api/choose', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({item_id: item, index: index})
  }).then(function(r){ return r.json().then(function(d){ return {ok:r.ok, d:d}; }); })
    .then(function(res){
      if (res.ok) {
        btn.textContent = 'applied'; btn.parentElement.classList.add('done');
        toast(item + ': ' + (res.d.result || 'done'), 'ok');
        setTimeout(function(){ location.reload(); }, 1200);
      } else {
        btn.disabled = false; btn.textContent = 'apply';
        toast('Could not apply: ' + (res.d.detail || res.d.error || 'unknown'), 'err');
      }
    }).catch(function(e){
      btn.disabled = false; btn.textContent = 'apply';
      toast('Request failed: ' + e, 'err');
    });
}
"""


def _options_html(item) -> str:
    """One button per option. This is the whole point of the options feature — an
    option nobody can act on is just a longer way of saying 'stuck'."""
    try:
        opts = json.loads(item["options"] or "[]")
    except (ValueError, TypeError):
        return ""
    if not opts:
        return ""
    rows = []
    for n, o in enumerate(opts, 1):
        if not isinstance(o, dict):
            continue
        rec = '<span class="rec">✓ recommended</span>' if o.get("recommended") else ""
        lbl = _esc(o.get("label") or f"option {n}")
        what = _esc(o.get("what") or "")
        meta = _esc(f"cost: {o.get('cost', '?')}  |  risk: {o.get('risk', '?')}")
        action = o.get("action") or {}
        kind = str(action.get("kind") or "manual").lower()
        if kind == "exec" and action.get("command"):
            meta += _esc(f"  |  runs: {action['command']}")
        elif kind == "manual":
            meta += "  |  needs a human — recorded, not executed"
        rows.append(
            f'<div class="opt"><button onclick="choose(\'{_esc(item["id"])}\', {n}, this, '
            f'\'{lbl}\')">apply</button>'
            f'<div class="body"><div class="lbl">{lbl}{rec}</div>'
            f'<div>{what}</div><div class="meta">{meta}</div></div></div>')
    return f'<div class="opts">{"".join(rows)}</div>' if rows else ""


def _totals(t: dict) -> str:
    return ('<div class="totals">'
            f'<div class="tot"><b>{t["plans"]}</b><span>plans</span></div>'
            f'<div class="tot"><b>{t["verified"]}/{t["items"]}</b><span>verified</span></div>'
            f'<div class="tot"><b>{t["pending"]}</b><span>queued</span></div>'
            f'<div class="tot"><b>{t["blocked"]}</b><span>stuck</span></div>'
            '</div>')


def _needs_block(needs: list) -> str:
    if not needs:
        return ('<div class="ok-banner">Nothing needs you. The team is working or waiting '
                'on its own.</div>')
    rows = "".join(
        f'<div class="row"><b>{_esc(n["what"])}</b>'
        + (f'<div class="note err">{_esc(n["detail"])}</div>' if n.get("detail") else "")
        + f'<div><code>{_esc(n["action"])}</code></div></div>' for n in needs)
    return f'<div class="needs"><h3>Needs a human decision ({len(needs)})</h3>{rows}</div>'


# ------------------------------------------------------------------ silent

@app.get("/", response_class=HTMLResponse)
def silent():
    d = collect()
    now, needs = d["now"], d["needs"]
    if needs:
        cls, big = "need", f"{len(needs)} decision(s) waiting"
        if d["busy"]:
            sub = ("The team is still working on everything else. "
                   + ", ".join(i["id"] for i in d["active"][:3]))
        else:
            sub = "The team is paused until you decide."
    elif d["busy"]:
        cls, big = "on", "Working"
        sub = "on " + ", ".join(f'{i["id"]} — {i["title"]}' for i in d["active"][:3])
    else:
        cls, big = "", "Idle"
        sub = (f'last activity {_ago(now - d["last_run"])}'
               if d["last_run"] else "nothing has run yet")

    body = ('<div class="silent">'
            f'<div class="big {cls}"><span class="dot {cls}"></span>{_esc(big)}</div>'
            f'<div class="sub">{_esc(sub)}</div>'
            '</div>')
    body += _totals(d["totals"])
    body += _needs_block(needs)
    if d["plans"]:
        body += '<div class="section"><h3>plans</h3>'
        for p in d["plans"]:
            body += (f'<div class="plan"><div class="head"><h2>{_esc(p["title"] or p["id"])}</h2>'
                     f'<span class="badge {p["state"].split()[0]}">{_esc(p["state"])} · '
                     f'{p["done"]}/{p["total"]}</span></div>'
                     f'<div class="bar"><i style="width:{p["pct"]}%"></i></div></div>')
        body += '</div>'
    return _shell("Jarvis dev team", "silent", d, body, SILENT_REFRESH)


# ------------------------------------------------------------------ architect

@app.get("/architect", response_class=HTMLResponse)
def architect():
    """The reasoning: what was decided, in what order, and why."""
    d = collect()
    now = d["now"]
    body = _needs_block(d["needs"])
    if not d["plans"]:
        body += '<div class="quiet">No plans yet.</div>'

    for p in d["plans"]:
        body += ('<div class="plan">')
        body += (f'<div class="head"><h2>{_esc(p["title"] or p["id"])}</h2>'
                 f'<span class="badge {p["state"].split()[0]}">{_esc(p["state"])} · '
                 f'{p["done"]}/{p["total"]} verified</span></div>')
        body += f'<div class="bar"><i style="width:{p["pct"]}%"></i></div>'

        if p.get("brief"):
            body += (f'<details><summary>the brief the architect was given</summary>'
                     f'<pre>{_esc(p["brief"])}</pre></details>')
        if p.get("design"):
            body += (f'<details open><summary>canonical design contract — every item is '
                     f'built against this</summary><pre>{_esc(p["design"])}</pre></details>')
        else:
            body += ('<div class="note">Not yet reviewed — no design contract recorded. '
                     'The coder will not start until this is done.</div>')

        body += '<div class="section"><h3>work items — the specification each one was built from</h3>'
        for it in p["items"]:
            st = it["status"]
            body += (f'<div class="item"><span class="id">{_esc(it["id"])}</span>'
                     f'<span class="st {_esc(st)}">{_esc(STATUS_LABEL.get(st, st))}</span>'
                     f'<span class="ti">{_esc(it["title"])}</span>'
                     f'<span class="ago">{_ago(now - (it["updated"] or now))}</span></div>')
            if it.get("detail"):
                body += (f'<details><summary>specification</summary>'
                         f'<pre>{_esc(it["detail"])}</pre></details>')
            if it.get("verify"):
                body += (f'<div class="note">verified by: <code>{_esc(it["verify"])}</code></div>')
            if it.get("depends_on"):
                body += f'<div class="note">after: {_esc(it["depends_on"])}</div>'
            if st == "blocked" and it.get("evidence"):
                tail = [l for l in it["evidence"].strip().splitlines() if l.strip()]
                body += f'<div class="note err">{_esc(tail[-1][:300] if tail else "")}</div>'
            body += _options_html(it)
            if it.get("notes") and not it.get("options"):
                body += f'<div class="note">{_esc(it["notes"][:400])}</div>'
        body += '</div></div>'

    return _shell("Jarvis — architect", "architect", d, body, REFRESH)


# ------------------------------------------------------------------ tasks

@app.get("/tasks", response_class=HTMLResponse)
def tasks():
    d = collect()
    now = d["now"]
    body = _totals(d["totals"])

    by_status: dict[str, list] = {}
    for it in d["items"]:
        by_status.setdefault(it["status"], []).append(it)

    for st in ORDER:
        group = by_status.get(st) or []
        if not group:
            continue
        body += f'<div class="section"><h3>{_esc(STATUS_LABEL.get(st, st))} ({len(group)})</h3>'
        for it in group:
            body += (f'<div class="item"><span class="id">{_esc(it["id"])}</span>'
                     f'<span class="st {_esc(it["status"])}">{_esc(it["plan_id"] or "")}</span>'
                     f'<span class="ti">{_esc(it["title"])}</span>'
                     f'<span class="ago">{_ago(now - (it["updated"] or now))}</span></div>')
            body += _options_html(it)
            if it["status"] == "blocked" and it.get("evidence"):
                body += (f'<details><summary>why it is stuck</summary>'
                         f'<pre>{_esc(it["evidence"][-1200:])}</pre></details>')
        body += '</div>'

    if d["runs"]:
        body += '<div class="section"><h3>what ran</h3>'
        for r in d["runs"][:30]:
            mark = "ok" if r["ok"] else "FAILED"
            secs = f'{r["ms"] / 1000:.0f}s' if r["ms"] else "—"
            body += (f'<div class="item"><span class="id">{_ago(now - r["ts"])}</span>'
                     f'<span class="st {"" if r["ok"] else "blocked"}">{mark}</span>'
                     f'<span class="ti">{_esc(r["item_id"] or r["plan_id"] or "plan")} '
                     f'&nbsp;{_esc(r["role"])}</span>'
                     f'<span class="ago">{secs}</span></div>')
        body += '</div>'

    return _shell("Jarvis — tasks", "tasks", d, body, REFRESH)


# ------------------------------------------------------------------ api

@app.get("/api/board")
def api_board():
    return JSONResponse(collect())


@app.get("/api/needs")
def api_needs():
    """Just the decisions. This is what anything wanting to notify the owner should poll."""
    d = collect()
    return JSONResponse({"needs": d["needs"], "count": len(d["needs"])})


@app.post("/api/choose")
def api_choose(payload: dict):
    """Apply one of the architect's options — the button behind the panels.

    This is the ONE write path in this service; every view endpoint remains strictly
    read-only. Declared `def`, not `async def`, so FastAPI runs it in a threadpool:
    applying an option can shell out, and that must not stall the board's event loop.
    """
    item_id = (payload or {}).get("item_id")
    index = (payload or {}).get("index")
    if not item_id or not isinstance(index, int) or isinstance(index, bool):
        return JSONResponse({"error": "item_id and an integer index are required"},
                            status_code=400)
    lines: list[str] = []
    try:
        import devteam
        result = devteam.apply_option(item_id, index,
                                      log=lambda m="": lines.append(str(m)))
    except SystemExit as exc:            # a refusal, with a reason for the human
        return JSONResponse({"error": "refused", "detail": str(exc)},
                            status_code=400)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": type(exc).__name__, "detail": str(exc)},
                            status_code=500)
    return {"ok": True, "item_id": item_id, "index": index, "result": result,
            "log": lines}


@app.get("/health")
def health():
    try:
        _db().execute("SELECT 1")
    except sqlite3.Error as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=503)
    return {"ok": True, "db": DB_PATH}
