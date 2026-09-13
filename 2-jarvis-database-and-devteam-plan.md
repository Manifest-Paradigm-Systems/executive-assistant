# Jarvis Database + 3-Brain Devteam — architecture review and build plan

**Date:** 2026-09-13
**Source spec:** `17-Expanding_Jarvis_capabilities_Google_Gemini.md` (167 messages, the real brief)
**Status:** plan → build in progress

---

## 0. Review finding: the file you pointed at is empty

`1-creating_Jarvis_database.md` is a 20-line Gemini export containing one question
("do we have any mail for Michael P Meyer in the emails or Google Drive?") and the reply
"Initiating Understanding". No architecture in it. The actual architecture dialog is
**document 17** — I reviewed that in full and this plan is built from it.

## 1. What the spec actually asks for

Reading past the conversational surface, document 17 specifies five systems:

1. **A two-tier (really four-tier) memory** so Jarvis keeps decisions and flushes filler
   ("are you there?"). Session store → async fact extractor → epistemic fact pool →
   chunked debate archive, with source attribution back to the original document/URL.
2. **A unified personality facade** — the owner talks to one Jarvis; behind it the
   *architect* and *coder* are queried as sub-routines and their findings are synthesised
   into one voice. No model switching in the UI.
3. **Plan-and-build handoff** — "Jarvis, draw up a plan and begin the work" must chain
   actor → architect (writes the plan) → coder (implements, tests, self-corrects) →
   report back by voice/notification.
4. **A capability expansion list** — Google Workspace (4 scoped Gmail accounts, Calendar,
   Drive folder ingestion), telephony/voice agents (Pipecat/LiveKit + SIP), SMS/email
   outbound, PDF automation (fill, flatten, CAC-safe signing with pyHanko, DocuSeal
   e-sign, receipt generation), self-hosted accounting, home/server APIs, research
   (JSTOR/OpenAlex/Unpaywall), booking (restaurant/ride via Playwright + HITL), and
   multi-user profiles with RBAC.
5. **A product thesis** — package as a private appliance. (Out of scope for this build;
   it is a consequence of the infrastructure, not a prerequisite.)

## 2. Current state audit (verified live, 2026-09-13)

Already standing — more than expected:

| Layer | Where | State |
|---|---|---|
| Director brain (R1) | cerebro lane `:8081` | up 2 days |
| Coder brain (Qwen-Coder-32B) | cerebro lane `:8082` | up 2 days |
| Actor brain (Kunou) | cerebro lane `:8083` | up 2 days |
| Vision / hearing | cerebro `:8084` / `:8085` | up |
| **Brain facade** | cerebro `jarvis-brain.service` `:8092` (`~/jarvis/brain/brain.py`) | live, 45 KB — OpenAI-shaped `/v1/chat/completions`, `@@DELEGATE`/`@@PLAN`/`@@HANDOFF`/`@@VOICE`/`@@PLAY` directive protocol, background jobs, `record()`/`recall()` |
| Architect docks panel | cerebro `jarvis-docks.service` `:8093` (`panel.py`) | live |
| Front ends | Open WebUI `:8080`, docks `/live` hands-free voice | live |
| Voice out | Workhorse voices-adapter (Higgs) `:7863` | live |
| Raw session store | cerebro `~/jarvis/conversations.db` | **turns + FTS5**, 186 rows |
| Programmatic devteam | Workhorse `~/AI-devteam` (Node, 4 roles, worktrees, Playwright+vision) | exists, proven |
| Headless Claude sessions | `~/bin/claude-coder.sh`, `claude-director.sh`, `handoff-to-coder.sh` | exist |

**So the gaps are specific, not general.** The three brains exist and talk. What is
missing is (a) *memory above raw transcripts* and (b) *durable execution state* — the
brains can chat about work but the work itself has nowhere to live.

## 3. Gap analysis

### 3.1 Memory
- Raw turns are kept, but **nothing extracts meaning from them**. `recall()` is an FTS
  scan over raw turns — retrieving "are you there?" and a load-bearing decision with
  equal weight.
- **No fact tier**: no canonical decisions, no preferences, no configs.
- **No consolidation**: a decision revised later coexists with its predecessor instead of
  superseding it.
- **No debate archive**: the spec is explicit that rejected options must stay retrievable
  ("why didn't we use a 70B architect?"), which raw-turn FTS cannot answer.
- **No source attribution**: nothing links a fact back to a document or URL.
- **No injection**: the brain does not put curated facts in front of the model.

### 3.2 Devteam
- Delegation is **in-memory only** (`JOBS` dict in `brain.py`). A restart loses every job.
- **The coder lane cannot edit files.** Qwen on `:8082` is a chat endpoint; it returns
  text. There is no apply → test → verify path, so "the coder implemented it" is not a
  statement the current system can truthfully make.
- **No verification role.** Nothing checks that a claimed implementation works.
- **No work items.** No dependency ordering, no status machine, no evidence trail.
- **No approval durability.** The approval gate lives in a session dict, not a record.

## 4. Design

### 4.1 The Jarvis database

**One database, on cerebro** (`~/jarvis/conversations.db`) — the architecture rule is
data/text DBs on cerebro, and `brain.py` already owns this file. New tables are strictly
additive; the live service is untouched.

```
turns            (exists)  raw session log — UI history, short-term context
sessions         (new)     one row per conversation: rollup, title, archived flag
facts            (new)     the epistemic pool: decisions, preferences, configs
facts_fts        (new)     FTS5 over facts
archive_chunks   (new)     chunked debate archive (keeps rejected options + reasoning)
archive_fts      (new)     FTS5 over the archive
plans            (new)     a plan: brief, body, approval state
work_items       (new)     the unit of devteam work: owner, status, deps, evidence
runs             (new)     every model call the devteam makes (audit + cost)
memory_cursor    (new)     high-water mark for the extraction worker
```

Key columns:

- `facts(entity, topic, statement, kind, status, confidence, supersedes, source_session,
  source_turn, source_url)`. `status ∈ active|superseded|rejected`. **Supersede, never
  silently overwrite** — the old fact stays, pointed at by `supersedes` from the new one,
  so "what did we believe in August?" is still answerable. This is the one place I depart
  from the spec's "overwrite the outdated fact": an overwrite throws away the audit trail
  that makes the archive worth having.
- `work_items(id, plan_id, title, detail, owner, status, depends_on, artifact, evidence)`.
  `status ∈ pending|in-progress|implemented|verified|complete|blocked|rejected` — the same
  status vocabulary the existing `AI-devteam` harness uses, so the two can interoperate.
- `runs(...)` records every lane call, so the devteam can be audited and debugged.

Embeddings: **deferred deliberately.** FTS5 + entity/topic filters answers the spec's
actual queries, runs on CPU, adds no dependency and no VRAM, and is already proven in this
file. A vector column can be added when a real query fails on FTS — not before.

### 4.2 Memory worker

`~/jarvis/brain/memory.py` on cerebro, driven by a systemd user timer (2 min).

1. Read `turns` past `memory_cursor`.
2. Group into windows (a session's recent span), skip windows that are pure filler.
3. Run the **discriminator prompt** (from the spec) against the cheap lane — output strict
   JSON `{"extracted_facts": [...], "chunks": [...]}`.
4. Consolidate: for each candidate fact, FTS-match existing active facts on
   (entity, topic). If it *revises* one → insert new + mark old `superseded` + set
   `supersedes`. If it *duplicates* one → bump `confidence`, touch `updated`. Else insert.
5. Archive: write the window's debate chunk (including rejected options and reasoning)
   into `archive_chunks` with source attribution.
6. Advance the cursor. Never blocks the live brain — separate process, WAL-friendly reads.

Failure policy: a bad model reply advances nothing and retries next tick; the worker never
writes a partial window.

### 4.3 The 3-brain devteam

The design point the current system is missing: **brains think, the worker executes.**

```
                    ┌──────────────────────────────────────────┐
   owner ──voice──► │ ACTOR (Kunou :8083) — the one voice       │
                    │  @@DELEGATE director: <brief>            │
                    └──────────────────┬───────────────────────┘
                                       ▼
                    ┌──────────────────────────────────────────┐
                    │ DIRECTOR (R1 :8081) — plans, not executes │
                    │  writes PLAN + work items (with deps)     │
                    │  approval gate: owner says yes            │
                    └──────────────────┬───────────────────────┘
                                       ▼   (approved)
                    ┌──────────────────────────────────────────┐
                    │ DEVTEAM WORKER (cerebro) — the hands      │
                    │  pull next ready item (deps satisfied)    │
                    │  CODER (Qwen :8082) → strict-JSON diffs   │
                    │  apply in git worktree (never live)       │
                    │  run the item's verify command            │
                    │    fail → DIRECTOR course-corrects (≤3)   │
                    │    pass → item verified + evidence        │
                    └──────────────────┬───────────────────────┘
                                       ▼
                    report → actor speaks it / Matrix notify
```

Decisions:

- **Coder output is strict JSON**, reusing the contract already proven in `~/AI-devteam`:
  `{status, diffs[], files[], tests_written, notes}`. `diffs` must pass `git apply --check`;
  every `files[].path` is validated to stay inside the worktree. This is borrowed
  deliberately — it is the part of that harness that already works.
- **Worktrees only.** The live checkout is never written by the loop. Applying to live is a
  separate, explicit step.
- **Verification is a command, not an opinion.** Each work item carries a `verify` command;
  the item is only `verified` if that command exits 0. A model saying "done" is not
  evidence — this is the defect that makes most local-model devteams useless.
- **Bounded correction.** 3 failed attempts → item goes `blocked` with the failure log, and
  the director is asked once for a re-plan. Matches the existing `ag2_director.py`
  convention (`AG2_CODER_FAILURE_LIMIT=3`).
- **Durability.** All state in the DB. Killing the worker mid-run leaves items
  `in-progress`; the next start reclaims them. No work is lost to a restart.
- **Approval stays human.** The plan gate is unchanged: plans are `approved=0` until the
  owner approves. `--autonomous` exists for unattended runs but is off by default.

### 4.4 Memory injection (the payoff)

`brain.py` gets one additive change: `context_for(query)` returns, in order —
active facts matching the query (with entity/topic), then archive hits, then raw turns.
That block is prepended to the conversational system prompt as
`[Known context — verified decisions and facts]`. This is what makes Jarvis *remember*
rather than *scroll back*.

## 5. Build phases

| # | Phase | Deliverable | Gate |
|---|---|---|---|
| 1 | Jarvis DB | schema + migration on cerebro | tables live, existing service unaffected |
| 2 | Memory worker | `memory.py` + timer | filler rejected, decisions captured, supersede works |
| 3 | Injection | `brain.py` context upgrade | restart clean, facts appear in replies |
| 4 | Devteam worker | `devteam.py` + CLI | one item end-to-end: plan → diff → apply → verify |
| 5 | Tests | unit + live end-to-end | green |
| 6 | Dogfood | doc 17 → plan + work items in the DB | items are concrete and ordered |
| 7 | **Decompose** | roadmap → offline-team-sized tasks in the DB | every item is Qwen-sized |

### Scope decision (owner, 2026-09-13)

Two things are **deferred, by owner instruction** — captured as planned work items but not
built in this pass:

- **Google OAuth / Workspace keys** — the owner will supply credentials later. The
  architecture absorbs them without change; nothing here blocks on them.
- **Telephony** — the owner's direction is a *free open-source SIP trunk*, with **Telnyx**
  as the candidate for inbound and outbound calls. That is a Wave 3 concern.

The priority instead is explicit: **"get our 3-brain coding team and database working
exceptionally well, then break all this down into manageable tasks for the offline coding
team to handle."** So phase 7 is a *decomposition* phase, not a build-everything phase.
The deliverable is a roadmap the offline local team can actually execute.

**Sizing rule for the offline team** (from hard-won fleet experience, see
`local-model-chain-findings` in memory): Qwen-Coder cannot produce byte-exact patches
against large existing files. Therefore every work item handed to the local team must
either **create a new module/small file**, or **touch a small file**. Work items carry an
explicit `workspace` and `verify` command so "done" is mechanically checkable rather than
asserted.

## 6. Expansion roadmap seed (what phase 6 will plan)

From document 17, ordered by leverage — not by what is easiest:

**Wave 1 — the memory and documents spine** (everything else depends on it)
- Google Workspace OAuth for the four scoped accounts (`coding`, `family`, `rental`,
  `dave_care`), tokens local, one GCP project, per-account scoping.
- Drive folder ingestion → archive chunks + facts, with `webViewLink` attribution.
- PDF pipeline: AcroForm fill (`pypdf`), flat-PDF overlay (`PyMuPDF`), field detection
  (`pdfplumber`, vision lane for scans), receipt generation (`ReportLab`).
- **CAC-safe prep** (`pyHanko` incremental update + empty signature widget; native Linux
  `pcscd`/`OpenSC` signing — no Wine, no Acrobat, no rasterisation).

**Wave 2 — outbound channels**
- Gmail send + SMS/email (Resend/Telnyx) with verbal read-back confirmation and HITL.
- Notifications via Matrix/ntfy.
- PDF dispatch: fill → screenshot → owner approves → send.

**Wave 3 — voice agent / telephony**
- Pipecat on cerebro: SIP trunk (Telnyx/Plivo), local Whisper STT → brain → Higgs TTS.
- Outbound assistant framing ("calling on behalf of…"), low latency, human fallback for
  authorisation-restricted calls (medical/financial — explicitly out of autonomous scope).

**Wave 4 — world interfaces**
- Google Calendar (free/busy, create/move), self-hosted accounting (Firefly III / Actual),
  Docker/Tailscale/Uptime-Kuma server ops, Home Assistant, research (OpenAlex/Unpaywall),
  booking via Playwright with screenshot-confirmed HITL.

**Wave 4b — visual lookup (owner request, 2026-09-13)**

> "Jarvis can use a webcam or camera in a phone in combination with our image reader to be
> capable of seeing things and finding more information about them. Very useful for
> hardware parts or computer technology."

The owner holds a part, a chip, a connector or a board up to a camera and asks what it is.
Jarvis identifies it and returns real specifications — not a guess.

The fleet already has the pieces, which is why this is cheap to build:

| Piece | Where | Role |
|---|---|---|
| eyes lane (vision reader) | cerebro `:8084` | describe/read the image |
| `jarvis-wiki` model | brain.py → wiki `:8090` | authoritative lookup by name |
| `searxng` | cerebro `:8888` | web fallback for part numbers |
| Jarvis Android app | `~/jarvis-android` | phone camera as the capture device |
| webcam | Workhorse | desk-side capture, same pipeline |

Pipeline: **capture → read → name → resolve → answer with sources.**
The vision lane's job is only to produce a *candidate identity* (markings, part number,
connector shape, silkscreen text); the resolution step then uses wiki/web to confirm and
fetch the datasheet-level detail. Keeping those separate is deliberate — vision models
misread part numbers confidently, so the identifier must be verified before it is spoken.

Guardrails: never state a part number the reader is unsure of — say what is legible and
what is inferred, separately. Surface the source link so the owner can check it.


**Wave 5 — multi-user**
- Profiles + RBAC, per-user memory namespaces, scoped tokens, per-user voice.

## 7. Safety invariants (carried from the existing fleet rules)

- **vox-conjurata containers are never touched** by any of this without explicit owner
  permission.
- Work happens in **worktrees**; live application is explicit.
- Data DBs on cerebro; **audio stays on the Workhorse**.
- Heavy jobs honour `/tmp/cinematome-suspend`; the GPU/heat watchdogs stay in force.
- No credentials in the DB, in prompts, or in logs. Tokens live in `0600` env files.
- Outbound email/SMS/telephony always requires human confirmation in this build.
