# Executive Assistant ("Jarvis")

The private, self-hosted executive assistant: a voice-first interface backed by three
local model "brains" (director, coder, actor), a durable memory database, and — at the
time of writing — a working agentic development team that plans, implements, and *verifies*
software on the owner's own hardware. Nothing leaves the house.

## What is in this repository

| Path | What it is |
|---|---|
| `2-jarvis-database-and-devteam-plan.md` | **Start here.** The architecture review, gap analysis, design, and build plan for the memory database and the 3-brain devteam. |
| `1-creating_Jarvis_database.md` | Raw Gemini export (untracked — see below). |
| `17-Expanding_Jarvis_capabilities_Google_Gemini.md` | The original expansion conversation that specifies the capabilities roadmap (untracked — see below). |

The two raw transcripts are **deliberately untracked**: they contain personal and family
information (names, rental-property details, a family member's healthcare) that should not
enter a git history. They remain on disk as the design record. See `.gitignore`.

## Where it actually runs

The assistant itself lives on **cerebro** (`/var/home/admin/jarvis/`), not in this
repository. This repo holds the design and — going forward — the source that is deployed
there.

| Component | Host | Port | Role |
|---|---|---|---|
| `jarvis-brain` | cerebro | 8092 | OpenAI-shaped facade over the three brains; directive protocol, background jobs, memory injection |
| director lane | cerebro | 8081 | R1 — plans, does not execute |
| coder lane | cerebro | 8082 | Qwen-Coder — implements |
| actor lane | cerebro | 8083 | the voice the owner talks to |
| vision / hearing | cerebro | 8084 / 8085 | eyes and ears |
| architect docks | cerebro | 8093 | panel + hands-free voice loop |
| Open WebUI | cerebro | 8080 | chat front end |
| wiki / search | cerebro | 8090 / 8888 | local knowledge tier |
| voices (Higgs) | Workhorse | 7863 | speech synthesis |

The memory database is a single SQLite file on cerebro
(`/var/home/admin/jarvis/conversations.db`) holding four tiers: raw turns, extracted
facts, a debate archive, and devteam work items. See the plan document for the schema and
the reasoning behind it.

## Deployment state

At the time of writing, the code on cerebro is **not yet deployed from this repository** —
it was written directly on the host and is being brought under version control here. Until
that is reconciled, this repository is a record rather than a source of truth. That
reconciliation is tracked work, not an oversight.

## Working rules

- Data/text databases live on **cerebro**; audio and sound libraries stay on the
  **Workhorse**.
- Model work happens in **git worktrees**; the live checkout is only touched deliberately.
- Work items reach `verified` only when a verify command exits 0 — never because a model
  said it was done.
- Outbound email, SMS and telephony always require human confirmation.
