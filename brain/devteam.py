"""The 3-brain devteam — director plans, coder implements, and *the machine verifies*.

The gap this closes: on this fleet the three brains already talk, but a language
model cannot edit a file. Asking Qwen "implement X" and reading its reply is not
software development, it is a suggestion. This worker gives the brains hands and,
more importantly, gives their claims a way to be **wrong** — which is what makes
the output trustworthy.

    director (R1 :8081)   --plan-->  work items in the DB (with dependencies)
    coder   (Qwen :8082)  --code-->  complete files / git diffs
    the machine           --verify-> a shell command exits 0, or it did not work
    director (R1 :8081)   --repair-> consulted only when an item is stuck

Four rules, each one the fix for a specific way local-model devteams fail:

1. **Status is earned, never asserted.** An item reaches `verified` because its
   verify command exited 0. A model saying "done" is worth nothing.
2. **The coder proposes, the machine disposes.** Paths are resolved and checked to
   be inside the workspace; existing files are backed up before being overwritten;
   nothing outside the workspace is reachable from a model reply.
3. **Small items, new files.** Qwen-Coder cannot make byte-exact edits to a large
   existing file (measured repeatedly on this fleet). The director is instructed to
   plan new-file work, and the runner refuses to guess when that is violated.
4. **Bounded failure.** Three failed attempts and the item is `blocked` with the
   real error text — not retried forever, and not quietly marked done.

Run:
    python3 devteam.py plan "<brief>" --workspace /path/to/dir
    python3 devteam.py approve <plan_id>
    python3 devteam.py run <plan_id> [--max-items N]
    python3 devteam.py status <plan_id>
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import fcntl
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
import time

import db as jarvis_db
import editor
import llm
import sandbox

MAX_ATTEMPTS = int(os.getenv("DEVTEAM_MAX_ATTEMPTS", "3"))

# Output budget, not context. R1 serves 128k of context across 2 parallel slots and
# strips its own <think> before this code sees it — the plan is a structured artifact,
# and a cap that truncates one mid-JSON throws the entire generation away, so set it
# comfortably ABOVE what a full plan needs rather than just above the average.
DIRECTOR_TOKENS = int(os.getenv("DEVTEAM_DIRECTOR_TOKENS", "12000"))
REPAIR_TOKENS = int(os.getenv("DEVTEAM_REPAIR_TOKENS", "8000"))
VERIFY_TIMEOUT = int(os.getenv("DEVTEAM_VERIFY_TIMEOUT", "300"))
FILE_LIST_LIMIT = 120
# The coder slot is 32k (see /props), and a prompt that has to quote an existing file
# spends most of its budget on that file. 6k was tight enough that the coder could be
# shown the first half of a file it was expected to edit whole.
INLINE_FILE_LIMIT = 16000         # bytes of an existing file worth showing the coder


# ------------------------------------------------------------------ prompts

DIRECTOR_PROMPT = """You are the DIRECTOR planning work for a small engineering team. You do not write code.

Break this into work items. HARD CONSTRAINTS:

1. The coder is a 32-billion-parameter local model working in ONE PASS per item.
   Items must be SMALL. An item that needs more than ~80 lines of code is too big —
   split it.
2. Every item must CREATE A NEW FILE or touch a small file. Never ask for surgical
   edits to a large existing file: this model cannot do that reliably, and an item
   nobody can complete is worse than no item.
3. Every item needs a `verify`: ONE shell command, run from the workspace, that exits 0
   only when this item is genuinely finished and NON-ZERO when it is not. This command
   is the only thing that decides whether the item is done, so it must be able to fail.

   GOOD — these fail when the work is missing or wrong:
     python3 -c "import wordcount; assert wordcount.count_words('a b a') == {{'a': 2, 'b': 1}}"
     python3 -c "from visual_lookup.read import read_image, build_prompt; assert callable(read_image) and 'part' in build_prompt('x').lower()"
     python3 -m pytest visual_lookup/tests/test_resolve.py -q

   BAD — never emit these, they exit 0 no matter what:
     python3 -c "print(read_image(b''))"        # prints; proves nothing
     ls visual_lookup/__init__.py               # existence only, not behaviour
     python3 -c "from x import y"               # import alone is not a behaviour check
     python3 -c "from m import f; assert callable(f)"   # a stub passes this

   Assert on BEHAVIOUR: a return value, a raised error, the contents of a file.
     python3 -c "from m import f; assert f({{'a': 1}}) == {{'a': 2}}"
     python3 -c "from m import f; f(None)"   # should raise — non-zero exit proves it

   QUOTING IS PART OF THE CONTRACT: put DOUBLE quotes on the outside of the -c
   argument so single quotes inside are safe. Never nest quotes of the same kind —
   `python3 -c 'f('x')'` does not parse and the item can never pass.
4. Use `depends_on` (a list of item ids) where order matters.
5. Produce 5 to 12 items, ordered so each is buildable on top of the last.
6. DEPENDENCIES ARE FIXED, AND THESE ARE THEM. The sandbox that runs your verify
   commands has NO NETWORK: nothing can be installed while a check runs. What exists is
   the Python 3.12 standard library plus exactly these packages:
       pypdf          AcroForm fields, page objects, reading and writing PDFs
       pdfplumber     text and geometry extraction, locating blank regions
       PyMuPDF        imported as `fitz` — rendering, overlaying onto an existing page
       reportlab      generating a PDF from scratch
       pytest         only for verifies written as `python3 -m pytest <path> -q`
       Pillow         imaging
   Use these and no others. An item that imports anything outside this list cannot
   pass — there is no network to install it — and it blocks every item behind it.
7. NAME THINGS BY FULL PATH, ALWAYS. Write every file as its complete path from the
   workspace root — `visual_lookup/read/__init__.py`, never `read/__init__.py` and never a
   bare `__init__.py`. The coder is given these paths as the exact set of files it may
   edit, so an incomplete path is either an un-editable item or, worse, a file created in
   the wrong place. A bare `__init__.py` in a specification has already produced a stray
   file one directory off.
8. Say MODULE or PACKAGE explicitly. A module and a package cannot share a name: if both
   exist, Python SILENTLY prefers the package and the module becomes dead code that still
   looks fine. To add a function to something an earlier item created, name the exact file
   — `thing/__init__.py`, not `thing.py`. Getting this wrong has cost this team four items.

THE BRIEF:
{brief}

WORKSPACE: {workspace}

WHAT ALREADY EXISTS IN THE WORKSPACE — do not re-create any of it, and do not assume
anything else is present:
{listing}

Reply with ONE JSON object and no markdown fence:
{{
  "title": "<short plan title>",
  "summary": "<3 sentences: what the finished system does>",
  "items": [
    {{"id": "JV-001",
      "title": "<short imperative title>",
      "detail": "<the exact specification: the file to create, its full path, every
                 function and its signature, the behaviour, and one example of using it>",
      "depends_on": [],
      "verify": "<one shell command>"}}
  ]
}}"""

CODER_PROMPT = """You are the CODER on a small team, working on a real machine. You write complete, working code.

RULES — these are hard:
1. Write COMPLETE file contents. Never emit "..." or "# rest of the file unchanged"
   or a placeholder. Every line of every file you write is in your reply.
2. Only create or modify files inside the workspace. Use paths RELATIVE to it.
3. Prefer creating a NEW file over editing an existing one.
4. Dependencies are FIXED. The sandbox that runs your verify has no network, so use
   only the standard library plus these: pypdf, pdfplumber, PyMuPDF (`import fitz`),
   reportlab, Pillow, pytest. Import anything else and the item cannot pass.
5. Keep it small enough to be correct. A working 40-line file beats a broken 400-line one.
6. Match the interface in the item specification exactly — other items depend on it.

Reply with ONE JSON object and no markdown fence:
{{
  "status": "ok" or "failed",
  "files": [{{"path": "relative/path.py", "content": "<the complete file>"}}],
  "diffs": [],
  "verify": "<one shell command that exits 0 iff your work is correct>",
  "notes": "<two sentences: what you did, and anything you are unsure about>"
}}


YOUR TASK:

WORK ITEM {item_id}: {title}

{detail}

WORKSPACE: {workspace}

DESIGN CONTRACT — the whole plan was reviewed against this and every item must honour it.
Do NOT invent your own names, paths or signatures; other items are being written against
this exact contract at the same time:
{design}

FILES ALREADY PRESENT:
{listing}
{context}"""

REVIEW_PROMPT = """You are the DIRECTOR. You are reviewing a plan for internal consistency BEFORE any code is written.

These items were produced in one session, and a model's ideas drift while it writes. Read
them as ONE design and find every place where they contradict each other or cannot fit
together:

- the same thing given two different names, paths or interfaces
- a file AND a package with the same name (read.py vs read/__init__.py) — this fails
  outright and is a real defect, not a style question
- interface drift: one item defines f(x) -> dict, another calls f(x, y) or expects a list
- two items that both create or claim ownership of the same file
- a later item that assumes an approach an earlier item rules out
- an item that depends on something no item produces

Report ONLY real problems. Do not invent issues. Do not comment on wording or style.
If the plan is sound, say so — a false alarm wastes more time than it saves.

THE PLAN:
{plan}

Reply with ONE JSON object and no markdown fence:
{{
  "consistent": true or false,
  "issues": [
    {{"items": ["<id>", "<id>"],
      "problem": "<what contradicts what, concretely>",
      "fix": "<the single decision that resolves it>",
      "respecs": [
        {{"id": "<the item id to rewrite>",
          "title": "<its corrected title>",
          "detail": "<its FULL corrected specification, precise enough to implement>",
          "verify": "<its corrected verify command>",
          "depends_on": ["<ids>"]}}
      ]}}
  ],
  "design": "<the canonical design, 5-8 sentences: the file layout, the exact public
             function of each file WITH ITS SIGNATURE, and the shape of the data passed
             between them. For EVERY name, state whether it is a module (thing.py) or a
             package (thing/__init__.py) — never leave that to be inferred, because a
             module and a package cannot share a name and Python silently prefers the
             package. Vague naming here has already cost this team three items.
             This becomes the contract every item is implemented against.>"
}}"""

REPAIR_PROMPT = """You are the DIRECTOR. A work item has failed {attempts} times and the coder is stuck.

WORK ITEM {item_id}: {title}

SPECIFICATION:
{detail}

THE ERROR, from running the verify command:
{error}

EVERY ATTEMPT SO FAR, oldest first. Do not propose an approach that has already failed:
{history}

FILES PRESENT IN THE WORKSPACE (the plan may disagree with reality — check carefully,
especially for a name that exists BOTH as a module `x.py` and a package `x/`; the package
wins the import and the module becomes dead code):
{listing}

ALWAYS OFFER OPTIONS. This is a standing instruction from the owner. Never present a single
way forward when there is a real choice: give the human 2 to 4 genuinely different ways to
proceed, each with its cost and its risk stated plainly, and mark the one you recommend.
Options that differ only in wording are not options.

Reply with ONE JSON object and no markdown fence:
{{"verdict": "respec" or "retry" or "abandon",
  "title": "<new title if you are respeccing, else the old one>",
  "detail": "<the new, smaller specification if respeccing; else the old one>",
  "verify": "<the verify command to use>",
  "reason": "<one sentence>",
  "options": [
    {{"label": "<short name for this approach>",
      "what": "<what it would do, concretely>",
      "cost": "<what it takes — time, rework, files touched>",
      "risk": "<what could go wrong, or 'low'>",
      "recommended": true or false,
      "action": {{
        "kind": "exec",
        "command": "<ONE shell command run in the workspace that performs this option —
                     e.g. moving a file. Use an empty string if the option needs no
                     filesystem change.>"
      }} or {{
        "kind": "respec",
        "title": "<the item's new title>",
        "detail": "<the item's new, complete specification>",
        "verify": "<the command that will prove it>"
      }}
    }}
  ],
  "recommendation": "<one sentence saying which option and why>"}}

Each option must be APPLICABLE — picking it has to be enough to proceed. An option whose
action is "someone should think about this" is not an option. If the right answer is simply
to re-specify the item, use the respec action. If it needs a filesystem change first (a file
move, a deletion), use exec and keep it to one command that only touches the workspace."""


# ------------------------------------------------------------------ workspace safety

def _root(workspace: str) -> str:
    return os.path.realpath(os.path.expanduser(workspace))


def safe_path(workspace: str, rel: str) -> str | None:
    """Resolve `rel` inside `workspace`, or return None if it escapes.

    Models write absolute paths, `../..` and symlinks without meaning any harm;
    the check is here so that intent never has to be reasoned about.
    """
    if not rel or os.path.isabs(rel) or "\x00" in rel:
        return None
    root = _root(workspace)
    full = os.path.realpath(os.path.join(root, rel))
    if full != root and not full.startswith(root + os.sep):
        return None
    return full


def list_files(workspace: str) -> str:
    root = _root(workspace)
    if not os.path.isdir(root):
        return "(workspace does not exist)"
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d != "__pycache__"]
        for fn in sorted(filenames):
            # ".bak-" catches our own timestamped backups (good.py.bak-20260913-114500),
            # which endswith(".bak") alone misses.
            if fn.startswith(".") or ".bak-" in fn or fn.endswith((".pyc", ".bak")):
                continue
            rel = os.path.relpath(os.path.join(dirpath, fn), root)
            out.append(rel)
            if len(out) >= FILE_LIST_LIMIT:
                return "\n".join(sorted(out)) + f"\n... ({FILE_LIST_LIMIT}+ files)"
    return "\n".join(sorted(out)) or "(empty)"


def inline_context(workspace: str, detail: str) -> str:
    """Show the coder the contents of small existing files it is likely to need.

    Only files named in the item spec, only if small — this is prompt budget spent
    to prevent the coder from inventing an interface that already exists.
    """
    chunks = []
    for name in sorted(set(re.findall(r"[\w./-]+\.py", detail or ""))):
        full = safe_path(workspace, name)
        if full and os.path.isfile(full) and os.path.getsize(full) <= INLINE_FILE_LIMIT:
            try:
                with open(full, encoding="utf-8", errors="replace") as fh:
                    chunks.append(f"--- existing {name} ---\n{fh.read()}")
            except OSError:
                pass
    return ("\n\n" + "\n\n".join(chunks)) if chunks else ""


# ------------------------------------------------------------------ apply + verify

def _public_names(source: str) -> set[str]:
    """Top-level function and class names — the file's visible surface."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return set()
    return {n.name for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}


def check_no_clobber(full_path: str, new_source: str) -> str | None:
    """Would writing this remove something the file already provides?

    This is the fix for the failure that cost us four verified items. Two work items can
    name the same file — and when the second writes the whole file, it silently deletes
    the first's work. The clobbered item is still marked `verified`, because it *was*
    verified when it was checked, so nothing notices until a later re-verify fails and
    four items are already red.

    A merge is not the answer: trusting a 32B model to reproduce another item's code
    faithfully is the same bet that just lost. Refusing the write and saying exactly what
    would be lost is deterministic, and it turns a silent deletion into a repair the
    director can actually act on.
    """
    try:
        with open(full_path, encoding="utf-8", errors="replace") as fh:
            had = _public_names(fh.read())
    except OSError:
        return None
    if not had:
        return None
    lost = had - _public_names(new_source)
    if lost:
        return (f"this write would REMOVE {', '.join(sorted(lost))} from "
                f"{os.path.basename(full_path)}, which another item put there and which "
                f"is still being verified. Write the complete file — including those — "
                f"or target a different file.")
    return None


def apply_files(workspace: str, files: list, log) -> list[str]:
    """Write the coder's files. Backs up anything it overwrites, and refuses a write
    that would delete another item's work."""
    written = []
    for entry in files or []:
        if not isinstance(entry, dict):
            continue
        rel, content = entry.get("path"), entry.get("content")
        if not rel or content is None:
            log.append(f"skipped malformed file entry: {str(entry)[:120]}")
            continue
        full = safe_path(workspace, rel)
        if full is None:
            log.append(f"REFUSED path outside workspace: {rel}")
            continue
        if os.path.exists(full):
            clobber = check_no_clobber(full, content)
            if clobber:
                log.append(f"REFUSED {rel}: {clobber}")
                continue
            shutil.copy2(full, full + f".bak-{time.strftime('%Y%m%d-%H%M%S')}")
        os.makedirs(os.path.dirname(full) or workspace, exist_ok=True)
        with open(full, "w", encoding="utf-8") as fh:
            fh.write(content)
        written.append(rel)
        log.append(f"wrote {rel} ({len(content)} bytes)")
    return written


def apply_diffs(workspace: str, diffs: list, log) -> bool:
    """git apply the coder's patches. Only meaningful in a git workspace."""
    if not diffs:
        return True
    root = _root(workspace)
    if not os.path.isdir(os.path.join(root, ".git")):
        log.append("diffs supplied but workspace is not a git repo — ignored")
        return False
    ok = True
    for entry in diffs:
        patch = entry.get("patch") if isinstance(entry, dict) else entry
        if not patch:
            continue
        check = subprocess.run(["git", "apply", "--check", "-"], cwd=root, input=patch,
                               text=True, capture_output=True, timeout=60)
        if check.returncode != 0:
            log.append(f"git apply --check failed: {check.stderr.strip()[:300]}")
            ok = False
            continue
        subprocess.run(["git", "apply", "-"], cwd=root, input=patch, text=True,
                       capture_output=True, timeout=60)
        log.append("applied a patch")
    return ok


_CMD_FRAME = re.compile(r'File "<string>", line 1')


def _command_is_at_fault(evidence: str) -> bool:
    """Did the verify command itself fail, rather than the code it was testing?

    A traceback whose deepest frame is `<string>`, line 1 is the -c argument
    blowing up — the nested quotes collapsed, or the command calls something that
    does not exist. Retrying the coder against it produces three identical
    failures and a confused repair; the plan is what needs to change.
    """
    if not _CMD_FRAME.search(evidence or ""):
        return False
    tail = (evidence or "").strip().splitlines()[-1:] or [""]
    return any(k in tail[0] for k in ("NameError", "AttributeError", "TypeError"))


def clean_verify(command: str) -> str:
    """Strip a leading shell prompt that got copied into the command itself.

    Local models copy the `$ ` they see in transcripts and in error reports. Left in,
    the shell tries to run a program called `$` and every attempt fails with
    `/bin/sh: line 1: $: command not found` — which reads like a code bug and sends the
    whole retry loop chasing the wrong thing. It cost two items before this went in.
    """
    return re.sub(r"^\s*\$+\s+", "", command or "").strip()


def run_verify(workspace: str, command: str) -> tuple[bool, str]:
    """Exit 0 or it did not happen — and preferably exit 0 inside a container.

    The container is the point: a verify command is model-authored, and until now it ran
    on the host with the same rights as the worker that owns the database. See sandbox.py
    for what the isolation is and is not.
    """
    command = clean_verify(command)
    if not command:
        return False, "no verify command — refusing to call this done"
    result = sandbox.run(_root(workspace), command, timeout=VERIFY_TIMEOUT)
    return result.ok, result.output


# A bare `print(...)` exits 0 whether or not the code works; `ls missing.py` does not.
_ASSERTIVE = re.compile(
    r"\bassert\b|\btest\b|\bgrep\b|\bdiff\b|\bcmp\b|\bexit\b|==|!=|\bpytest\b|"
    r"\bpy_compile\b|\bsys\.exit\b|\[ -[fedrwxs]")

# Checks that assert a *name exists* rather than that anything *works*. A stub
# (`def read_image(b): raise NotImplementedError`) passes every one of these, so
# an item carrying only this kind of check can be marked verified while being empty.
_SHALLOW = re.compile(r"assert\s+(callable|hasattr)\s*\(|^\s*ls\s+\S+\s*$")
_SHALLOW_ISINSTANCE = re.compile(r"assert\s+isinstance\s*\(")


def is_shallow(command: str) -> bool:
    """A bare `assert isinstance(x, str)` proves only that a function returned
    something. `assert isinstance(x, str) and 'resistor' in x` proves it returned
    the right thing — so a type check only counts as shallow on its own."""
    if _SHALLOW.search(command or ""):
        return True
    if not _SHALLOW_ISINSTANCE.search(command or ""):
        return False
    return not re.search(r"==|!=|\bin\b|\[.+\]|\.[a-z_]+\(|[<>]", command or "")


def lint_verify(command: str, workspace: str | None = None) -> tuple[bool, str]:
    """Decide whether a verify command is capable of failing. Returns (ok, reason).

    Two questions, most decisive first:

    1. **Does the shell parse it at all?** `python3 -c 'f('x')'` is a syntax error,
       so it can never exit 0 no matter how good the code is. Burning three coder
       attempts on that is waste — and worse, it teaches the repair loop that the
       *code* is at fault when the *plan* is. A planning defect has to be routed
       back to the planner.
    2. **Does it already pass?** Run it against the workspace as it stands, before
       any work exists. If it exits 0 now, it cannot distinguish done from
       not-done, and "verified" would be a lie.
    """
    command = clean_verify(command)
    if not command:
        return False, "empty verify command"
    try:
        p = subprocess.run(["bash", "-n", "-c", command], text=True,
                           capture_output=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"could not syntax-check the command: {exc}"
    if p.returncode != 0:
        tail = (p.stderr or "").strip().splitlines()
        return False, f"does not parse as a shell command ({tail[-1][:160] if tail else 'syntax error'})"
    malformed = _malformed_command(command)
    if malformed:
        return False, malformed
    return True, "ok"


def already_passes(command: str, workspace: str | None) -> bool:
    """Does this command exit 0 against the workspace as it stands right now?

    Deliberately NOT part of `lint_verify`. "It already passes" is ambiguous: it may
    mean the check is broken, or it may mean the work was finished by an earlier run.
    Treating it as a hard failure would send finished items back to the director for
    re-speccing forever. It is reported as a warning and left to the human.
    """
    if not command or not workspace or not os.path.isdir(_root(workspace)):
        return False
    return run_verify(workspace, command)[0]


_SYNTAX_SIGNS = re.compile(
    r"<string>, line|SyntaxError.*<string>|unexpected EOF while looking for matching|"
    r"unterminated quoted string")


def _malformed_command(command: str) -> str | None:
    """Run the command where nothing exists. If it fails with a *syntax* error in the
    command text — rather than a missing module — it can never pass, for any code.

    `bash -n` is not enough here: `python3 -c 'f('x')'` is perfectly valid bash
    (it concatenates the quoted parts) and only falls over inside Python. Running it
    in an empty directory separates "the command is broken" from "the file is missing".
    """
    try:
        with tempfile.TemporaryDirectory() as scratch:
            ok, evidence = run_verify(scratch, command)
    except OSError:
        return None
    if ok:
        return None
    return ("command text is malformed (broken quoting) — it cannot pass for any code"
            if _SYNTAX_SIGNS.search(evidence) else None)


# A verify can be well-formed and still be impossible: it may import a package that is
# not installed and that no item in the plan creates. The sandbox refuses installs by
# design, so an item like that cannot pass however good the coder is — it just burns the
# plan's budget and blocks everything behind it.
_IMPORT_CLAUSE_RE = re.compile(r"\bimport\s+([^;\n]+)")
_FROM_IMPORT_RE = re.compile(r"\bfrom\s+([A-Za-z_][\w.]*)\s+import\b")
_RUN_MODULE_RE = re.compile(r"\bpython3?\s+-m\s+([A-Za-z_][\w.]*)")
_CREATES_PATH_RE = re.compile(r"\b([A-Za-z_][\w]*)/(?:[\w.-]+/)*[\w.-]+\.py\b|\b([A-Za-z_][\w]*)\.py\b")


def _plan_provides(conn, plan_id: str) -> set:
    """Top-level module names the plan itself creates, read off the paths in its items."""
    provides = set()
    for item in jarvis_db.list_items(conn, plan_id):
        text = f"{item['title'] or ''}\n{item['detail'] or ''}"
        for pkg, mod in _CREATES_PATH_RE.findall(text):
            provides.add(pkg or mod)
    return provides


def _module_available(name: str) -> bool:
    if name in sys.stdlib_module_names:
        return True
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _imported_names(command: str) -> set:
    """Top-level module names a verify command imports.

    Handles `import a, b, c` as well as `import a` — capturing only the first name of a
    comma list (as this did at first) means a package imported second is never checked.
    """
    names = set(_FROM_IMPORT_RE.findall(command))
    # Strip `from X import a, b` first: `a` and `b` are attributes of X, not modules,
    # and the bare `import` regex below would otherwise collect them as if they were.
    for clause in _IMPORT_CLAUSE_RE.findall(_FROM_IMPORT_RE.sub(" ", command)):
        for part in clause.split(","):
            token = part.strip().split(" as ")[0].strip()
            if re.fullmatch(r"[A-Za-z_][\w.]*", token):
                names.add(token)
    names |= set(_RUN_MODULE_RE.findall(command))
    return {n.split(".")[0] for n in names if n}


def _modules_missing_in_sandbox(names) -> set:
    """Which of these modules cannot be imported WHERE THE VERIFIES ACTUALLY RUN.

    Asking this host's Python (as the first version of this did) answers the wrong
    question: verifies run inside the sandbox image, so a library can be present in one
    and absent from the other. sandbox.run already targets wherever verifies will run —
    the container when there is one, the host when there is not — so ask through it.
    """
    names = sorted({n.split(".")[0] for n in names if n})
    if not names:
        return set()

    # The probe answers in JSON on stdout. It is parsed from after the run's OUTPUT:
    # marker and never from the whole blob: sandbox.run echoes the command, and the
    # command text contains every module name being asked about — reading the whole
    # blob back is how this first reported the stdlib `base64` as missing.
    ws = tempfile.mkdtemp(prefix="dt-lint-")
    try:
        with open(os.path.join(ws, "_lint_modules.py"), "w") as fh:
            fh.write("import importlib.util as u, sys, json\n"
                     "print(json.dumps(sorted(n for n in sys.argv[1:]\n"
                     "                       if u.find_spec(n) is None)))\n")
        res = sandbox.run(ws, "python3 _lint_modules.py " + " ".join(names), timeout=120)
    except Exception:  # noqa: BLE001  — a broken sandbox must not break the lint
        return {n for n in names if not _module_available(n)}
    finally:
        shutil.rmtree(ws, ignore_errors=True)

    tail = res.output.rpartition("OUTPUT:")[2]
    try:
        reported = set(json.loads([l for l in tail.splitlines() if l.strip()][-1]))
    except (ValueError, IndexError):  # no OUTPUT: marker (host fallback) — be honest
        return {n for n in names if not _module_available(n)}
    return {n for n in names if n in reported}


def lint_environment(plan_id: str, log=print) -> int:
    """Count verify commands that depend on a package the environment does not have.

    A missing dependency is a HARD failure, not a warning: the sandbox has no network,
    so such an item cannot pass however good the coder is — it burns the plan's budget
    and blocks everything behind it. This is what let a plan built on pdfplumber reach
    `approved` on a machine that had no pdfplumber.
    """
    conn = jarvis_db.open_db()
    provides = _plan_provides(conn, plan_id)
    per_item = {}
    for item in jarvis_db.list_items(conn, plan_id):
        cmd = item["verify"] or ""
        per_item[item["id"]] = (cmd, _imported_names(cmd) - provides)

    wanted = {n for _, names in per_item.values() for n in names}
    missing = _modules_missing_in_sandbox(wanted)

    count = 0
    for item_id, (cmd, names) in per_item.items():
        for name in sorted(names & missing):
            count += 1
            log(f"  MISSING   {item_id:<12} needs '{name}' — the verify environment has "
                f"no network and does not have it")
            log(f"            {cmd[:120]}")
    return count


def lint_plan(plan_id: str, log=print) -> int:
    """Report which verify commands cannot do their job. Returns the hard-failure count."""
    conn = jarvis_db.open_db()
    items = jarvis_db.list_items(conn, plan_id)
    unusable = weak = 0
    for item in items:
        ok, why = lint_verify(item["verify"] or "", item["workspace"])
        if not ok:
            unusable += 1
            log(f"  UNUSABLE  {item['id']:<12} {why}")
            log(f"            {item['verify'][:120]}")
        elif not _ASSERTIVE.search(item["verify"] or ""):
            weak += 1
            log(f"  weak      {item['id']:<12} no assertion — may not be able to fail")
            log(f"            {item['verify'][:120]}")
        elif is_shallow(item["verify"] or ""):
            weak += 1
            log(f"  shallow   {item['id']:<12} only checks that a name exists, "
                f"not that anything works — a stub would pass this")
            log(f"            {item['verify'][:120]}")
        elif already_passes(item["verify"], item["workspace"]):
            log(f"  done?     {item['id']:<12} already exits 0 — either the check is "
                f"vacuous, or this item is already finished")
    env_missing = lint_environment(plan_id, log=log)
    solid = len(items) - unusable - weak - env_missing
    log(f"  {len(items)} item(s): {solid} solid, {weak} weak, {unusable} unusable, "
        f"{env_missing} needing a package this machine lacks")
    return unusable + env_missing


# ------------------------------------------------------------------ planning

def do_plan(brief: str, workspace: str, plan_id: str, title_hint: str | None = None) -> str:
    conn = jarvis_db.open_db()
    prompt = DIRECTOR_PROMPT.format(brief=brief, workspace=_root(workspace),
                                   listing=list_files(workspace))
    reply, ms = llm.timed_chat("director", prompt, max_tokens=DIRECTOR_TOKENS,
                               temperature=0.4)
    jarvis_db.log_run(conn, role="director", prompt=prompt, output=reply, ok=bool(reply),
                      ms=ms, plan_id=plan_id, engine=":8081")

    data = llm.extract_json(reply)
    if not data or not data.get("items"):
        raise SystemExit(f"director returned no usable plan.\n--- raw reply ---\n{reply[:2500]}")

    jarvis_db.create_plan(conn, plan_id, title_hint or data.get("title") or brief[:60],
                          brief, data.get("summary") or "")

    # The director numbers its items JV-001, JV-002... locally, so a second plan
    # generates the *same* ids and silently overwrites the first plan's items
    # (they share one primary key). Namespace every id under its plan, and rewrite
    # the dependency references through the same map so ordering survives.
    items = [i for i in data["items"] if isinstance(i, dict) and i.get("title")]
    idmap = {}
    for n, item in enumerate(items, start=1):
        raw = str(item.get("id") or f"{n:03d}").strip()
        idmap[raw] = raw if raw.startswith(f"{plan_id}:") else f"{plan_id}:{raw}"

    # Re-planning the same id replaces the work still to do, but never discards
    # items that already verified — that is finished work, not a draft.
    stale = [i["id"] for i in jarvis_db.list_items(conn, plan_id)
             if i["status"] in ("pending", "in-progress")]
    if stale:
        print(f"  re-planning {plan_id}: replacing {len(stale)} unfinished item(s)")
        with conn:
            conn.executemany("DELETE FROM work_items WHERE id=?", [(s,) for s in stale])

    kept = 0
    for n, item in enumerate(items, start=1):
        raw = str(item.get("id") or f"{n:03d}").strip()
        deps = [idmap.get(str(d).strip(), str(d).strip()) for d in (item.get("depends_on") or [])]
        jarvis_db.add_item(conn, {
            "id": idmap[raw], "plan_id": plan_id, "ordinal": n,
            "title": item["title"][:200],
            "detail": item.get("detail") or "",
            "depends_on": deps,
            "verify": clean_verify(item.get("verify")),
            "workspace": _root(workspace),
            "status": "pending", "owner": "coder",
        })
        kept += 1
    print(f"plan {plan_id}: {kept} item(s)")
    print("  checking every verify command can actually fail:")
    lint_plan(plan_id, log=print)
    print(f"  next:  python3 devteam.py review {plan_id}    (architect sanity check)")
    print(f"  then:  python3 devteam.py approve {plan_id}")
    return plan_id


# ------------------------------------------------------------------ design review

def _apply_respecs(conn, plan_id: str, issues: list, log) -> tuple[int, str]:
    """Apply the architect's own corrected specifications. Returns (applied, needs_owner).

    Routine repairs — naming, layout, interfaces, a wrong import path — are made here.
    A repair that would touch a container or an LLM setting is NOT: that is the owner's
    call, and the second return value says which one stopped it.
    """
    applied = 0
    for issue in issues:
        for spec in (issue.get("respecs") or []):
            if not isinstance(spec, dict) or not spec.get("id"):
                continue
            item_id = str(spec["id"]).strip()
            if ":" not in item_id:
                item_id = f"{plan_id}:{item_id}"
            row = conn.execute("SELECT * FROM work_items WHERE id=?", (item_id,)).fetchone()
            if row is None:
                log(f"    cannot repair unknown item {item_id}")
                continue
            major, why = classify_major(json.dumps(spec))
            if major:
                return applied, f"{item_id} ({why})"
            jarvis_db.add_item(conn, {
                "id": item_id, "plan_id": row["plan_id"], "ordinal": row["ordinal"] or 0,
                "title": (spec.get("title") or row["title"])[:200],
                "detail": spec.get("detail") or row["detail"],
                "verify": clean_verify(spec.get("verify") or row["verify"]),
                "workspace": row["workspace"],
                "depends_on": spec.get("depends_on") or row["depends_on"],
                "status": "pending", "owner": "coder", "options": "",
                "notes": "repaired by the consistency review: "
                         + (issue.get("fix") or "")[:200],
            })
            applied += 1
            log(f"    repaired {item_id}")
    return applied, ""


def do_review(plan_id: str, fix: bool = False, rounds: int = 3, log=print) -> bool:
    """The architect sanity-checks its own plan before any code is written.

    Why this exists: a model writing twelve items in one pass drifts. Item 2 assumes a
    module called `reader.py`, item 7 imports `read.py`; an early item says the function
    returns a dict and a late item expects a list. Each item looks fine alone and the
    plan is unsound as a whole — and the coder, seeing one item at a time, cannot notice.

    So the same plan is handed back to the director as a *single design* to be checked
    for contradictions, and the review produces the canonical design contract that every
    item is then implemented against. That contract is what keeps early and late items
    from describing two incompatible systems.

    With `fix=True` the review also *repairs* what it finds, by applying the corrected
    item specifications the architect supplies. Naming and layout inconsistencies are
    routine decisions — the owner has said explicitly that halting on them is a defect,
    not caution — so they are fixed here and only genuinely major changes stop for a
    person. `rounds` bounds the repair/re-review loop.
    """
    conn = jarvis_db.open_db()
    plan = jarvis_db.get_plan(conn, plan_id)
    if plan is None:
        raise SystemExit(f"no such plan: {plan_id}")
    items = jarvis_db.list_items(conn, plan_id)
    if not items:
        raise SystemExit(f"plan {plan_id} has no items to review")
    if rounds <= 0:
        log("  repair loop exhausted — the plan still contradicts itself; leaving it "
            "inconsistent rather than guessing again")
        return False

    listing = "\n\n".join(
        f"{i['id']} — {i['title']}\n"
        f"  detail: {(i['detail'] or '').strip()[:1200]}\n"
        f"  verify: {i['verify']}\n"
        f"  depends_on: {i['depends_on'] or 'none'}"
        for i in items)

    prompt = REVIEW_PROMPT.format(plan=listing)
    log(f"  architect is checking {len(items)} items for consistency…")
    reply, ms = llm.timed_chat("director", prompt, max_tokens=3000, temperature=0.2)
    jarvis_db.log_run(conn, role="director", prompt=prompt, output=reply, ok=bool(reply),
                      ms=ms, plan_id=plan_id, engine=":8081")

    data = llm.extract_json(reply)
    if not data:
        raise SystemExit(f"architect returned no usable review.\n--- raw ---\n{reply[:2000]}")

    design = (data.get("design") or "").strip()
    issues = [i for i in (data.get("issues") or []) if isinstance(i, dict)]
    consistent = bool(data.get("consistent"))

    # A review that produces no design contract has not done the work, whatever it
    # claims about consistency — there would be nothing for the items to agree on.
    if not design:
        consistent = False
        log("  the review produced no design contract — treating as NOT consistent")

    jarvis_db.set_plan_review(conn, plan_id, design, consistent)

    if issues and fix:
        applied, stopped = _apply_respecs(conn, plan_id, issues, log)
        if applied and not stopped:
            log(f"\n  repaired {applied} item(s) — re-reviewing")
            return do_review(plan_id, fix=fix, rounds=rounds - 1, log=log)
        if stopped:
            log(f"\n  {stopped} needs the owner — not applying it automatically")

    log("")
    log("  CANONICAL DESIGN (every item is implemented against this):")
    for line in _wrap(design, 96):
        log(f"    {line}")
    log("")
    if issues:
        log(f"  {len(issues)} contradiction(s) found — the coder will NOT start:")
        for i, issue in enumerate(issues, 1):
            who = ", ".join(str(x) for x in (issue.get("items") or [])) or "?"
            log(f"    {i}. [{who}] {issue.get('problem', '')}")
            log(f"       fix: {issue.get('fix', '')}")
    else:
        log("  no contradictions found.")
    log("")
    log(f"  verdict: {'CONSISTENT — clear to approve' if consistent else 'INCONSISTENT — resolve before coding'}")
    return consistent


def _wrap(text: str, width: int) -> list[str]:
    import textwrap
    out = []
    for para in (text or "").splitlines() or [""]:
        out.extend(textwrap.wrap(para, width) or [""])
    return out


# ------------------------------------------------------------------ execution

def _attempts(conn, item_id: str) -> int:
    return conn.execute("SELECT count(*) n FROM runs WHERE item_id=? AND role='coder'",
                        (item_id,)).fetchone()["n"]


def _editor_instruction(item, design: str) -> str:
    """What the editor is told. It gets the same design contract the coder path used —
    a swap of executor should not quietly downgrade the item's context."""
    parts = [f"WORK ITEM {item['id']}: {item['title']}", "", (item["detail"] or "").strip()]
    if design:
        parts += ["", "DESIGN CONTRACT — the whole plan was reviewed against this. Do not "
                      "invent your own names, paths or signatures; other items are being "
                      "written against this exact contract:", design]
    parts += ["", "Make ONLY the change described above. Do not refactor, rename or delete "
                  "anything that is already in the file — other work items depend on it."]
    return "\n".join(parts)


def _editor_attempt(conn, item, design: str, workspace: str, last_error: str, log):
    """One attempt via the editor container. Returns (error, suggested verify).

    Same contract as _builtin_attempt. The editor edits files in place, so there is
    no suggested verify to offer — the item's own verify is the check.
    """
    instruction = _editor_instruction(item, design)
    if last_error:
        instruction += (f"\n\nThe previous attempt failed. The verify command reported:"
                        f"\n{last_error[-1500:]}\n\nFix that specific failure.")
    res = editor.run(workspace, instruction)
    jarvis_db.log_run(conn, role="editor", prompt=instruction, output=res.transcript,
                      ok=res.ok, ms=int(res.seconds * 1000), plan_id=item["plan_id"],
                      item_id=item["id"], engine="aider")
    if res.changed:
        log(f"    edited: {', '.join(res.changed)}  [{res.seconds:.0f}s]")
    if res.ok:
        return "", None
    return (res.error or "the editor made no change"), None


def _builtin_attempt(conn, item, design: str, workspace: str, last_error: str, log):
    """The original path: ask the coder lane for whole files as JSON, write them here.

    Kept as a working fallback rather than deleted — and kept honest by the same clobber
    guard, which is what makes it survivable at all. Returns (error, suggested verify).
    """
    prompt = CODER_PROMPT.format(
        item_id=item["id"], title=item["title"], detail=item["detail"] or "",
        workspace=_root(workspace), listing=list_files(workspace),
        context=inline_context(workspace, item["detail"] or ""),
        design=design or "(no design contract recorded)")
    if last_error:
        prompt += (f"\n\nYOUR PREVIOUS ATTEMPT FAILED. The verify command was run and "
                   f"reported:\n{last_error[-1500:]}\n\nFix that specific failure.")

    try:
        reply, ms = llm.timed_chat("coder", prompt, max_tokens=4000, temperature=0.2)
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
        log(f"    lane error: {err}")
        return err, None

    jarvis_db.log_run(conn, role="coder", prompt=prompt, output=reply, ok=True, ms=ms,
                      plan_id=item["plan_id"], item_id=item["id"], engine=":8082")

    data = llm.extract_json(reply)
    if not data:
        log("    reply was not parseable JSON")
        return "reply was not parseable JSON", None

    aplog: list[str] = []
    apply_files(workspace, data.get("files") or [], aplog)
    apply_diffs(workspace, data.get("diffs") or [], aplog)
    for line in aplog:
        log(f"    {line}")

    refused = [line for line in aplog if line.startswith("REFUSED")]
    if refused:
        # Tell the coder what it nearly destroyed, in its own retry prompt. This reads
        # as a correction rather than a mystery verify failure.
        log("    a write was refused; feeding that back")
        return ("your write was refused to protect another item's work:\n"
                + "\n".join(refused)), None
    return "", data.get("verify")


def run_item(conn, item, log=print) -> str:
    """One work item, start to finish. Returns the final status."""
    item_id = item["id"]
    workspace = item["workspace"] or os.getcwd()
    plan = jarvis_db.get_plan(conn, item["plan_id"]) if item["plan_id"] else None
    design = (plan["design"] if plan is not None else "") or ""
    jarvis_db.set_item(conn, item_id, status="in-progress")
    log(f"\n=== {item_id}: {item['title']}")

    attempts = 0
    last_error = ""
    history: list[str] = []

    # A verify that cannot fail is a planning defect. Handing it to the coder
    # would burn three attempts proving nothing, so go straight to the director.
    lint_ok, lint_why = lint_verify(item["verify"] or "", workspace)
    if not lint_ok:
        log(f"  verify is unusable ({lint_why}) — routing to the director, not the coder")
        last_error = (f"this item's verify command is unusable: {lint_why}\n"
                      f"command was: {item['verify']!r}")
        attempts = MAX_ATTEMPTS

    while attempts < MAX_ATTEMPTS:
        attempts += 1
        via_editor = executor_for_attempt(attempts)
        log(f"  attempt {attempts}/{MAX_ATTEMPTS} via {'editor' if via_editor else 'coder lane'}")
        if via_editor:
            err, suggested = _editor_attempt(conn, item, design, workspace, last_error, log)
        else:
            err, suggested = _builtin_attempt(conn, item, design, workspace, last_error, log)
        if err:
            last_error = err
            history.append(f"attempt {attempts} ({'editor' if via_editor else 'coder lane'})"
                           f" made no usable change: {err[:600]}")
            log(f"    attempt failed: {err[:160]}")
            continue

        verify_cmd = clean_verify(item["verify"] or suggested or "")
        ok, evidence = run_verify(workspace, verify_cmd)
        log(f"    verify: {'PASS' if ok else 'FAIL'}  ({verify_cmd[:70]})")
        if ok:
            jarvis_db.set_item(conn, item_id, status="verified", evidence=evidence,
                               notes=f"verified on attempt {attempts} via "
                                     f"{'the editor' if use_editor() else 'the coder lane'}")
            log(f"  -> {item_id} verified")
            return "verified"
        last_error = evidence
        history.append(f"attempt {attempts} ({'editor' if via_editor else 'coder lane'}) "
                       f"changed files but the verify still failed:\n{evidence[:800]}")

        # Some broken commands only reveal themselves once the module they import
        # actually exists — `print(f('file'))` is valid Python right up until `f`
        # resolves. When the deepest traceback frame is the -c string itself, the
        # fault is in the command, not the code, and the coder cannot fix it.
        if _command_is_at_fault(evidence):
            log("    failure is in the verify command itself, not the code — routing to the director")
            break

    # Out of attempts: consult the director once, rather than grinding.
    log(f"  exhausted {MAX_ATTEMPTS} attempts — asking the director")
    repair_prompt = REPAIR_PROMPT.format(attempts=attempts + 1, item_id=item_id,
                                         title=item["title"], detail=item["detail"] or "",
                                         error=last_error[-2000:],
                                         history="\n\n".join(history) or "(none recorded)",
                                         listing=list_files(workspace))
    try:
        rreply, rms = llm.timed_chat("director", repair_prompt, max_tokens=REPAIR_TOKENS,
                                     temperature=0.3)
        jarvis_db.log_run(conn, role="director", prompt=repair_prompt, output=rreply, ok=True,
                          ms=rms, plan_id=item["plan_id"], item_id=item_id, engine=":8081")
        rdata = llm.extract_json(rreply) or {}
    except Exception as exc:  # noqa: BLE001
        rdata = {}
        log(f"    director unavailable: {exc}")

    # The owner's standing instruction: never a single way forward when there is a real
    # choice. Keep the options with the item so the board and the human can see them.
    raw_opts = [o for o in (rdata.get("options") or []) if isinstance(o, dict)]
    options_json = json.dumps(raw_opts) if raw_opts else ""
    options_text = _options_block(rdata)
    if options_text:
        log("  options the architect offered:")
        for n, line in enumerate(options_text.splitlines()[1:], 1):
            log(f"    {line}")

    verdict = (rdata.get("verdict") or "abandon").lower()

    # Persist the options first — apply_option reads them back from the row.
    if raw_opts:
        jarvis_db.set_item(conn, item_id, options=options_json)

    if verdict == "respec" and rdata.get("detail"):
        jarvis_db.add_item(conn, {
            "id": item_id, "plan_id": item["plan_id"], "ordinal": item["ordinal"],
            "title": (rdata.get("title") or item["title"])[:200],
            "detail": rdata["detail"],
            "verify": clean_verify(rdata.get("verify") or item["verify"]),
            "workspace": workspace, "depends_on": item["depends_on"],
            "status": "pending", "owner": "coder",
            "notes": f"respec after {attempts} failures: {rdata.get('reason', '')[:300]}",
        })
        log(f"  -> {item_id} re-specced, will retry")
        return "respec"

    # The architect's own recommendation, taken on the team's own authority — unless it
    # falls in one of the categories the owner asked to be consulted about.
    rec_index = next((n for n, o in enumerate(raw_opts, 1) if o.get("recommended")), None)
    if rec_index:
        rec = raw_opts[rec_index - 1]
        major, why = classify_major(
            f"{rec.get('label', '')} {rec.get('what', '')} "
            f"{json.dumps(rec.get('action') or {})}")
        if major:
            notes = ((rdata.get("reason") or "")[:300]
                     + f"\n\nNEEDS THE OWNER — {why}\n" + _options_block(rdata)).strip()
            jarvis_db.set_item(conn, item_id, status="blocked", evidence=last_error,
                               notes=notes, options=options_json)
            log(f"  -> {item_id} BLOCKED — needs the owner: {why}")
            return "blocked"
        if _auto_applies(conn, item_id) < AUTO_APPLY_LIMIT:
            log("  taking the architect's recommendation on its own authority "
                "(routine — not a container or LLM change)")
            outcome = apply_option(item_id, rec_index, log=log)
            jarvis_db.log_run(conn, role="auto",
                              prompt=f"auto-applied recommended option {rec_index}",
                              output=outcome, ok=outcome not in ("failed", "refused"),
                              ms=0, plan_id=item["plan_id"], item_id=item_id, engine="auto")
            log(f"  -> {item_id} {outcome}; retrying")
            return "auto"

    notes = (rdata.get("reason") or "")[:400]
    if options_text:
        notes = (notes + "\n\n" + options_text).strip()
    jarvis_db.set_item(conn, item_id, status="blocked", evidence=last_error,
                       notes=notes, options=options_json)
    log(f"  -> {item_id} BLOCKED")
    if raw_opts:
        log(f"     {len(raw_opts)} option(s) offered — choose one at /team/architect")
    return "blocked"


# An option's exec action is a shell command written by a language model and fired by a
# button on a web page. That combination earns a gate.
_EXEC_DENY = re.compile(
    r"\bsudo\b|\bsu\b|\bdoas\b|rm\s+-[a-z]*r[a-z]*f?\s+/|/etc/|/usr/|/boot/|"
    r"\bcurl\b|\bwget\b|\bnc\b|\bncat\b|\bssh\b|\bscp\b|\bdd\b|\bmkfs\b|\bmount\b|"
    r"\.\./|`|\$\(|\|\s*(ba)?sh\b|>\s*/|>>\s*/|;\s*rm\b|&&\s*rm\s+-[a-z]*r")


def safe_exec(command: str, workspace: str) -> tuple[bool, str]:
    """Is this command safe to apply from a stated option? Returns (ok, reason).

    The gate is a denylist plus an allowlist of verbs, because the honest situation is
    that a model-authored shell command cannot be proven safe automatically. What this
    does catch: reaching outside the workspace, parent traversal, the network, and
    privilege escalation. What it does NOT catch: a `python3 -c` one-liner that does
    something foolish. The mitigation for that is that every exec is logged to `runs`
    and shown on the board — it is reviewable rather than silent.
    """
    cmd = clean_verify(command)
    if not cmd:
        return False, "empty command"
    if _EXEC_DENY.search(cmd):
        return False, ("reaches outside the workspace (absolute path, parent traversal, "
                       "network, or privilege escalation)")
    if not cmd.startswith(("mv ", "cp ", "rm ", "mkdir ", "rmdir ", "touch ", "ln ",
                           "python3 ", "git ")):
        return False, ("only plain file moves and python/git commands in the workspace "
                       "are allowed")
    return True, "ok"


# ---------------------------------------------------------------- autonomy boundary
#
# Owner's instruction, 2026-09-13:
#
#   "I don't care about small decisions like that. I want the architect and coder to
#    solve problems like that on their own and give me a working product. I want to know
#    if they plan to erase or rewrite a container or make a major change to an LLM like
#    quantization or kv cache size. I don't care about little things that aren't going to
#    change a major feature."
#
# So the team stops for a human only in the named categories. Everything else — file
# layout, module vs package, import paths, naming, re-speccing a stuck item — is the
# team's own call, and halting on it is a defect, not caution.
_MAJOR_PATTERNS = (
    (re.compile(r"\b(podman|docker|compose)\b.{0,40}\b(rm|kill|stop|prune|rebuild|recreate|down)\b",
                re.I | re.S), "container operation"),
    (re.compile(r"\b(erase|delete|remove|rewrite|recreate|rebuild)\b.{0,40}\bcontainer\b"
                r"|\bcontainer\b.{0,40}\b(erase|delete|remove|rewrite|recreate|rebuild)\b",
                re.I | re.S), "container change"),
    (re.compile(r"quantiz|\bkv[_ -]?cache\b|--ctx-size|--cache-type|\bn-gpu-layers\b|"
                r"\bgguf\b|context (size|window)|max_tokens\s*=|gpu_layers", re.I),
     "LLM runtime change (quantization / context / KV cache)"),
    (re.compile(r"git\s+(reset\s+--hard|push\s+--force|clean\s+-[a-z]*f|filter-branch)",
                re.I), "destructive git operation"),
    (re.compile(r"\brm\s+-[a-z]*[rf][a-z]*\s+\S*(manifest-paradigm|/repo|\.git|worktrees)",
                re.I), "destructive delete outside a scratch workspace"),
    (re.compile(r"\b(drop\s+table|truncate\s+table|delete\s+from)\b", re.I),
     "destructive database operation"),
)

#: How many times one item may take its own recommended option before it must ask.
AUTO_APPLY_LIMIT = int(os.getenv("DEVTEAM_AUTO_APPLY_LIMIT", "2"))

#: Who writes the code.
#:   aider   — the editor container (default when its image is present)
#:   builtin — ask the coder lane for whole files as JSON and write them ourselves
#:   auto    — aider when available, builtin otherwise
#: The builtin path is kept because it still works and because a fallback that has been
#: deleted is not a fallback.
EXECUTOR = os.getenv("DEVTEAM_EXECUTOR", "auto").strip().lower()


def use_editor() -> bool:
    if EXECUTOR == "builtin":
        return False
    if EXECUTOR == "aider":
        return True
    return editor.available()


def executor_for_attempt(attempt: int) -> bool:
    """Should THIS attempt go through the editor container?

    The two executors fail differently: the editor path breaks on edit-format drift and
    the builtin path on whole-file JSON. Retrying the same one three times spends all
    three attempts inside a single failure mode, so under "auto" the second attempt
    switches. An operator who pinned DEVTEAM_EXECUTOR gets exactly what they pinned.
    """
    if EXECUTOR == "builtin":
        return False
    if EXECUTOR == "aider":
        return True
    if not editor.available():
        return False          # only one path exists on this machine
    return attempt == 1


def classify_major(text: str) -> tuple[bool, str]:
    """Does this decision need the owner? Returns (major, reason).

    Erring toward `major` costs a question; erring toward `routine` can cost a container
    or a rebuilt model. The asymmetry is why the patterns are blunt and the default for
    anything unrecognised-but-destructive is to ask.
    """
    blob = text or ""
    for pattern, why in _MAJOR_PATTERNS:
        if pattern.search(blob):
            return True, why
    return False, ""


def _auto_applies(conn, item_id: str) -> int:
    return conn.execute("SELECT count(*) n FROM runs WHERE item_id=? AND role='auto'",
                        (item_id,)).fetchone()["n"]


def do_reverify(plan_id: str, log=print) -> dict:
    """Re-run every verified item's verify command, and demote anything that now fails.

    This exists because verification is a point-in-time fact, not a property. Item 2 is
    verified; item 3 then creates a package with the same name as item 2's module, and
    Python silently prefers the package — item 2's functions become unreachable while
    its row still says `verified`. Nothing was wrong when it was checked; the ground
    moved afterwards.

    Catching that is the difference between "verified" meaning something and it being a
    note that the machine was once happy.
    """
    conn = jarvis_db.open_db()
    tally = {"ok": 0, "regressed": 0, "skipped": 0}
    for item in jarvis_db.list_items(conn, plan_id):
        if item["status"] not in ("verified", "complete"):
            tally["skipped"] += 1
            continue
        ok, evidence = run_verify(item["workspace"], item["verify"])
        if ok:
            tally["ok"] += 1
            continue
        tally["regressed"] += 1
        jarvis_db.set_item(conn, item["id"], status="pending", evidence=evidence[-4000:],
                           notes="REGRESSED: this passed when it was built and no longer "
                                 "does — something added afterwards broke it")
        log(f"  REGRESSED  {item['id']}  {item['title']}")
        log(f"             {evidence.strip().splitlines()[-1][:170]}")
    log(f"  {tally['ok']} still good, {tally['regressed']} regressed, "
        f"{tally['skipped']} not previously verified")
    return tally


def apply_option(item_id: str, index: int, log=print) -> str:
    """Do what the chosen option says. This is what the panels' buttons call.

    `index` is 1-based, exactly as shown to the human.

    Three kinds of action, in order of how much the machine can carry on its own:

      exec    — run one shell command in the workspace (a file move, a deletion), then
                reopen the item so the coder retries against the new reality
      respec  — rewrite the item from the option's own specification, then reopen it
      (none)  — record the choice and stop; a human has to do this one

    Whatever happens is written to the runs table, so the board shows what the button did
    rather than merely that someone clicked it.
    """
    conn = jarvis_db.open_db()
    row = conn.execute("SELECT * FROM work_items WHERE id=?", (item_id,)).fetchone()
    if row is None:
        raise SystemExit(f"no such item: {item_id}")

    try:
        opts = json.loads(row["options"] or "[]")
    except ValueError:
        opts = []
    if not opts:
        raise SystemExit(f"{item_id} has no options recorded — "
                         f"run: python3 devteam.py options {item_id}")
    if not isinstance(index, int) or not 1 <= index <= len(opts):
        raise SystemExit(f"option {index} out of range (1..{len(opts)})")

    opt = opts[index - 1]
    action = opt.get("action") or {}
    kind = str(action.get("kind") or "manual").lower()
    workspace = row["workspace"] or os.getcwd()
    label = opt.get("label") or f"option {index}"
    log(f"applying option {index} for {item_id}: {label}")

    if kind == "exec":
        cmd = clean_verify(action.get("command") or "")
        ran = "no filesystem change needed"
        if cmd:
            allowed, why = safe_exec(cmd, workspace)
            if not allowed:
                log(f"  REFUSED: {why}")
                log(f"  command was: {cmd}")
                jarvis_db.set_item(conn, item_id,
                                   notes=f"option {index} ({label}) was refused: {why}")
                return "refused"
            ok, evidence = run_verify(workspace, cmd)
            jarvis_db.log_run(conn, role="option-exec", prompt=cmd, output=evidence,
                              ok=ok, ms=0, plan_id=row["plan_id"], item_id=item_id,
                              engine="exec")
            log(f"  ran: {cmd}")
            log(f"  {'ok' if ok else 'FAILED'}")
            if not ok:
                log("  " + evidence.strip().splitlines()[-1][:200])
                jarvis_db.set_item(conn, item_id, evidence=evidence[-4000:],
                                   notes=f"option {index} ({label}) failed to apply")
                return "failed"
            ran = f"ran: {cmd}"
        jarvis_db.set_item(conn, item_id, status="pending", options="",
                           evidence=None, notes=f"option {index} applied — {label}\n{ran}")
        log(f"  {item_id} reopened; the coder will retry against the new state")
        return "reopened"

    if kind == "respec":
        new_verify = clean_verify(action.get("verify") or row["verify"])
        jarvis_db.add_item(conn, {
            "id": item_id, "plan_id": row["plan_id"], "ordinal": row["ordinal"] or 0,
            "title": (action.get("title") or row["title"])[:200],
            "detail": action.get("detail") or row["detail"],
            "verify": new_verify, "workspace": workspace,
            "depends_on": row["depends_on"], "status": "pending", "owner": "coder",
            "notes": f"option {index} applied — {label}",
        })
        if new_verify != (row["verify"] or ""):
            # The verify changed, so the plan is no longer the thing that was reviewed.
            _rearm_review(conn, row["plan_id"], log)
        log(f"  {item_id} re-specified and reopened")
        return "respecified"

    jarvis_db.set_item(conn, item_id, options="",
                       notes=f"option {index} chosen: {label} — {opt.get('what', '')}")
    log("  no automatic action was defined — the choice is recorded for a human")
    return "recorded"


def _rearm_review(conn, plan_id: str, log=print) -> None:
    with conn:
        conn.execute("UPDATE plans SET consistent=NULL, reviewed_at=NULL, updated=?"
                     " WHERE id=?", (time.time(), plan_id))
    log(f"  review gate re-armed for {plan_id} — re-review before the coder runs")


def do_options(item_id: str, log=print) -> None:
    """Ask the architect for genuinely different ways to resolve an item.

    The owner's standing instruction: always offer options. Used when something is stuck
    or when a decision is genuinely open — a single prescribed path hides the trade-off
    from the person who has to live with it.
    """
    conn = jarvis_db.open_db()
    row = conn.execute("SELECT * FROM work_items WHERE id=?", (item_id,)).fetchone()
    if row is None:
        raise SystemExit(f"no such item: {item_id}")
    workspace = row["workspace"] or os.getcwd()

    prompt = REPAIR_PROMPT.format(
        attempts=0, item_id=item_id, title=row["title"], detail=row["detail"] or "",
        error=(row["evidence"] or "this item has not been attempted yet")[-2000:],
        history=(row["notes"] or "(nothing recorded)")[:1500],
        listing=list_files(workspace))
    log(f"  asking the architect about {item_id}…")
    reply, ms = llm.timed_chat("director", prompt, max_tokens=REPAIR_TOKENS, temperature=0.4)
    jarvis_db.log_run(conn, role="director", prompt=prompt, output=reply, ok=bool(reply),
                      ms=ms, plan_id=row["plan_id"], item_id=item_id, engine=":8081")

    data = llm.extract_json(reply) or {}
    block = _options_block(data)
    log("")
    log(block or "the architect offered no options — itself a defect worth flagging")
    if data.get("verdict"):
        log(f"\nverdict: {data['verdict']}")
    raw_opts = [o for o in (data.get("options") or []) if isinstance(o, dict)]
    if raw_opts:
        notes = "\n\n".join(x for x in (row["notes"], block) if x)
        jarvis_db.set_item(conn, item_id, notes=notes[-4000:],
                           options=json.dumps(raw_opts))
        log(f"\n  choose one:  python3 devteam.py choose {item_id} <n>")


def _options_block(data: dict) -> str:
    """Render the architect's options as a compact block for the item notes."""
    opts = [o for o in (data.get("options") or []) if isinstance(o, dict)]
    if not opts:
        return ""
    lines = ["Options offered by the architect:"]
    for o in opts:
        mark = "  ← recommended" if o.get("recommended") else ""
        lines.append(f"- {o.get('label', 'option')}{mark}: {o.get('what', '')}")
        lines.append(f"    cost: {o.get('cost', '?')}  |  risk: {o.get('risk', '?')}")
    if data.get("recommendation"):
        lines.append(f"Recommendation: {data['recommendation']}")
    return "\n".join(lines)


def reclaim_interrupted(conn, plan_id: str, log=None) -> int:
    """A previous run may have died mid-item. Reclaim rather than strand it.

    Safe without a liveness check because every entry point that reaches this holds
    the devteam lock (see _lock): an in-progress item seen while we hold the lock is
    by definition stale, not live work.
    """
    rows = jarvis_db.list_items(conn, plan_id, status="in-progress")
    for row in rows:
        jarvis_db.set_item(conn, row["id"], status="pending",
                           notes="reclaimed after an interrupted run")
    if rows and log:
        log(f"reclaimed {len(rows)} interrupted item(s)")
    return len(rows)


def do_run(plan_id: str, max_items: int, force: bool = False, log=print) -> dict:
    conn = jarvis_db.open_db()
    plan = jarvis_db.get_plan(conn, plan_id)
    if plan is None:
        raise SystemExit(f"no such plan: {plan_id}")

    # Two gates before the coder starts, in order. Both are deliberate.
    if plan["reviewed_at"] is None:
        raise SystemExit(
            f"plan {plan_id} has not been sanity-checked by the architect, so the coder "
            f"does not start.\n"
            f"  run:  python3 devteam.py review {plan_id}")
    if not plan["consistent"] and not force:
        raise SystemExit(
            f"plan {plan_id} was reviewed and found INCONSISTENT — see the issues with:\n"
            f"      python3 devteam.py review {plan_id}\n"
            f"  Coding it as written would build two incompatible halves. Fix the plan, "
            f"or re-run with --force to proceed anyway.")
    if not plan["approved"]:
        raise SystemExit(f"plan {plan_id} is not approved — the gate is deliberate.\n"
                         f"  python3 devteam.py approve {plan_id}")

    tally = {"verified": 0, "blocked": 0, "respec": 0, "reclaimed": 0}
    tally["reclaimed"] = reclaim_interrupted(conn, plan_id, log=log)

    # An item the director had to re-spec is set aside for this run rather than
    # retried immediately: the fresh specification deserves a fresh pass, and
    # re-running it here would starve every other item behind it.
    deferred: set[str] = set()

    for _ in range(max_items):
        item = None
        for cand in jarvis_db.list_items(conn, plan_id, status="pending"):
            if cand["id"] in deferred:
                continue
            if jarvis_db._deps_satisfied(conn, cand):
                item = cand
                break
        if item is None:
            break
        try:
            outcome = run_item(conn, item, log=log)
        except Exception as exc:  # noqa: BLE001
            # An unexpected failure here used to kill the whole tick and leave the
            # item in-progress — a silent, permanent stall. Surface it instead.
            tb = traceback.format_exc()
            log(f"  !! {item['id']} hit an internal error — marking blocked, not stranding")
            log(tb)
            jarvis_db.set_item(conn, item["id"], status="blocked", evidence=tb[-2000:],
                               notes=f"internal error in the runner: {type(exc).__name__}: {exc}")
            outcome = "blocked"
        tally[outcome] = tally.get(outcome, 0) + 1
        if outcome == "respec":
            deferred.add(item["id"])
    if deferred:
        log(f"\n  set aside for a fresh pass: {', '.join(sorted(deferred))}")
    return tally


# ------------------------------------------------------------------ autopilot

LOCK_PATH = os.getenv("DEVTEAM_LOCK", "/tmp/jarvis-devteam.lock")


class AlreadyRunning(RuntimeError):
    """Another run holds the lock. Not an error condition — a scheduler tick colliding."""


@contextlib.contextmanager
def _lock():
    """One workflow at a time.

    The coder lane is a single 32B model: two runs driving it at once interleave their
    outputs and each sees half of the other's work. A timer and a human both being able
    to start a run makes that a certainty rather than a risk, so the lock is not optional.
    """
    fh = open(LOCK_PATH, "w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        raise AlreadyRunning("another devteam run is already working") from None
    try:
        fh.write(f"{os.getpid()}\n")
        fh.flush()
        yield
    finally:
        fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()


def _ready(conn, plan_id: str) -> bool:
    """Is there an item on this plan that could be started right now?"""
    items = jarvis_db.list_items(conn, plan_id)
    if not items or any(i["status"] == "in-progress" for i in items):
        return False
    return jarvis_db.next_ready_item(conn, plan_id) is not None


def do_autopilot(max_items: int = 4, log=print) -> dict:
    """Work whatever is ready, across every plan that is fit to run.

    This is the difference between an offline team and a tool that works when someone
    remembers to push it: the scheduler calls this on a timer, so queued work actually
    moves. Bounded per call so a frequent tick stays cheap.
    """
    conn = jarvis_db.open_db()
    plans = list(conn.execute(
        "SELECT id, title FROM plans WHERE approved=1 AND consistent=1"
        " AND reviewed_at IS NOT NULL ORDER BY created"))
    if not plans:
        log("nothing to do — no plan is reviewed, approved and consistent")
        return {}

    out: dict = {}
    budget = max_items
    for p in plans:
        if budget <= 0:
            break
        # Reclaim BEFORE the readiness test. _ready() reports a plan with an
        # interrupted item as not-ready, and do_run's own reclaim is downstream of
        # that test — so without this the plan is skipped forever, silently.
        reclaim_interrupted(conn, p["id"], log=log)
        if not _ready(conn, p["id"]):
            continue
        log(f"\n### {p['id']} — {p['title']}")
        tally = do_run(p["id"], budget, log=log)
        out[p["id"]] = tally
        budget -= sum(v for k, v in tally.items()
                      if k in ("verified", "blocked", "auto", "respec"))
    if not out:
        log("nothing ready to start")
    return out


# ------------------------------------------------------------------ reporting

def do_status(plan_id: str | None) -> None:
    conn = jarvis_db.open_db()
    plans = ([jarvis_db.get_plan(conn, plan_id)] if plan_id
             else list(conn.execute("SELECT * FROM plans ORDER BY created DESC LIMIT 10")))
    for plan in plans:
        if plan is None:
            print(f"no such plan: {plan_id}")
            continue
        items = jarvis_db.list_items(conn, plan["id"])
        done = sum(1 for i in items if i["status"] in ("verified", "complete"))
        flag = "APPROVED" if plan["approved"] else "awaiting approval"
        print(f"\n{plan['id']}  [{flag}]  {plan['title']}")
        print(f"  {done}/{len(items)} verified   —  {plan['brief'][:100]}")
        for i in items:
            mark = {"verified": "OK ", "blocked": "XX ", "in-progress": ".. "}.get(i["status"], "   ")
            print(f"   {mark}{i['id']:<12} {i['status']:<12} {i['title'][:64]}")
            if i["status"] == "blocked" and i["evidence"]:
                print(f"       └─ {i['evidence'].strip().splitlines()[-1][:110]}")


FALSIFY_PROMPT = """You are the DIRECTOR, and this pass is adversarial: try to BREAK this plan's verify commands.

A verify command is the only thing that decides an item is finished. A weak one is worse
than no check at all, because the item is then marked `verified` while the work is still
missing — and everything built on top of it inherits the gap.

For EACH item ask: could this command exit 0 WITHOUT the work having been done? Ways that
happens:

- it only imports, prints, or lists a file, proving nothing about behaviour
- it asserts a name exists, so any stub passes
- it reads no output and writes no fixture: a no-op function satisfies it
- it tests something the standard library provides rather than the item's own code
- it would already exit 0 on the workspace as it stands, before the item is written

Report ONLY the commands that could pass without the work, and give a stronger command
that fails when the work is missing. Do not invent problems to seem useful — if a command
genuinely proves the behaviour it claims, say so and move on.

YOUR REPLACEMENT MUST BE PASSABLE BY CORRECT WORK. A check that fails on a correct
implementation is worse than the weak one it replaces: the item can then never verify,
and everything behind it is blocked. Three ways this goes wrong, all of which have
happened on this project:

- Do NOT search raw PDF bytes for a string. PDF content is usually compressed, so text
  a library just wrote is not literally present in the file and the check fails on
  correct work. Assert on the value the item's own function RETURNS, or re-open the
  output with the library that wrote it (pypdf / fitz / pdfplumber) and read it through
  that library.
- Do NOT assert on a name you were not given. If the specification does not state a
  field name, an option name or a key, you do not know it and must not invent one.
- Every path a command READS must already exist or be created by another item in this
  plan. Nothing in this sandbox can be downloaded or generated out of thin air. If the
  fixture a command needs is created by no item, say that plainly in `why` and leave
  `stronger_verify` empty — the plan needs a new item, not a cleverer assertion.

THE PLAN:
{plan}

Reply with ONE JSON object and no markdown fence:
{{
  "findings": [
    {{"id": "<item id>",
      "why": "<how this command could pass with the work missing>",
      "stronger_verify": "<a replacement command that FAILS when the work is missing>"}}
  ],
  "sound": ["<ids whose verify is genuinely able to fail>"]
}}"""


TRIAGE_PROMPT = """You are the DIRECTOR reading your own team's board.

Below is the state of the work and the recent failed calls. Say what is actually wrong and
what should be done about it. Be concrete and brief — an operator reads this to decide
whether to intervene.

Prefer a diagnosis over a description: "DOCS:JV-001 cannot pass because the fixture it
reads is created by no item, so it should be re-specced to build its own" beats
"DOCS:JV-001 is in progress". If the board is healthy, say so plainly.

THE BOARD:
{board}

RECENT FAILED RUNS (most recent first):
{runs}

Reply with ONE JSON object and no markdown fence:
{{"summary": "<two sentences: what is stuck, and why>",
  "healthy": true or false,
  "actions": [
    {{"kind": "respec" or "block" or "none",
      "id": "<item id>",
      "why": "<one sentence>",
      "detail": "<if respec: the corrected FULL specification; else empty>",
      "verify": "<if respec: the corrected verify command; else empty>"}}
  ]}}"""


def _board_text(conn, plan_id=None, limit=5) -> str:
    plans = ([jarvis_db.get_plan(conn, plan_id)] if plan_id
             else list(conn.execute("SELECT * FROM plans ORDER BY created DESC LIMIT ?",
                                    (limit,))))
    lines = []
    for p in plans:
        if p is None:
            continue
        lines.append(f"PLAN {p['id']} — {p['title']}   approved={p['approved']} "
                     f"reviewed={bool(p['reviewed_at'])}")
        for i in jarvis_db.list_items(conn, p["id"]):
            lines.append(f"  {i['id']:<14} {i['status']:<12} {i['title'][:64]}")
            note = [l for l in (i["notes"] or "").splitlines() if l.strip()]
            if note:
                lines.append(f"      note: {note[-1][:160]}")
    return "\n".join(lines) or "(no plans)"


def do_falsify(plan_id: str, fix: bool = False, log=print) -> int:
    """Could any verify pass with the work left undone? Returns how many could.

    The consistency review asks whether the items contradict each other. This asks the
    other question — whether the checks would notice if the code were never written —
    and it is the failure this team actually ships: five of DOCS's seven verifies could
    not fail, and every one of them reached `approved`.
    """
    conn = jarvis_db.open_db()
    if jarvis_db.get_plan(conn, plan_id) is None:
        raise SystemExit(f"no such plan: {plan_id}")
    block = "\n\n".join(
        f"{i['id']}: {i['title']}\n  verify: {i['verify'] or '(none)'}\n"
        f"  detail: {(i['detail'] or '')[:400]}"
        for i in jarvis_db.list_items(conn, plan_id))
    prompt = FALSIFY_PROMPT.format(plan=block)
    try:
        reply, ms = llm.timed_chat("director", prompt, max_tokens=REPAIR_TOKENS,
                                   temperature=0.3)
    except Exception as exc:  # noqa: BLE001
        log(f"  director unavailable: {exc}")
        return 0
    jarvis_db.log_run(conn, role="director", prompt=prompt, output=reply, ok=bool(reply),
                      ms=ms, plan_id=plan_id, engine=":8081")

    findings = [f for f in ((llm.extract_json(reply) or {}).get("findings") or [])
                if isinstance(f, dict) and f.get("id")]
    if not findings:
        log("  every verify can fail on its own — nothing to strengthen")
        return 0

    log(f"  {len(findings)} verify command(s) could pass with the work missing:")
    replaced = 0
    for f in findings:
        log(f"    {f['id']}: {str(f.get('why'))[:150]}")
        stronger = clean_verify(str(f.get("stronger_verify") or ""))
        if not stronger:
            continue
        log(f"       stronger: {stronger[:150]}")
        if fix and conn.execute("SELECT 1 FROM work_items WHERE id=?",
                                (f["id"],)).fetchone():
            with conn:
                conn.execute("UPDATE work_items SET verify=?, updated=? WHERE id=?",
                             (stronger, time.time(), f["id"]))
            replaced += 1
    if fix and replaced:
        with conn:
            conn.execute("UPDATE plans SET consistent=NULL, reviewed_at=NULL, updated=?"
                         " WHERE id=?", (time.time(), plan_id))
        log(f"  replaced {replaced} verify command(s); the plan is no longer the one that "
            f"was reviewed:\n    python3 devteam.py review {plan_id}")
    return len(findings)


def do_triage(plan_id: str | None = None, log=print) -> None:
    """Ask the director to read the board and say what is stuck, and why.

    This is the local reasoner doing what a cloud agent would otherwise be paid to do on
    every overwatch tick: read the state, find the stall, propose the fix. R1 is the
    model the owner wants used to capacity, and this is a job it is good at.
    """
    conn = jarvis_db.open_db()
    board = _board_text(conn, plan_id)
    rows = list(conn.execute(
        "SELECT ts, role, item_id, output FROM runs WHERE ok=0"
        " ORDER BY CAST(ts AS REAL) DESC LIMIT 8"))
    runs = "\n".join(
        f"  {time.strftime('%m-%d %H:%M', time.localtime(float(r['ts'])))} "
        f"{r['role']} {r['item_id'] or '-'}: {str(r['output'] or '')[:300]}"
        for r in rows) or "  (no failures recorded)"

    prompt = TRIAGE_PROMPT.format(board=board, runs=runs)
    try:
        reply, ms = llm.timed_chat("director", prompt, max_tokens=REPAIR_TOKENS,
                                   temperature=0.3)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"director unavailable: {exc}") from None
    jarvis_db.log_run(conn, role="director", prompt=prompt, output=reply, ok=bool(reply),
                      ms=ms, plan_id=plan_id, engine=":8081")

    data = llm.extract_json(reply) or {}
    print(f"\n{data.get('summary') or reply[:400]}\n")
    for a in [a for a in (data.get("actions") or []) if isinstance(a, dict)]:
        print(f"  [{a.get('kind', 'none')}] {a.get('id', '-')}: {a.get('why', '')}")
    if not data.get("healthy"):
        print("\n  apply a respec by hand:  "
              "python3 devteam.py respec <id> --detail '...' --verify '...'")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="director turns a brief into work items")
    p.add_argument("brief")
    p.add_argument("--workspace", default="/var/home/admin/jarvis")
    p.add_argument("--id", default=f"PLAN-{time.strftime('%Y%m%d-%H%M%S')}")
    p.add_argument("--title")

    a = sub.add_parser("approve", help="approve a plan for execution")
    a.add_argument("plan_id")
    a.add_argument("--note")
    a.add_argument("--force", action="store_true",
                   help="approve even though the lint found hard failures")

    v = sub.add_parser("review",
                       help="architect sanity-checks the plan for contradictions")
    v.add_argument("plan_id")
    v.add_argument("--fix", action="store_true",
                   help="also repair routine contradictions (naming, layout, interfaces)")

    r = sub.add_parser("run", help="execute a reviewed, approved plan")
    r.add_argument("plan_id")
    r.add_argument("--max-items", type=int, default=8)
    r.add_argument("--force", action="store_true",
                   help="proceed even though the review found contradictions")

    s = sub.add_parser("status", help="show plans and their items")
    s.add_argument("plan_id", nargs="?")

    l = sub.add_parser("lint", help="check that a plan's verify commands can fail")
    l.add_argument("plan_id")

    doc = sub.add_parser("doctor", help="check the sandbox and the lanes")

    fa = sub.add_parser("falsify",
                        help="adversarial pass: could any verify pass without the work?")
    fa.add_argument("plan_id")
    fa.add_argument("--fix", action="store_true",
                    help="replace the weak commands the pass identifies")

    tr = sub.add_parser("triage", help="ask the director what is stuck, and why")
    tr.add_argument("plan_id", nargs="?")

    ap_ = sub.add_parser("autopilot",
                         help="work whatever is ready across all runnable plans (timer entry)")
    ap_.add_argument("--max-items", type=int, default=4)

    rv = sub.add_parser("reverify",
                        help="re-run every verified item's check (catches regressions)")
    rv.add_argument("plan_id")

    o = sub.add_parser("options", help="ask the architect for ways to resolve an item")
    o.add_argument("item_id")

    c = sub.add_parser("choose", help="apply one of the offered options")
    c.add_argument("item_id")
    c.add_argument("index", type=int, help="1-based, as shown")

    z = sub.add_parser("respec", help="rewrite one item (this is how a review fix is applied)")
    z.add_argument("item_id")
    z.add_argument("--detail")
    z.add_argument("--title")
    z.add_argument("--verify")
    z.add_argument("--status", choices=jarvis_db.STATUSES)
    z.add_argument("--undepend", action="store_true",
                   help="clear this item's dependencies (when the dependency was the mistake)")

    x = sub.add_parser("item", help="add one work item by hand")
    x.add_argument("item_id")
    x.add_argument("--plan", required=True)
    x.add_argument("--title", required=True)
    x.add_argument("--detail", default="")
    x.add_argument("--verify", default="")
    x.add_argument("--workspace", default="/var/home/admin/jarvis")
    x.add_argument("--depends-on", default="")
    x.add_argument("--ordinal", type=int, default=0)

    args = ap.parse_args()

    if args.cmd == "plan":
        do_plan(args.brief, args.workspace, args.id, args.title)
    elif args.cmd == "approve":
        # Approval is the last gate before the coder starts, so the checks that can be
        # made mechanically are made HERE. lint_plan reports verifies that cannot fail
        # and verifies that import a package the environment does not have; both mean
        # the item cannot do its job, and approving anyway just spends the team's time.
        print(f"checking {args.plan_id} before approving:")
        hard = lint_plan(args.plan_id, log=print)
        if hard and not args.force:
            raise SystemExit(
                f"\nrefusing to approve {args.plan_id}: {hard} hard failure(s) above.\n"
                f"  Fix them (re-plan the item, or respec it), or override with --force.")
        conn = jarvis_db.open_db()
        jarvis_db.approve_plan(conn, args.plan_id, args.note)
        print(f"approved {args.plan_id}")
    elif args.cmd == "review":
        raise SystemExit(0 if do_review(args.plan_id, fix=args.fix) else 1)
    elif args.cmd == "run":
        try:
            with _lock():
                tally = do_run(args.plan_id, args.max_items, force=args.force)
        except AlreadyRunning as exc:
            raise SystemExit(f"not started: {exc}") from None
        print(f"\n{tally}")
        do_status(args.plan_id)
    elif args.cmd == "autopilot":
        try:
            with _lock():
                result = do_autopilot(args.max_items)
        except AlreadyRunning as exc:
            print(f"skipped: {exc}")
            return 0
        print(f"\n{result}")
    elif args.cmd == "status":
        do_status(args.plan_id)
    elif args.cmd == "lint":
        bad = lint_plan(args.plan_id, log=print)
        raise SystemExit(1 if bad else 0)
    elif args.cmd == "falsify":
        raise SystemExit(1 if do_falsify(args.plan_id, fix=args.fix) else 0)
    elif args.cmd == "triage":
        do_triage(args.plan_id)
    elif args.cmd == "doctor":
        st = sandbox.status()
        print("sandbox:")
        for k, v in st.items():
            print(f"  {k:15} {v}")
        if st["available"]:
            ok, out = run_verify(tempfile.mkdtemp(), 'python3 -c "print(1)"')
            print(f"  self-test      {'PASS' if ok else 'FAIL'}")
            print("    " + out.replace("\n", "\n    ")[:400])
        else:
            print("  self-test      skipped (no sandbox)")
        print("\nlanes:")
        for role, url in llm.LANES.items():
            try:
                import urllib.request
                with urllib.request.urlopen(f"{url}/v1/models", timeout=4) as r:
                    print(f"  {role:15} up ({r.status})")
            except Exception as exc:  # noqa: BLE001
                print(f"  {role:15} DOWN ({type(exc).__name__})")
    elif args.cmd == "reverify":
        print(do_reverify(args.plan_id))
    elif args.cmd == "options":
        do_options(args.item_id)
    elif args.cmd == "choose":
        print(apply_option(args.item_id, args.index))
    elif args.cmd == "respec":
        conn = jarvis_db.open_db()
        row = conn.execute("SELECT * FROM work_items WHERE id=?", (args.item_id,)).fetchone()
        if row is None:
            raise SystemExit(f"no such item: {args.item_id}")
        fields = {k: v for k, v in
                  (("title", args.title), ("detail", args.detail), ("verify", args.verify),
                   ("status", args.status)) if v is not None}
        if not fields and not args.undepend:
            raise SystemExit("nothing to change — pass --detail, --verify, --title or --status")
        sets = ", ".join(f"{k}=?" for k in fields)
        with conn:
            if fields:
                conn.execute(f"UPDATE work_items SET {sets}, updated=? WHERE id=?",
                             (*fields.values(), time.time(), args.item_id))
            if args.undepend:
                conn.execute("UPDATE work_items SET depends_on='', updated=? WHERE id=?",
                             (time.time(), args.item_id))
        # A changed verify invalidates the old review: the plan is no longer the thing
        # that was checked, so the gate must be satisfied again before coding resumes.
        if "verify" in fields or "detail" in fields or "title" in fields:
            with conn:
                conn.execute("UPDATE plans SET consistent=NULL, reviewed_at=NULL, updated=?"
                             " WHERE id=?", (time.time(), row["plan_id"]))
            print(f"  re-review required: python3 devteam.py review {row['plan_id']}")
        print(f"updated {args.item_id}: {', '.join(fields) or 'dependencies cleared'}")
    elif args.cmd == "item":
        conn = jarvis_db.open_db()
        jarvis_db.add_item(conn, {
            "id": args.item_id, "plan_id": args.plan, "title": args.title,
            "detail": args.detail, "verify": args.verify, "workspace": args.workspace,
            "depends_on": [d for d in args.depends_on.split(",") if d],
            "ordinal": args.ordinal, "status": "pending", "owner": "coder"})
        print(f"added {args.item_id} to {args.plan}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
