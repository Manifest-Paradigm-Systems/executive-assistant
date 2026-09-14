"""The editor — Aider in a container, doing the actual code changes.

This replaces the hand-rolled "ask the coder for whole files as JSON and write them"
step. That approach had a structural flaw we hit repeatedly: a model asked to emit a
complete file will emit *its idea* of that file, and if two work items name the same file
the second silently deletes the first's work.

Aider edits the way a person does — it reads the file that is there and applies a targeted
change — and it does so through a **text patch format it parses itself**, which is exactly
why it works against our lanes when Claude Code and AG2 do not. Those two need an OpenAI
`tool_calls` field that llama.cpp never emits; Aider needs only text, like our own `@@`
protocol.

The container asymmetry is deliberate:

    editor  --network=host   it must reach the model lane, or it can do nothing
    verify  --network=none   it must reach nothing, and it holds the verdict

Only one of those needs a network. It is not the one that decides.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field

IMAGE = os.getenv("DEVTEAM_EDITOR_IMAGE", "localhost/jarvis-editor:latest")
LANE_URL = os.getenv("DEVTEAM_EDITOR_LANE", "http://127.0.0.1:8082/v1")
MODEL = os.getenv("DEVTEAM_EDITOR_MODEL", "openai/coder")
DEFAULT_TIMEOUT = int(os.getenv("DEVTEAM_EDITOR_TIMEOUT", "900"))

_ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")

_available: bool | None = None


@dataclass
class Result:
    ok: bool
    transcript: str
    changed: list[str] = field(default_factory=list)
    error: str = ""
    seconds: float = 0.0


def available() -> bool:
    global _available
    if _available is None:
        try:
            p = subprocess.run(["podman", "image", "exists", IMAGE],
                               capture_output=True, timeout=30)
            _available = p.returncode == 0
        except (OSError, subprocess.SubprocessError):
            _available = False
    return _available


def strip_ansi(text: str) -> str:
    """Aider draws a TUI. Captured output is full of cursor moves and colour codes, and
    a transcript nobody can read is not a transcript."""
    return _ANSI.sub("", text or "").replace("\r", "")


def _public_names(path: str) -> frozenset[str]:
    """Top-level defs and classes — the same surface check_no_clobber guards."""
    import ast
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            tree = ast.parse(fh.read())
    except (OSError, SyntaxError, ValueError):
        return frozenset()
    return frozenset(n.name for n in tree.body
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)))


def snapshot(workspace: str) -> dict[str, tuple[float, frozenset[str]]]:
    """File -> (mtime, public names).

    The names matter as much as the times: Aider edits files *directly*, so it never
    passes through devteam's check_no_clobber. Carrying the function list here is what
    keeps that protection alive for the new executor instead of quietly losing it when
    the writer changed.
    """
    seen: dict[str, tuple[float, frozenset[str]]] = {}
    for dirpath, dirnames, filenames in os.walk(workspace):
        dirnames[:] = [d for d in dirnames
                       if d not in ("__pycache__", ".git") and not d.startswith(".")]
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            if fn.startswith(".") or ".bak-" in fn:
                continue
            try:
                names = _public_names(full) if fn.endswith(".py") else frozenset()
                seen[os.path.relpath(full, workspace)] = (os.path.getmtime(full), names)
            except OSError:
                pass
    return seen


def changed_between(before: dict, after: dict) -> list[str]:
    out = []
    for path, value in after.items():
        if path not in before or before[path][0] != value[0]:
            out.append(path)
    for path in before:
        if path not in after:
            out.append(f"{path} (REMOVED)")
    return sorted(out)


def lost_functions(before: dict, after: dict) -> list[str]:
    """Did the edit delete another item's work?

    A file that loses a top-level function it used to have is the exact signature of the
    clobbering bug that cost four verified items. With whole-file writes it happened
    silently; with a targeted editor it should not happen at all, which is precisely why
    it is worth checking rather than assuming.
    """
    losses = []
    for path, (_, had) in before.items():
        if not had or path not in after:
            continue
        now = after[path][1]
        gone = had - now
        if gone:
            losses.append(f"{path} lost {', '.join(sorted(gone))}")
    return losses


_FILE_RE = re.compile(r"[\w./-]+\.(?:py|md|json|sh|txt)")


def _existing_under(workspace: str, basename: str) -> list[str]:
    hits = []
    for dirpath, dirnames, filenames in os.walk(workspace):
        dirnames[:] = [d for d in dirnames
                       if d != "__pycache__" and not d.startswith(".")]
        if basename in filenames:
            hits.append(os.path.relpath(os.path.join(dirpath, basename), workspace))
    return sorted(hits)


def _named_dirs(instruction: str, workspace: str) -> list[str]:
    """Directory names the specification talks about, whether or not they exist yet.

    `read`, `resolve`, `cli` and `tests` in "subdirectories for read, resolve, cli and
    tests" are as much a part of the target as the filename is.
    """
    found = []
    for word in re.findall(r"\b[\w-]{2,}\b", instruction or ""):
        if word in found:
            continue
        if os.path.isdir(os.path.join(workspace, word)):
            found.append(word)
    for word in re.findall(r"\b([\w-]{2,})/(?![\w/])", instruction or ""):
        if word not in found and word not in ("the", "a"):
            found.insert(0, word)
    return found


def target_files(instruction: str, workspace: str, limit: int = 8) -> list[str]:
    """Which files should the editor be allowed to change?

    Aider can only touch files it is given, so these come from the item's own
    specification — and getting them wrong is expensive rather than merely useless: a
    bare `__init__.py` taken literally creates a stray file at the workspace root, which
    then shows up in every later listing and misleads every later item.

    So a bare filename is resolved before it is trusted:

      1. known by its full path        -> use it
      2. exists in exactly one place   -> use that place
      3. exists in several places      -> prefer one under a directory the spec names
      4. does not exist yet            -> place it under a directory the spec names
      5. otherwise                     -> drop it and say so, rather than guess
    """
    out: list[str] = []
    dropped: list[str] = []
    dirs = _named_dirs(instruction, workspace)

    for candidate in dict.fromkeys(_FILE_RE.findall(instruction or "")):
        candidate = candidate.strip("./")
        if not candidate:
            continue
        if "/" in candidate:
            out.append(candidate)
            continue

        hits = _existing_under(workspace, candidate)
        if len(hits) == 1:
            out.append(hits[0])
            continue
        if len(hits) > 1:
            named = [h for h in hits if any(d in h.split(os.sep) for d in dirs)]
            if len(named) == 1:
                out.append(named[0])
                continue
            dropped.append(f"{candidate} (ambiguous: {', '.join(hits[:4])})")
            continue

        # Does not exist yet: anchor it to a directory the spec mentions.
        if dirs:
            out.append(f"{dirs[0]}/{candidate}")
        else:
            dropped.append(f"{candidate} (no directory to anchor it to)")

    if dropped:
        # Surfaced through the caller's log rather than swallowed: an item whose targets
        # cannot be determined needs a better specification, not a confident guess.
        out.append(f"__UNRESOLVED__: {'; '.join(dropped)}")

    return out[:limit]


def run(workspace: str, instruction: str, *, files: list[str] | None = None,
        timeout: int | None = None, lane_url: str | None = None,
        model: str | None = None) -> Result:
    """Let the editor make the change. Returns what it said and what it touched."""
    timeout = timeout or DEFAULT_TIMEOUT
    files = files if files is not None else target_files(instruction, workspace)
    unresolved = [f for f in files if f.startswith("__UNRESOLVED__")]
    files = [f for f in files if not f.startswith("__UNRESOLVED")]
    if unresolved and not files:
        # Every candidate was ambiguous. Guessing here creates files in the wrong place
        # and the mess compounds, so this goes back to the director as a bad
        # specification instead.
        return Result(False, "", error="cannot tell which file this item means: "
                                       + unresolved[0].split(": ", 1)[-1])
    if not files:
        return Result(False, "", error="no target files — the specification names no file "
                                       "for the editor to change")
    if not available():
        return Result(False, "", error=f"editor image {IMAGE} not present "
                                       f"(build it from ~/jarvis/editor/Containerfile)")

    before = snapshot(workspace)
    argv = [
        "podman", "run", "--rm",
        "--network=host",                     # must reach the model lane
        "--security-opt", "label=disable",
        "-v", f"{workspace}:/work",
        "-w", "/work",
        IMAGE,
        "--model", model or MODEL,
        "--openai-api-base", lane_url or LANE_URL,
        "--openai-api-key", "local-llama",
        "--no-git",                           # the worker owns version control, not aider
        "--no-auto-commits",
        "--no-check-update",
        "--no-show-model-warnings",
        "--no-stream",                        # a readable transcript, not a TUI redraw
        # Keep aider's bookkeeping out of the workspace: it otherwise drops
        # .aider.chat.history.md and .aider.input.history into the project, where they
        # show up in the coder's own file listing and confuse it about what exists.
        "--chat-history-file", "/tmp/aider-history.md",
        "--input-history-file", "/tmp/aider-input.txt",
        "--yes",
        "--message", instruction,
        *files,
    ]
    t0 = time.time()
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return Result(False, "", error=f"editor timed out after {timeout}s",
                      seconds=time.time() - t0)
    except (OSError, subprocess.SubprocessError) as exc:
        return Result(False, "", error=f"editor failed to start: {exc}")

    transcript = strip_ansi((p.stdout or "") + "\n" + (p.stderr or "")).strip()
    after = snapshot(workspace)
    changed = changed_between(before, after)

    losses = lost_functions(before, after)
    if losses:
        return Result(False, transcript[-6000:], changed=changed,
                      error="the edit DELETED code another item provides — "
                            + "; ".join(losses),
                      seconds=time.time() - t0)

    if p.returncode != 0 and not changed:
        return Result(False, transcript[-6000:],
                      error=f"editor exited {p.returncode} and changed nothing",
                      seconds=time.time() - t0)
    if not changed:
        return Result(False, transcript[-6000:],
                      error="the editor ran but changed no files",
                      seconds=time.time() - t0)
    return Result(True, transcript[-6000:], changed=changed,
                  seconds=time.time() - t0)


def status() -> dict:
    return {"image": IMAGE, "available": available(), "lane": LANE_URL, "model": MODEL,
            "timeout": DEFAULT_TIMEOUT}
