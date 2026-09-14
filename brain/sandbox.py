"""Run a verify command somewhere it cannot hurt anything.

Why this exists: until now every verify command ran **on the host**, as the same user
that owns the workspace and the database. A model-authored command — or a test that
happens to write outside its directory — had the run of the box. That is a poor place to
put "the only thing standing between a suggestion and a merge".

So a verify runs in a throwaway container instead:

    podman run --rm --network=none --cap-drop=ALL ... -v <workspace>:/work -w /work

Exit 0 inside the container is the same signal as before; what changes is what it took to
get it, and what it could reach while trying.

Deliberate choices:

* **Workspace is mounted read-WRITE.** Read-only was tempting and wrong: `python3 -m
  py_compile` writes `__pycache__`, and a check that fails for a reason unrelated to the
  code teaches the repair loop to chase ghosts. The container is what provides isolation
  here, not the mount — it has no network, no capabilities, no host filesystem, and it is
  destroyed on exit.
* **Rootless podman.** Works as `admin` on cerebro with no sudo, so the worker never needs
  privileges to run a test. A sandbox that requires root to enter is not much of a sandbox.
* **Local image only.** `python:3.12-slim` is already present, so verification never needs
  the network — which matters because the container has none.
* **Failing open is allowed, failing silently is not.** If podman is unavailable the caller
  may fall back to host execution, but the result carries `isolated=False` so the evidence
  says plainly that this check ran unprotected.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass

# The verify image is built locally and carries the libraries the plans need, because
# the sandbox runs with --network=none: nothing can be installed while a check runs, so
# anything a verify imports must already be baked in. See sandbox/Containerfile.verify.
IMAGE = os.getenv("DEVTEAM_SANDBOX_IMAGE", "localhost/jarvis-verify:latest")

# required -> no sandbox, no verification
# preferred -> use it when available, say so in the evidence when not  (default)
# off       -> run on the host, as before
MODE = os.getenv("DEVTEAM_SANDBOX", "preferred").strip().lower()

DEFAULT_TIMEOUT = int(os.getenv("DEVTEAM_VERIFY_TIMEOUT", "300"))
MEMORY_LIMIT = os.getenv("DEVTEAM_SANDBOX_MEMORY", "1g")
PIDS_LIMIT = os.getenv("DEVTEAM_SANDBOX_PIDS", "256")

_available: bool | None = None


@dataclass
class Result:
    ok: bool
    output: str
    isolated: bool          # did it actually run in a container?
    note: str = ""          # why not, when it did not

    @property
    def command_line(self) -> str:
        return ""


def available() -> bool:
    """Is a usable podman present? Probed once, then remembered."""
    global _available
    if _available is None:
        _available = bool(shutil.which("podman")) and _image_present()
    return _available


def _image_present() -> bool:
    try:
        p = subprocess.run(["podman", "image", "exists", IMAGE],
                           capture_output=True, timeout=30)
        return p.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def build_argv(workspace: str, command: str, image: str | None = None) -> list[str]:
    """The container invocation. Separated out so it can be asserted against in tests
    without needing podman, or a container, or a working workspace."""
    return [
        "podman", "run", "--rm",
        "--network=none",                     # no exfiltration, no surprise installs
        "--cap-drop=ALL",                     # nothing to escalate with
        "--security-opt", "no-new-privileges",
        "--security-opt", "label=disable",    # Fedora SELinux: avoid relabelling the host
        "--pids-limit", PIDS_LIMIT,           # no fork bombs
        "--memory", MEMORY_LIMIT,
        "--tmpfs", "/tmp:rw,size=64m",        # somewhere for temp files to go
        "-e", "PYTHONDONTWRITEBYTECODE=1",
        "-e", "PYTHONUNBUFFERED=1",
        "-v", f"{workspace}:/work",
        "-w", "/work",
        image or IMAGE,
        "/bin/sh", "-c", command,
    ]


def run(workspace: str, command: str, *, timeout: int | None = None,
        image: str | None = None) -> Result:
    """Run `command` in the workspace, inside a container when we can.

    The caller decides what to do about `isolated=False`; this function only reports it
    honestly rather than pretending the isolation happened.
    """
    timeout = timeout or DEFAULT_TIMEOUT
    if not command or not command.strip():
        return Result(False, "no verify command — refusing to call this done", False,
                      "empty command")

    if MODE == "off":
        return _host(workspace, command, timeout, "sandbox disabled by DEVTEAM_SANDBOX=off")
    if not available():
        reason = ("podman unavailable" if not shutil.which("podman")
                  else f"image {IMAGE} not present locally")
        if MODE == "required":
            return Result(False,
                          f"cannot verify: sandbox is required but {reason}. "
                          f"Pull the image or set DEVTEAM_SANDBOX=preferred.",
                          False, reason)
        return _host(workspace, command, timeout, reason)

    argv = build_argv(workspace, command, image)
    t0 = time.time()
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return Result(False,
                      f"verify timed out after {timeout}s in the sandbox\n"
                      f"COMMAND: {command}",
                      True, "")
    except (OSError, subprocess.SubprocessError) as exc:
        if MODE == "required":
            return Result(False, f"sandbox failed to start: {exc}", False, str(exc))
        return _host(workspace, command, timeout, f"sandbox failed to start: {exc}")

    elapsed = time.time() - t0
    out = ((p.stdout or "") + (p.stderr or "")).strip()
    evidence = (f"COMMAND: {command}\n"
                f"SANDBOX: {IMAGE} (network=none, cap-drop=ALL, --rm)\n"
                f"EXIT: {p.returncode}   [{elapsed:.1f}s]\n"
                f"OUTPUT:\n{out[-2500:]}")
    return Result(p.returncode == 0, evidence, True)


def _host(workspace: str, command: str, timeout: int, why: str) -> Result:
    """Fall back to running on the host — the old behaviour, clearly labelled."""
    try:
        p = subprocess.run(command, shell=True, cwd=workspace, text=True,
                           capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return Result(False, f"verify timed out after {timeout}s on the host\n"
                             f"COMMAND: {command}", False, why)
    out = ((p.stdout or "") + (p.stderr or "")).strip()
    evidence = (f"COMMAND: {command}\n"
                f"SANDBOX: NOT ISOLATED — {why}\n"
                f"EXIT: {p.returncode}\n"
                f"OUTPUT:\n{out[-2500:]}")
    return Result(p.returncode == 0, evidence, False, why)


def status() -> dict:
    """For the board and for `devteam.py doctor`."""
    return {
        "mode": MODE,
        "podman": bool(shutil.which("podman")),
        "image": IMAGE,
        "image_present": _image_present(),
        "available": available(),
    }
