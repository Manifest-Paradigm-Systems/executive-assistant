"""Talking to the local model lanes, and getting structured answers back out.

The lanes are llama.cpp servers exposing an OpenAI shape. Two things about them
shape this module:

* **They are chatty about JSON.** A local model will wrap the object in a
  markdown fence, prefix it with a sentence of preamble, or emit a trailing NUL.
  Insisting on strict output means dropping real work on the floor, so the parse
  is forgiving and the *validation* is where strictness lives.
* **They are slow.** A 32B model on this box can take minutes on a hard prompt.
  Timeouts are generous by default and callers are expected to run off the reply
  path — nothing here belongs in a request handler.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request

LANES = {
    "director": os.getenv("JARVIS_DIRECTOR_URL", "http://127.0.0.1:8081"),
    "coder": os.getenv("JARVIS_CODER_URL", "http://127.0.0.1:8082"),
    "actor": os.getenv("JARVIS_ACTOR_URL", "http://127.0.0.1:8083"),
    "vision": os.getenv("JARVIS_VISION_URL", "http://127.0.0.1:8084"),
}
MODELS = {"director": "director", "coder": "coder", "actor": "actor"}

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN = re.compile(r"<think>.*$", re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> str:
    """Drop R1's reasoning. The director lane emits it inline; it is not
    answer, it is homework, and it poisons anything downstream that parses."""
    text = _THINK.sub("", text or "")
    return _THINK_OPEN.sub("", text).strip()


def chat(role: str, prompt: str, *, system: str | None = None, max_tokens: int = 2000,
         temperature: float | None = None, timeout: int = 900) -> str:
    """One completion from a lane. Returns the text (think stripped).

    Raises on transport failure — callers decide whether that is a retry or a
    blocked work item, and swallowing it here would hide real outages.
    """
    url = LANES[role]
    model = MODELS.get(role, role)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens,
               "temperature": temperature if temperature is not None else
               (0.6 if role == "director" else 0.2)}
    req = urllib.request.Request(f"{url}/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read())
    msg = body["choices"][0]["message"]
    return strip_think((msg.get("content") or "") + "\n" + (msg.get("reasoning_content") or ""))


def timed_chat(role: str, prompt: str, **kw) -> tuple[str, int]:
    t0 = time.time()
    out = chat(role, prompt, **kw)
    return out, int((time.time() - t0) * 1000)


# ------------------------------------------------------------------ json

def extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of whatever the lane actually returned."""
    if not text:
        return None
    text = text.strip().replace("\x00", "")
    try:
        got = json.loads(text)
        return got if isinstance(got, dict) else None
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        try:
            got = json.loads(fenced.group(1).strip())
            return got if isinstance(got, dict) else None
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        got = json.loads(text[start:i + 1])
                        if isinstance(got, dict):
                            return got
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


# ------------------------------------------------------------------ the cloud lane
# The local lanes are the team. This one exists for the rare call a 32B cannot be
# trusted with and cannot be argued out of: deciding whether a review rejection is real,
# where a false "no" stops a plan and a false "yes" ships a contradiction.
#
# Volume is tiny — one call per rejection — and the price is nothing. Measured
# 2026-09-14: a test call cost $0.0000009; a full adjudication a fraction of a cent.
#
# The key is read from the same 0600 env file the rest of the brain uses, because the
# systemd unit does not export it and a developer running devteam.py by hand should get
# the same behaviour as the timer.
CLOUD_URL = os.getenv("JARVIS_CLOUD_URL", "https://openrouter.ai/api/v1")
CLOUD_MODEL = os.getenv("JARVIS_CLOUD_MODEL", "deepseek/deepseek-v4-flash")
_ENV_FILE = os.getenv("JARVIS_ENV_FILE",
                      os.path.expanduser("~/jarvis/brain/brain.env"))


def _key_from_env_file(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("OPENROUTER_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        pass
    return ""


CLOUD_KEY = (os.getenv("JARVIS_CLOUD_KEY") or os.getenv("OPENROUTER_API_KEY")
             or _key_from_env_file(_ENV_FILE))


def cloud_available() -> bool:
    return bool(CLOUD_KEY)


def cloud_chat(prompt: str, *, max_tokens: int = 2500, timeout: int = 180) -> str:
    """One completion from the frontier model.

    Raises on transport or auth failure rather than returning something empty: the
    caller has to decide whether that means "fall back to the local verdict" or "stop",
    and a revoked key must not present itself as a silent downgrade.
    """
    if not CLOUD_KEY:
        raise RuntimeError("no cloud key: set OPENROUTER_API_KEY or JARVIS_CLOUD_KEY")
    payload = json.dumps({"model": CLOUD_MODEL,
                          "messages": [{"role": "user", "content": prompt}],
                          "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request(f"{CLOUD_URL}/chat/completions", data=payload,
                                 headers={"Authorization": f"Bearer {CLOUD_KEY}",
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read())
    if "error" in body:
        raise RuntimeError(f"cloud error: {str(body['error'])[:200]}")
    return strip_think(body["choices"][0]["message"].get("content") or "").strip()
