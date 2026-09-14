"""Contradictions that can be counted.

The review is asked to find places where a dozen items cannot all be true at once. It is
good at the ones needing judgement — one item calling a module `reader.py` while another
imports `read.py` — and reliably bad at the ones needing arithmetic.

On 2026-09-14 it reported that `generate_rent_receipt` was "specified to accept three
arguments, but the verify command only provides two", when the verify passed three; it
reported two items as conflicting because they had both edited one file, though both
were VERIFIED hours earlier and that is history, not conflict; and it once paired an
item with itself.

Every one of those is a count or a set membership. So they are done here, where the
answer is the same on every run, and the reviewer is handed the results as facts rather
than asked to derive them.

Deliberately narrow. A deterministic check that is wrong is worse than a model that is
sometimes wrong, because it is wrong every single time.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

DONE = ("verified", "complete")

# A signature as a specification writes it: `fill_form(a, b, out="x") -> str`.
_SIGNATURE = re.compile(r"\b([A-Za-z_]\w*)\s*\(([^()]*)\)\s*->")
# A call anywhere in a verify command.
_CALL = re.compile(r"\b([A-Za-z_]\w*)\s*\(([^()]*)\)")
# An item claiming to bring a file into existence.
_CREATES = re.compile(
    r"\b[Cc]reate(?:s)?\s+(?:the\s+|a\s+|its\s+)?(?:new\s+)?"
    r"(?:file\s+|module\s+|test\s+file\s+)?[`\"']?([\w][\w./-]*\.py)\b")

_NOT_A_FUNCTION = {"if", "for", "while", "return", "print", "assert", "def", "and", "or",
                   "not", "in", "with", "lambda", "else", "elif"}


@dataclass
class Finding:
    kind: str
    items: list
    problem: str
    fix: str = ""

    def __str__(self) -> str:
        return f"[{', '.join(self.items)}] {self.problem}"


def as_dict(item) -> dict:
    """A plain dict, whatever the database handed us.

    sqlite3.Row answers item["key"] and knows .keys(), but has no .get() — so code that
    works on dicts raises AttributeError the moment it meets a real row from
    jarvis_db.list_items(). Both callers exist, so normalise at the door.
    """
    if isinstance(item, dict):
        return item
    try:
        return {k: item[k] for k in item.keys()}
    except AttributeError:
        return dict(item)


def split_args(text: str) -> list:
    """Top-level comma split — commas inside brackets or quotes do not count."""
    args, depth, quote, cur = [], 0, "", ""
    for ch in text:
        if quote:
            cur += ch
            if ch == quote:
                quote = ""
            continue
        if ch in "\"'":
            quote, cur = ch, cur + ch
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            args.append(cur.strip())
            cur = ""
        else:
            cur += ch
    args.append(cur.strip())
    return [a for a in args if a]


def _arity_bounds(params: str) -> tuple:
    """(fewest, most) arguments a call may legally pass.

    Parameters with a default are optional, so a function written
    `fill_form(a, b, out="x")` is correctly called with two arguments and must not be
    reported as a mismatch by a check that simply counts the parameters listed.
    """
    args = split_args(params)
    required = sum(1 for a in args if "=" not in a and not a.startswith("*"))
    return required, len(args)


def check_arity(items) -> list:
    """Does each verify call the item's functions with a legal number of arguments?"""
    findings = []
    for item in (as_dict(i) for i in items):
        detail, verify = item.get("detail") or "", item.get("verify") or ""
        if not detail or not verify:
            continue
        signatures = {}
        for match in _SIGNATURE.finditer(detail):
            name = match.group(1)
            if name in _NOT_A_FUNCTION:
                continue
            signatures[name] = _arity_bounds(match.group(2))
        if not signatures:
            continue
        for match in _CALL.finditer(verify):
            name, params = match.group(1), match.group(2)
            bounds = signatures.get(name)
            if bounds is None:
                continue
            low, high = bounds
            got = len(split_args(params))
            if got < low or got > high:
                expected = f"{low}" if low == high else f"{low} to {high}"
                findings.append(Finding(
                    "arity", [item["id"]],
                    f"{name}() is specified with {expected} argument(s) but the verify "
                    f"calls it with {got}",
                    f"make the verify pass {expected} argument(s), or correct the "
                    f"signature in the specification"))
    return findings


def check_file_claims(items) -> list:
    """Two unfinished items both claiming to CREATE the same file.

    Only unfinished ones: two items that both edited a file and both SUCCEEDED are
    history, and reporting them as a conflict stops plans that are nearly done. That
    mistake was made on 2026-09-14 against two items verified hours earlier.
    """
    claims = {}
    for item in (as_dict(i) for i in items):
        if (item.get("status") or "") in DONE:
            continue
        for match in _CREATES.finditer(item.get("detail") or ""):
            claims.setdefault(match.group(1), []).append(item["id"])
    return [Finding("file-claim", sorted(ids),
                    f"{len(ids)} unfinished items each claim to CREATE {path}",
                    f"only one item may create {path}; the others should add to it or "
                    f"depend on it")
            for path, ids in claims.items() if len(ids) > 1]


def check_plan(items) -> list:
    """Every deterministic check, run over a plan's items."""
    return check_arity(items) + check_file_claims(items)
