"""What actually went wrong, and whose job it is to fix it.

Every failure this system hit on 2026-09-13/14 was distinguishable by a pattern in the
output, and each needed a DIFFERENT owner:

  * a library missing from the verify image is an OPERATOR's job — nothing can be
    installed while a check runs, so respec'ing the item only burns attempts;
  * an artifact nothing creates is the PLAN's job;
  * a bad mock, a signature mismatch, a broken test is the CODER's;
  * a traceback raised inside this program is a HARNESS fault and must never be retried
    as though the model were at fault.

Handing all four to the same place is why a missing library became five respecs and a
missing fixture burned forty-four runs. The director is a good architect and a poor
librarian.

Nothing here calls a model. It is string analysis over output we already have, so it is
cheap, deterministic and testable — which is the point: reading logs and saying "ah, it
needs click" is what a person did all night, and it is the part that has to become
mechanical before the team can run without one.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

OPERATOR = "operator"
PLAN = "plan"
CODER = "coder"
HARNESS = "harness"

_HARNESS_DIR = os.path.dirname(os.path.abspath(__file__))

# First match wins, so these are ordered most specific first. Every pattern below was
# written from a real failure, not imagined — see test_diagnose.py for the corpus.
_NEEDS = re.compile(r"ModuleNotFoundError: No module named '([\w.]+)'")
_FIXTURE_PARAM = re.compile(r"<pytest_fixture\(<function ([\w]+)")
_FIXTURE_NAMED = re.compile(r"fixture '([\w]+)' not found")
_TYPE_ARGS = re.compile(r"(\w+)\(\) takes (\d+) positional argument?s? but (\d+) (?:was|were) given")
_CANNOT_IMPORT = re.compile(r"ImportError: cannot import name '([\w]+)' from '([\w.]+)'")
_REFUSED = re.compile(r"REFUSED ([^\s:]+): this write would REMOVE")
_NO_FILE = re.compile(r"(?:FileNotFoundError|No such file or directory)[^\n']*'([^']+)'")
_TIMEOUT = re.compile(r"verify timed out after (\d+)s")
_SYNTAX = re.compile(r"SyntaxError: ([^\n]{0,120})")
_ASSERT = re.compile(r"AssertionError[^\n]{0,160}")
_FRAME = re.compile(r'File "([^"]+)", line (\d+)')

# Names that are only older names for something the image already has. For these the
# operator advice is the wrong advice: adding PyPDF2 alongside pypdf, or Pillow
# alongside PIL, installs the same library twice under two names, one of them dead.
# Every one of these pairs was written by a coder reaching for the name it remembered.
_ALIASES = {
    "PyPDF2": "pypdf",
    "PIL": "Pillow",
    "fitz": "pymupdf",
    "yaml": "PyYAML",
    "sklearn": "scikit-learn",
}


@dataclass
class Diagnosis:
    kind: str
    owner: str
    subject: str
    advice: str
    retryable: bool
    evidence: str = ""

    @property
    def stops_the_item(self) -> bool:
        """Should this end the item instead of spending the remaining attempts?

        An operator fault and a harness fault are both beyond the coder's reach.
        Retrying each of them three times is exactly how a missing library came to look
        like a coding problem.
        """
        return self.owner in (OPERATOR, HARNESS)

    def __str__(self) -> str:
        return f"{self.kind} ({self.owner}): {self.subject or '-'}"


def _harness_raised(text: str) -> str:
    """The first traceback frame pointing inside this program, if there is one.

    Absolute paths only: see the note in the loop.

    A frame in this directory means OUR code raised — not the code under test, and not
    the verify command. That distinction is the difference between "the coder has work
    to do" and "the harness is broken", and the three crash bugs of 2026-09-13 all
    looked like the former while being the latter.
    """
    for match in _FRAME.finditer(text or ""):
        path = match.group(1)
        # Only an ABSOLUTE path can name a file in this program. `abspath()` here would
        # resolve "<string>" — the frame a `python3 -c` verify always shows — against
        # the process's working directory, which IS the harness directory, so every
        # ordinary verify traceback would be reported as our own code raising.
        if not os.path.isabs(path):
            continue
        if os.path.normpath(path).startswith(_HARNESS_DIR + os.sep):
            return f"{os.path.basename(path)}:{match.group(2)}"
    return ""


def classify(output: str, *, provided=frozenset(), plan_paths=frozenset()) -> Diagnosis:
    """Whose problem is this failure, and what should be said about it.

    `provided` is the set of top-level module names the plan itself creates; `plan_paths`
    the file paths its items name. Both let the classifier tell "nobody has written this
    yet" from "this does not exist anywhere", which are different problems with
    different fixes.
    """
    text = output or ""

    where = _harness_raised(text)
    if where:
        return Diagnosis(
            "harness", HARNESS, where,
            f"the runner itself raised at {where}. This is not the item's fault and the "
            f"coder cannot fix it; it needs a code change and a regression test.",
            False, text[-600:])

    match = _NEEDS.search(text)
    if match:
        name = match.group(1).split(".")[0]
        alias = _ALIASES.get(name)
        if alias:
            return Diagnosis(
                "old_name", CODER, f"{name} -> {alias}",
                f"'{name}' is the OLD name for '{alias}', which is what this environment "
                f"has. Write `import {alias}` — the old name is the same library under a "
                f"deprecated alias and is not installed. Do not ask for it to be added.",
                True, match.group(0))
        if name in provided:
            return Diagnosis(
                "missing_module", PLAN, name,
                f"'{name}' belongs to this plan but has not been written yet. The item that "
                f"creates it has to land first — this item depends on work that is not done.",
                True, match.group(0))
        return Diagnosis(
            "missing_module", OPERATOR, name,
            f"the verify environment has no '{name}', and nothing can be installed while a "
            f"check runs. An operator has to add it to sandbox/Containerfile.verify and "
            f"rebuild the image. Do NOT respec this item: no work item can fix it.",
            False, match.group(0))

    match = _FIXTURE_PARAM.search(text)
    if match:
        return Diagnosis(
            "test_bug", CODER, match.group(1),
            f"the test takes a parameter named '{match.group(1)}', and pytest reads EVERY "
            f"test parameter as a fixture request — so it received the fixture FUNCTION "
            f"OBJECT rather than a value. Rename the parameter, or use a module-level "
            f"constant instead.", True, match.group(0))

    match = _FIXTURE_NAMED.search(text)
    if match:
        return Diagnosis(
            "test_bug", CODER, match.group(1),
            f"pytest looked for a fixture named '{match.group(1)}' and there is none. "
            f"Define it, or stop asking for it.", True, match.group(0))

    match = _TYPE_ARGS.search(text)
    if match:
        fn, wanted, given = match.group(1), match.group(2), match.group(3)
        return Diagnosis(
            "signature", CODER, fn,
            f"{fn}() takes {wanted} argument(s) but was called with {given}. The call and "
            f"the definition disagree; fix whichever does not match the specification.",
            True, match.group(0))

    match = _CANNOT_IMPORT.search(text)
    if match:
        symbol, module = match.group(1), match.group(2)
        return Diagnosis(
            "cannot_import", CODER, f"{module}.{symbol}",
            f"'{symbol}' is not exported by '{module}'. Use the exact name that module "
            f"defines, and do not rename or remove what is already there.", True,
            match.group(0))

    match = _REFUSED.search(text)
    if match:
        return Diagnosis(
            "clobber", CODER, match.group(1),
            "a whole-file write was refused because it would remove things the file "
            "already defines. Write the COMPLETE file, or target a different file.",
            True, match.group(0)[:300])

    match = _NO_FILE.search(text)
    if match:
        path = match.group(1)
        rel = path.lstrip("./")
        named = any(p.endswith(rel) or rel.endswith(p) for p in plan_paths)
        if named:
            return Diagnosis(
                "missing_artifact", PLAN, path,
                f"'{path}' is named by this plan but does not exist yet, so an earlier "
                f"item has to produce it first. Check this item's dependencies.", False,
                match.group(0))
        return Diagnosis(
            "missing_artifact", PLAN, path,
            f"'{path}' does not exist and NO item in this plan creates it, so the item "
            f"cannot pass however correct the code is. Either the verify command must "
            f"produce it as part of the check, or the plan needs an item that does.",
            False, match.group(0))

    match = _TIMEOUT.search(text)
    if match:
        return Diagnosis(
            "timeout", CODER, f"{match.group(1)}s",
            "the verify itself timed out. Make it cheap: it should prove the behaviour in "
            "milliseconds, not exercise the world.", True, match.group(0))

    match = _SYNTAX.search(text)
    if match:
        return Diagnosis(
            "syntax", CODER, match.group(1)[:60],
            "the file does not parse. Fix the syntax before anything else.", True,
            match.group(0))

    match = _ASSERT.search(text)
    if match:
        return Diagnosis(
            "assertion", CODER, match.group(0)[:80],
            "the check ran and said no. Read the assertion: it names the value it "
            "expected and the value it got.", True, text[-600:])

    return Diagnosis(
        "unknown", CODER, "",
        "the failure did not match a known shape. Read the output and fix the specific "
        "thing it names.", True, text[-600:])
