"""Tests for the failure classifier.

Every case below is a REAL failure from 2026-09-13/14, copied out of the transcripts of
that night's runs. That is the point of this file: it is not a suite of invented
examples, it is the record of what actually went wrong, and it grows every time
something new goes wrong. A classifier tested on imagined failures classifies imagined
failures.

Run it directly, like the others:  python3 test_diagnose.py
"""
import sys

sys.path.insert(0, "/var/home/admin/jarvis/brain")
import diagnose  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


# --------------------------------------------------------------- the corpus
# (label, the output as it really appeared, expected kind, expected owner)
CORPUS = [
    (
        "a library the image does not have (click)",
        "visual_lookup/tests/test_cli.py:3: in <module>\n"
        "    from click.testing import CliRunner\n"
        "E   ModuleNotFoundError: No module named 'click'",
        "missing_module", diagnose.OPERATOR,
    ),
    (
        "a module the PLAN is supposed to write",
        "E   ModuleNotFoundError: No module named 'visual_lookup'",
        "missing_module", diagnose.PLAN,
    ),
    (
        "the runner itself raised (the `data` crash)",
        "Traceback (most recent call last):\n"
        '  File "/var/home/admin/jarvis/brain/devteam.py", line 890, in run_item\n'
        '    notes=(data.get("notes") or "")[:800])\n'
        "NameError: name 'data' is not defined",
        "harness", diagnose.HARNESS,
    ),
    (
        "the editor wrote a path that did not change (wrong root)",
        "Traceback (most recent call last):\n"
        '  File "/var/home/admin/jarvis/brain/devteam.py", line 872, in run_item\n'
        "    err, suggested = _editor_attempt(conn, item, design, workspace, last_error, log)\n"
        "ValueError: not enough values to unpack (expected 2, got 0)",
        "harness", diagnose.HARNESS,
    ),
    (
        "pytest treating a parameter as a fixture (the mock_answer trap)",
        "def test_full_pipeline(..., mock_answer):\n"
        ">       assert captured.out == mock_answer + \"\\n\"\n"
        "E       AssertionError: assert '<pytest_fixt...' == 'mock_answer\\n'\n"
        "E         - mock_answer\n"
        "E         + <pytest_fixture(<function mock_answer at 0x7f1dcfbdd760>)>",
        "test_bug", diagnose.CODER,
    ),
    (
        "a fixture that does not exist",
        "E       fixture 'mock_candidate' not found",
        "test_bug", diagnose.CODER,
    ),
    (
        "a signature that drifted",
        "TypeError: build_prompt() takes 1 positional argument but 2 were given",
        "signature", diagnose.CODER,
    ),
    (
        "a name that is not exported",
        "ImportError: cannot import name 'read_image' from 'visual_lookup.read'",
        "cannot_import", diagnose.CODER,
    ),
    (
        "a whole-file write that would delete another item's functions",
        "REFUSED visual_lookup/read/__init__.py: this write would REMOVE build_prompt, "
        "parse_vision_response, query_vision from __init__.py, which another item put there",
        "clobber", diagnose.CODER,
    ),
    (
        "an artifact nothing creates",
        "FileNotFoundError: [Errno 2] No such file or directory: 'tests/fixtures/sample_form.pdf'",
        "missing_artifact", diagnose.PLAN,
    ),
    (
        "a check that ran and said no",
        ">       assert f.get_fields(), 'sample_form.pdf has no AcroForm fields'\n"
        "E       AssertionError: sample_form.pdf has no AcroForm fields",
        "assertion", diagnose.CODER,
    ),
]

PLAN_PATHS = {
    "tests/fixtures/sample_form.pdf",
    "tests/fixtures/sample_flat.pdf",
    "visual_lookup/tests/test_cli.py",
}


def run():
    print("\n-- the corpus (every case is a real failure from 2026-09-13/14) --")
    for label, output, kind, owner in CORPUS:
        got = diagnose.classify(output, provided={"visual_lookup", "documents"},
                                plan_paths=PLAN_PATHS)
        check(f"{label} -> {kind}/{owner}",
              got.kind == kind and got.owner == owner,
              f"got {got.kind}/{got.owner}")

    print("\n-- who has to fix it --")
    coder = diagnose.classify("AssertionError: nope")
    check("a coder failure is retryable", coder.retryable is True)
    check("and does not stop the item", coder.stops_the_item is False)

    operator = diagnose.classify("ModuleNotFoundError: No module named 'click'")
    check("an operator failure is NOT retryable", operator.retryable is False)
    check("and stops the item after one attempt", operator.stops_the_item is True)
    check("and says plainly that a respec cannot fix it",
          "Do NOT respec" in operator.advice, operator.advice)

    harness = diagnose.classify(
        '  File "/var/home/admin/jarvis/brain/devteam.py", line 12, in x\n'
        "TypeError: bad")
    check("a harness fault stops the item", harness.stops_the_item is True)
    check("and is not blamed on the coder",
          "not the item's fault" in harness.advice, harness.advice)

    print("\n-- a plan-provided module is not an operator problem --")
    provided = diagnose.classify("ModuleNotFoundError: No module named 'documents'",
                                 provided={"documents"})
    check("it is the plan's", provided.owner == diagnose.PLAN, provided.owner)
    check("and is a dependency problem, not a missing package",
          "depends on work that is not done" in provided.advice, provided.advice)

    print("\n-- an artifact the plan names is a dependency; one it does not is a hole --")
    named = diagnose.classify(
        "FileNotFoundError: [Errno 2] No such file or directory: 'tests/fixtures/sample_form.pdf'",
        plan_paths=PLAN_PATHS)
    unnamed = diagnose.classify(
        "FileNotFoundError: [Errno 2] No such file or directory: '/nowhere/at/all.txt'",
        plan_paths=PLAN_PATHS)
    check("a named artifact reads as a dependency",
          "earlier item has to produce it" in named.advice, named.advice)
    check("an unnamed artifact reads as a hole in the plan",
          "NO item in this plan creates it" in unnamed.advice, unnamed.advice)

    print("\n-- something unrecognised still says something useful --")
    unknown = diagnose.classify("the model replied in Klingon")
    check("it falls back to the coder", unknown.owner == diagnose.CODER, unknown.owner)
    check("and is retryable", unknown.retryable is True)
    check("and never raises on empty input", diagnose.classify("").kind == "unknown")
    check("nor on None", diagnose.classify(None).kind == "unknown")

    print("\n" + ("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}"))
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(run())
