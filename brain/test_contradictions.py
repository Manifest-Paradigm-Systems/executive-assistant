"""Tests for the deterministic contradiction checks.

The cases marked "live" are the reviewer's actual errors from 2026-09-14, and the point
of this file is that each one becomes impossible rather than merely rarer.
"""
import sys

sys.path.insert(0, "/var/home/admin/jarvis/brain")
import contradictions  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def item(iid, detail, verify="", status="pending"):
    return {"id": iid, "detail": detail, "verify": verify, "status": status}


def run():
    print("\n-- arity --")
    # LIVE: the reviewer said three-were-given-two and rejected the plan on it.
    live = item(
        "DOCS:JV-005",
        "Modify `documents/receipt.py` to update `generate_rent_receipt` to accept three "
        "arguments: output path, tenant name, and street address.",
        "python3 -c \"import documents.receipt; "
        "documents.receipt.generate_rent_receipt('tests/output/receipt.pdf', "
        "'Tenant Name', '123 Street')\"")
    check("the live miscount is not reproduced", contradictions.check_arity([live]) == [],
          contradictions.check_arity([live]))

    real = item("P:1",
                "Add `fill_form(pdf_path, values) -> str` to documents/fill.py.",
                'python3 -c "from documents.fill import fill_form; fill_form(a, b, c)"')
    got = contradictions.check_arity([real])
    check("a genuine mismatch IS reported", len(got) == 1 and got[0].kind == "arity", got)
    check("and it names the item and the counts",
          "2" in got[0].problem and "3" in got[0].problem, got[0].problem)

    defaulted = item("P:2",
                     'Add `fill_form(pdf_path, values, out_path="tests/output/f.pdf") -> str`.',
                     'python3 -c "from documents.fill import fill_form; fill_form(a, b)"')
    # A parameter with a default is optional; a check that just counts the list would
    # reject the correct call this way.
    check("a defaulted parameter is optional, not required",
          contradictions.check_arity([defaulted]) == [], contradictions.check_arity([defaulted]))

    optional_high = item("P:3",
                         'Add `f(a, b=1, c=2) -> str`.',
                         'python3 -c "from m import f; f(1, 2, 3, 4)"')
    check("too many arguments is still too many",
          len(contradictions.check_arity([optional_high])) == 1)

    prose = item("P:4", "Accept three arguments: path, name, address.",
                 'python3 -c "import m; m.g(1)"')
    check("prose with no signature is not guessed at",
          contradictions.check_arity([prose]) == [])

    check("commas inside a dict do not inflate the count",
          contradictions.split_args("a, {'x': 1, 'y': 2}, b") == ["a", "{'x': 1, 'y': 2}", "b"])
    check("commas inside a string do not either",
          contradictions.split_args("a, 'x, y', b") == ["a", "'x, y'", "b"])

    print("\n-- file claims --")
    same = [item("P:5", "Create `pkg/mod.py` with the parser."),
            item("P:6", "Create the file `pkg/mod.py` with the writer.")]
    got = contradictions.check_file_claims(same)
    check("two unfinished items creating one file is reported",
          len(got) == 1 and got[0].items == ["P:5", "P:6"], got)

    # LIVE: the reviewer reported two VERIFIED items as conflicting over one file.
    history = [item("P:7", "Create `pkg/mod.py`.", status="verified"),
               item("P:8", "Create `pkg/mod.py`.", status="pending")]
    check("finished work is history, not a claim",
          contradictions.check_file_claims(history) == [],
          contradictions.check_file_claims(history))

    both_done = [item("P:9", "Create `pkg/mod.py`.", status="verified"),
                 item("P:10", "Create `pkg/mod.py`.", status="complete")]
    check("two finished items are not a conflict either",
          contradictions.check_file_claims(both_done) == [])

    distinct = [item("P:11", "Create `pkg/a.py`."), item("P:12", "Create `pkg/b.py`.")]
    check("distinct files are not a conflict",
          contradictions.check_file_claims(distinct) == [])

    adding = [item("P:13", "Create `pkg/a.py`."),
              item("P:14", "Add `parse()` to `pkg/a.py`.")]
    check("one creating and one adding to the same file is fine",
          contradictions.check_file_claims(adding) == [],
          contradictions.check_file_claims(adding))

    print("\n-- the combined check --")
    check("check_plan runs both",
          len(contradictions.check_plan(same + [real])) == 2,
          contradictions.check_plan(same + [real]))
    check("and is empty for a clean plan",
          contradictions.check_plan([item("P:15", "Create `x.py`.", 'python3 -c "import x"')])
          == [])

    print("\n" + ("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}"))
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(run())
