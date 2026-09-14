"""One-shot patch: make brain.py inject curated memory (facts + archive) instead of
only raw-turn excerpts.

Run on cerebro:  python3 patch_brain.py [--apply]
Without --apply it reports what it would change and touches nothing.
"""
import os
import py_compile
import re
import shutil
import sys
import time

P = "/var/home/admin/jarvis/brain/brain.py"

NEW_FUNC = '''def memory_block(question: str, max_facts: int = 4) -> str:
    """Curated memory for the reply prompt: verified facts first, then the archive,
    and only then the raw turn log.

    Facts cost one short line each, so this is both cheaper and sharper than the
    turn excerpts it replaces: a fact is what the house decided, whereas a turn
    excerpt is merely something that was once said.

    Never raises — this sits on the reply path, and a memory miss must degrade to
    "Jarvis remembers nothing" rather than "Jarvis is down".
    """
    lines: list[str] = []
    try:
        import db as _mem  # same directory; stdlib-only module
        conn = _mem.open_db()
        facts = _mem.search_facts(conn, question, limit=max_facts)
        chunks = _mem.search_archive(conn, question, limit=1) if len(facts) < max_facts else []
    except Exception as exc:  # noqa: BLE001
        print(f"[jarvis-brain] memory lookup failed: {exc}", flush=True)
        facts, chunks = [], []

    if facts:
        lines.append("What you already know (your verified memory — use only what is relevant, "
                     "do not read this list aloud):")
        for f in facts:
            tag = "/".join(p for p in (f["entity"], f["topic"]) if p)
            lines.append(f"- {f['statement']}" + (f"  [{tag}]" if tag else ""))
    if chunks:
        lines.append("Relevant past discussion:")
        for c in chunks:
            lines.append(f"- {c['title']}: {(c['text'] or '')[:220]}")

    if not lines:
        # Nothing curated matched — fall back to the old raw-log behaviour so
        # long-running conversations do not lose their only thread of continuity.
        hits = recall(question, limit=3)
        if hits:
            lines.append("Things said before that may bear on this (use only what is "
                         "genuinely relevant; do not read this list aloud):")
            for h in hits:
                lines.append(f"- [{time.strftime('%Y-%m-%d', time.localtime(h['ts']))} "
                             f"{h['role']}] {h['content'][:200]}")
    return "\\n".join(lines)


'''

# Matches the token-budget comment and the raw-recall block that follows it.
OLD_BLOCK = re.compile(
    r"            # Kept small on purpose:.*?\{excerpts\}\"\}\)\n",
    re.DOTALL)

NEW_BLOCK = '''            # Kept small on purpose: every token here is prompt the model must
            # read before it can say a word, and it is read BEFORE the reply
            # starts. Measured: a 4x400-char memory block cost ~2.3s of
            # time-to-first-token. Curated facts are one short line each, so this
            # is cheaper than the excerpts it replaces.
            block = memory_block(question)
            if block:
                out.append({"role": "system", "content": block})
'''


def main() -> int:
    apply = "--apply" in sys.argv
    src = open(P, encoding="utf-8").read()
    orig = src

    if "def memory_block(" in src:
        print("already patched — nothing to do")
        return 0

    m = OLD_BLOCK.search(src)
    if not m:
        print("FAILED: could not locate the recall block to replace")
        return 1
    print(f"found injection block at chars {m.start()}..{m.end()}")

    src = src[:m.start()] + NEW_BLOCK + src[m.end():]

    anchor = "def to_ag2_messages("
    idx = src.find(anchor)
    if idx == -1:
        print("FAILED: could not find to_ag2_messages to anchor the new function")
        return 1
    src = src[:idx] + NEW_FUNC + src[idx:]
    print("inserted memory_block() before to_ag2_messages()")

    if not apply:
        print("\n-- dry run; showing the patched region --")
        i = src.find("def memory_block(")
        print(src[i:i + 400])
        print("...")
        j = src.find("            block = memory_block(question)")
        print(src[j - 300:j + 200])
        return 0

    backup = f"{P}.bak-{time.strftime('%Y%m%d-%H%M%S')}"
    shutil.copy2(P, backup)
    print(f"backup: {backup}")

    tmp = P + ".new"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(src)
    try:
        py_compile.compile(tmp, doraise=True)
    except py_compile.PyCompileError as exc:
        os.unlink(tmp)
        print(f"FAILED syntax check, original untouched:\n{exc}")
        return 1
    os.replace(tmp, P)
    print("patched and syntax-checked OK")
    print(f"  {len(orig)} -> {len(src)} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
