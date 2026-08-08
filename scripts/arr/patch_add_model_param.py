from pathlib import Path

PATCHES = [
    {
        "file": "src/repairs/generation_repair.py",
        "replacements": [
            ("def generation_repair(question, context, baseline_answer):",
             'def generation_repair(question, context, baseline_answer,\n'
             '                      model="gpt-4o-mini"):'),
            ("out = clean(gpt(prompt))",
             "out = clean(gpt(prompt, model=model))"),
        ],
    },
    {
        "file": "src/repairs/generation_repair_v2.py",
        "replacements": [
            ("def generation_repair_v2(question, context, baseline_answer,\n"
             "                         grounding_threshold=0.6):",
             'def generation_repair_v2(question, context, baseline_answer,\n'
             '                         grounding_threshold=0.6,\n'
             '                         model="gpt-4o-mini"):'),
            ("raw = gpt(PROMPT.format(question=question, context=context, base=base))",
             "raw = gpt(PROMPT.format(question=question, context=context, base=base),\n"
             "         model=model)"),
        ],
    },
    {
        "file": "src/repairs/generation_repair_v3.py",
        "replacements": [
            ("def generation_repair_v3(question, context, baseline_answer=None,\n"
             "                         question_type=None):",
             'def generation_repair_v3(question, context, baseline_answer=None,\n'
             '                         question_type=None,\n'
             '                         model="gpt-4o-mini"):'),
            ("raw = gpt(PROMPT.format(question=question, context=context, hint=hint))",
             "raw = gpt(PROMPT.format(question=question, context=context, hint=hint),\n"
             "         model=model)"),
        ],
    },
]


def main():
    for spec in PATCHES:
        path = Path(spec["file"])
        if not path.exists():
            print(f"[SKIP] {path} not found")
            continue
        text = path.read_text()
        changed = False
        for old, new in spec["replacements"]:
            if old not in text:
                print(f"[MISS] {path}: pattern not found, skipping this "
                      f"replacement -- file may differ from expected:")
                print(f"       {old[:80]}...")
                continue
            if text.count(old) > 1:
                print(f"[WARN] {path}: pattern appears {text.count(old)} "
                      f"times, expected 1 -- skipping to avoid ambiguous edit")
                continue
            text = text.replace(old, new)
            changed = True
        if changed:
            path.write_text(text)
            print(f"[OK] {path} patched")
        else:
            print(f"[NOCHANGE] {path}")


if __name__ == "__main__":
    main()