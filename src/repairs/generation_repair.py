import re
from src.generation.gpt4omini import gpt

UNK = {
    "",
    "unknown",
    "i don't know",
    "i do not know",
    "not enough information",
    "not enough evidence",
    "cannot be determined",
    "unanswerable",
    "none",
    "n/a",
}


def norm(x):
    if x is None:
        return ""
    x = str(x).strip().lower()
    x = re.sub(r"^(final answer|answer)\s*:\s*", "", x)
    x = x.strip(" \n\t\"'`.")
    x = re.sub(r"\s+", " ", x)
    return x


def clean(x):
    if x is None:
        return ""

    x = str(x).strip()
    x = re.sub(r"(?i)^(final answer|answer)\s*:\s*", "", x).strip()

    lines = [line.strip() for line in x.splitlines() if line.strip()]
    if lines:
        x = lines[0]

    x = re.sub(r"^[-*]\s*", "", x).strip()
    return x.strip(" \"'`")


def is_unknown(x):
    return norm(x) in UNK


def in_context(ans, ctx):
    ans = norm(ans)
    ctx = norm(ctx)

    if not ans or is_unknown(ans):
        return False

    return ans in ctx


def too_long(ans, limit=12):
    return not ans or len(ans.split()) > limit


def generation_repair(question, context, baseline_answer):
    base = clean(baseline_answer)

    if in_context(base, context):
        return base

    prompt = f"""
You are given a question, retrieved context, and an initial answer from a RAG system.

Check the initial answer using only the context.

Rules:
- If the initial answer is supported, return it exactly.
- If the initial answer is wrong and the context clearly supports another answer, return the corrected short answer.
- If you are unsure, return the initial answer.
- If there is no answer in the context and the initial answer is empty or unknown, return "unknown".
- Return only the final short answer.

Question:
{question}

Context:
{context}

Initial answer:
{base}

Final answer:
"""

    out = clean(gpt(prompt))

    if not out:
        return base

    if is_unknown(out) and not is_unknown(base):
        return base

    if too_long(out) and not is_unknown(base):
        return base

    if not in_context(out, context) and not is_unknown(base):
        return base

    return out
