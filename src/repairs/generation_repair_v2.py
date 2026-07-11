import re
from collections import Counter

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


def _norm(x):
    if x is None:
        return ""
    x = str(x).strip().lower()
    x = re.sub(r"^(final answer|answer)\s*:\s*", "", x)
    x = x.strip(" \n\t\"'`.")
    x = re.sub(r"\s+", " ", x)
    return x


def _clean(x):
    if x is None:
        return ""
    x = str(x).strip()
    x = re.sub(r"(?i)^(final answer|answer)\s*:\s*", "", x).strip()
    lines = [line.strip() for line in x.splitlines() if line.strip()]
    if lines:
        x = lines[-1]  # answer is on the last line (after the quoted span)
    x = re.sub(r"^[-*]\s*", "", x).strip()
    return x.strip(" \"'`")


def is_unknown(x):
    return _norm(x) in UNK


def _tokens(x):
    return re.findall(r"[a-z0-9]+", _norm(x))


def soft_grounded(ans, ctx, threshold=0.6):
    """Token recall of answer tokens against the context.

    Unlike V1's exact-substring check this tolerates surface variation
    (e.g. 'May 18th, 2018' vs 'May 18, 2018', reordered names).
    """
    a = _tokens(ans)
    if not a:
        return False
    c = Counter(_tokens(ctx))
    covered = sum(min(n, c[t]) for t, n in Counter(a).items())
    return covered / len(a) >= threshold


def too_long(ans, limit=12):
    return not ans or len(ans.split()) > limit


PROMPT = """You are fixing the answer of a RAG system.

Question:
{question}

Context:
{context}

Initial answer (may be wrong):
{base}

Instructions:
1. Find the single passage span that best answers the question.
2. On the first line, write: EVIDENCE: <verbatim quote of that span>
3. On the last line, write ONLY the minimal short answer span
   (a few words, no full sentence, no label).
If the initial answer is already correct, output it as the last line.
If the context truly has no answer, output: unknown
"""


def generation_repair_v2(question, context, baseline_answer,
                         grounding_threshold=0.6):
    base = _clean(baseline_answer)

    raw = gpt(PROMPT.format(question=question, context=context, base=base))
    out = _clean(raw)

    # fall back to baseline only on clearly degenerate outputs
    if not out:
        return base
    if is_unknown(out) and not is_unknown(base):
        return base
    if too_long(out) and not is_unknown(base):
        return base
    # soft grounding gate (V1 used exact substring here)
    if not soft_grounded(out, context, grounding_threshold) and not is_unknown(base):
        return base

    return out