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

FORMAT_HINTS = {
    "when": "a date or year (e.g. 'May 18, 2018' or '1997')",
    "who": "a person, group, or organization name",
    "where": "a place, location, or venue name",
    "how_many": "a number (digits preferred, e.g. '7')",
    "how many": "a number (digits preferred, e.g. '7')",
    "which": "the specific entity name asked about",
    "what": "the shortest entity, title, or phrase that answers the question",
}
DEFAULT_HINT = "the shortest span that directly answers the question"


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
        x = lines[-1]
    x = re.sub(r"^[-*]\s*", "", x).strip()
    return x.strip(" \"'`")


def is_unknown(x):
    return _norm(x) in UNK


def infer_question_type(question, question_type=None):
    if question_type and question_type in FORMAT_HINTS:
        return question_type
    q = question.lower()
    if "how many" in q:
        return "how_many"
    for w in ("when", "who", "where", "which", "what"):
        if re.search(rf"\b{w}\b", q):
            return w
    return "other"


PROMPT = """Extract the answer to the question from the context.

Question:
{question}

Context:
{context}

The answer must be {hint}.

Steps:
1. List up to 3 candidate spans copied verbatim from the context, one per
   line, prefixed with "CANDIDATE:".
2. On the last line, output ONLY the best candidate as the final short
   answer (a few words, no full sentence, no label, no punctuation at the
   end).
Only output "unknown" on the last line if no candidate exists at all.
"""


def generation_repair_v3(question, context, baseline_answer=None,
                         question_type=None):
    """baseline_answer is accepted for interface compatibility but is only
    used as a last-resort fallback for degenerate model outputs."""
    hint = FORMAT_HINTS.get(
        infer_question_type(question, question_type), DEFAULT_HINT
    )

    raw = gpt(PROMPT.format(question=question, context=context, hint=hint))
    out = _clean(raw)

    if not out and baseline_answer:
        return _clean(baseline_answer)
    return out