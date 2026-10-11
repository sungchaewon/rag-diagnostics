import argparse
import json
import os
import re
from datasets import load_dataset


STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "from", "by", "with",
    "and", "or", "is", "was", "were", "are", "be", "been", "being",
    "who", "what", "when", "where", "which", "why", "how",
    "did", "do", "does", "had", "has", "have",
    "next", "after", "before"
}

MONTHS = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december"
}


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_question_type(question: str) -> str:
    q = question.lower().strip()
    if q.startswith("who"):
        return "who"
    if q.startswith("when"):
        return "when"
    if q.startswith("where"):
        return "where"
    if q.startswith("what"):
        return "what"
    if q.startswith("how many"):
        return "how many"
    if q.startswith("how"):
        return "how"
    if q.startswith("which"):
        return "which"
    return "other"


def unique_answers(value, aliases):
    answers = []
    if value:
        answers.append(value)
    for a in aliases:
        if a and a not in answers:
            answers.append(a)
    return answers


def question_keywords(question: str):
    toks = normalize(question).split()
    kws = []
    for t in toks:
        if t in STOPWORDS:
            continue
        if len(t) <= 2 and not any(ch.isdigit() for ch in t):
            continue
        kws.append(t)
    return kws



def find_all_occurrences(text: str, pattern: str):
    lower = text.lower()
    pat = pattern.lower()
    positions = []
    start = 0

    while True:
        idx = lower.find(pat, start)
        if idx == -1:
            break
        positions.append(idx)
        start = idx + max(1, len(pat))

    return positions


def make_window(text: str, center_idx: int, answer_len: int, window: int = 1800):
    start = max(0, center_idx - window // 2)
    end = min(len(text), center_idx + answer_len + window // 2)

    # Move to cleaner boundary if possible
    while start > 0 and text[start] not in ".\n":
        start -= 1
    if start < len(text) and text[start] in ".\n":
        start += 1

    while end < len(text) and text[end - 1] not in ".\n":
        end += 1
        if end >= len(text):
            break

    return text[start:end].strip()


def score_window(window_text: str, title: str, question: str, answers):
    norm_window = normalize(window_text)
    norm_title = normalize(title)
    kws = question_keywords(question)

    score = 0

    main_answer = normalize(answers[0])
    if main_answer and main_answer in norm_window:
        score += 100

    if main_answer and (main_answer in norm_title or norm_title in main_answer):
        score += 30

    overlap = 0
    for kw in kws:
        if kw in norm_window or kw in norm_title:
            overlap += 1
            score += 8

    q_tokens = normalize(question).split()
    number_tokens = [t for t in q_tokens if any(ch.isdigit() for ch in t)]
    month_tokens = [t for t in q_tokens if t in MONTHS]

    number_hits = sum(1 for t in number_tokens if t in norm_window)
    month_hits = sum(1 for t in month_tokens if t in norm_window)

    score += number_hits * 25
    score += month_hits * 25

    # Important clue bonus for questions like autobiography, album, film, city, etc.
    clue_bonus_terms = [
        "autobiography", "biography", "novel", "book", "film", "movie",
        "album", "song", "single", "capital", "president", "prime", "minister",
        "mountain", "volcano", "died", "death", "premiered", "opened"
    ]

    clue_hits = 0
    for term in clue_bonus_terms:
        if term in normalize(question) and term in norm_window:
            clue_hits += 1
            score += 25

    return score, overlap, number_hits, month_hits, len(month_tokens), clue_hits


def choose_gold_passage(entity_pages, question, answers, window_size=1800):
    titles = entity_pages.get("title", [])
    contexts = entity_pages.get("wiki_context", [])

    candidates = []

    for title, ctx in zip(titles, contexts):
        if not ctx or not ctx.strip():
            continue

        # Generate windows for every occurrence of answer value and aliases.
        for ans in answers:
            if not ans:
                continue

            positions = find_all_occurrences(ctx, ans)
            if not positions:
                continue

            for pos in positions:
                win = make_window(ctx, pos, len(ans), window=window_size)
                if not win:
                    continue

                score, overlap, number_hits, month_hits, n_months, clue_hits = score_window(
                    win, title, question, answers
                )

                # Basic support filter:
                # answer exists + at least some question clues.
                if overlap < 2:
                    continue

                # If question contains a number/date, prefer windows containing it.
                # Do not hard-fail immediately, but penalize by not adding if too weak.
                q_has_number = any(any(ch.isdigit() for ch in t) for t in normalize(question).split())
                if q_has_number and number_hits == 0 and clue_hits == 0:
                    continue

                # If question contains a month, require that month in the passage.
                if n_months > 0 and month_hits == 0:
                    continue

                candidates.append((score, win, title, overlap, number_hits, month_hits, clue_hits))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="rc.wikipedia")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--sample_out", required=True)
    parser.add_argument("--corpus_out", required=True)
    parser.add_argument("--window", type=int, default=1800)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.sample_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.corpus_out), exist_ok=True)

    ds = load_dataset("mandarjoshi/trivia_qa", args.config, split=args.split)

    samples = []
    corpus = []
    skipped = 0

    for ex in ds:
        if len(samples) >= args.n:
            break

        qid = ex.get("question_id")
        question = ex.get("question", "").strip()
        answer = ex.get("answer", {})
        value = answer.get("value", "")
        aliases = answer.get("aliases", [])
        golden_answers = unique_answers(value, aliases)

        if not qid or not question or not golden_answers:
            skipped += 1
            continue

        passage = choose_gold_passage(
            ex.get("entity_pages", {}),
            question,
            golden_answers,
            window_size=args.window
        )

        if not passage:
            skipped += 1
            continue

        idx = len(samples)

        sample = {
            "id": qid,
            "question": question,
            "golden_answers": golden_answers,
            "golden_passage": passage,
            "split_answer_type": get_question_type(question),
            "split_retrieval": "unknown"
        }

        samples.append(sample)
        corpus.append({
            "id": f"gold_{idx}",
            "contents": passage
        })

    with open(args.sample_out, "w", encoding="utf-8") as f:
        for row in samples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(args.corpus_out, "w", encoding="utf-8") as f:
        for row in corpus:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved samples: {len(samples)} -> {args.sample_out}")
    print(f"Saved corpus:  {len(corpus)} -> {args.corpus_out}")
    print(f"Skipped while collecting: {skipped}")


if __name__ == "__main__":
    main()
