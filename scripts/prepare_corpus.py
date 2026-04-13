import json
import os
from datasets import load_dataset


def build_passages(tokens, window_size=100, stride=50):
    clean_tokens = [
        t for t, is_html in zip(tokens["token"], tokens["is_html"])
        if not is_html
    ]

    passages = []
    for start in range(0, len(clean_tokens), stride):
        chunk = clean_tokens[start:start + window_size]
        if len(chunk) < 20:
            continue

        text = " ".join(chunk).strip()
        passages.append(text)

        if start + window_size >= len(clean_tokens):
            break

    return passages


def main():
    dataset = load_dataset("natural_questions", split="validation")

    seen = set()
    corpus = []

    for item in dataset:
        doc_id = item["id"]
        title = item["document"]["title"]
        tokens = item["document"]["tokens"]

        passages = build_passages(tokens, window_size=100, stride=50)

        for idx, passage in enumerate(passages):
            if passage in seen:
                continue
            seen.add(passage)

            corpus.append({
                "id": f"{doc_id}_{idx}",
                "title": title,
                "contents": passage
            })

    os.makedirs("data/corpus", exist_ok=True)

    with open("data/corpus/nq_passage_corpus.jsonl", "w", encoding="utf-8") as f:
        for row in corpus:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved {len(corpus)} passages to data/corpus/nq_passage_corpus.jsonl")


if __name__ == "__main__":
    main()