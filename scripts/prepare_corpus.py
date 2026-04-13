from datasets import load_dataset
import json
import os

os.makedirs("data/corpus", exist_ok=True)

dataset = load_dataset("natural_questions", split="validation")

corpus = {}
for item in dataset:
    doc_id = item["document"]["title"]
    tokens = item["document"]["tokens"]
    text = " ".join(
        t for t, is_html in zip(tokens["token"], tokens["is_html"])
        if not is_html
    ).strip()

    if text:
        corpus[doc_id] = text

with open("data/corpus/nq_passage_corpus.jsonl", "w", encoding="utf-8") as f:
    for pid, text in corpus.items():
        f.write(json.dumps({"id": pid, "contents": text}, ensure_ascii=False) + "\n")

print(f"Corpus size: {len(corpus)}")