from datasets import load_dataset
import json

# NQ에 딸려오는 passage들만 추출
dataset = load_dataset("natural_questions", split="validation")

corpus = {}
for item in dataset:
    doc_id = item["document"]["title"]
    tokens = item["document"]["tokens"]
    text = " ".join([
        t for t, is_html in zip(
            tokens["token"], tokens["is_html"]
        ) if not is_html
    ])
    corpus[doc_id] = text

with open("data/corpus/nq_corpus.jsonl", "w") as f:
    for pid, text in corpus.items():
        f.write(json.dumps({"id": pid, "contents": text}) + "\n")

print(f"Corpus size: {len(corpus)}")