from datasets import load_dataset
import json
import os

dataset = load_dataset("natural_questions", split="validation", streaming=True)

samples = []
for item in dataset:
    short_answers = item["annotations"]["short_answers"]
    if not short_answers or not short_answers[0]["text"]:
        continue

    answer = short_answers[0]["text"][0]
    question = item["question"]["text"]

    doc_tokens = item["document"]["tokens"]
    start = short_answers[0]["start_token"][0]
    end = short_answers[0]["end_token"][0]

    golden_passage = " ".join(
        [
            t for t, is_html in zip(
                doc_tokens["token"][max(0, start - 50): end + 50],
                doc_tokens["is_html"][max(0, start - 50): end + 50]
            )
            if not is_html
        ]
    )

    samples.append({
        "id": item["id"],
        "question": question,
        "golden_answers": [answer],
        "golden_passage": golden_passage
    })

    if len(samples) >= 100:
        break

os.makedirs("data/nq_sample", exist_ok=True)

with open("data/nq_sample/nq_100.jsonl", "w", encoding="utf-8") as f:
    for s in samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

print(f"Saved {len(samples)} samples")