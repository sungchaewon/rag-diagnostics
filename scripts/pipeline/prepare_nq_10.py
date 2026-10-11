import json
from datasets import load_dataset

dataset = load_dataset("natural_questions", split="validation")

samples = []
for item in dataset:
    short_answers = item["annotations"]["short_answers"]

    # short answer 없는 샘플 제외
    if not short_answers or not short_answers[0]["text"]:
        continue

    answer = short_answers[0]["text"][0]
    question = item["question"]["text"]

    doc_tokens = item["document"]["tokens"]
    start = short_answers[0]["start_token"][0]
    end = short_answers[0]["end_token"][0]

    golden_passage = " ".join(
        [
            t
            for t, is_html in zip(
                doc_tokens["token"][max(0, start - 50): end + 50],
                doc_tokens["is_html"][max(0, start - 50): end + 50],
            )
            if not is_html
        ]
    ).strip()

    if not golden_passage:
        continue

    samples.append(
        {
            "id": item["id"],
            "question": question,
            "golden_answers": [answer],
            "golden_passage": golden_passage,
            "split_answer_type": "unknown",
            "split_retrieval": "unknown",
        }
    )

    if len(samples) >= 10:
        break

with open("data/nq_sample/nq_10.jsonl", "w", encoding="utf-8") as f:
    for sample in samples:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

print(f"Saved {len(samples)} samples to data/nq_sample/nq_10.jsonl")