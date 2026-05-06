import json
import os

INPUT_PATH = "data/nq_sample/nq_500.jsonl"
OUTPUT_PATH = "data/corpus/nq_passage_corpus_500.jsonl"

os.makedirs("data/corpus", exist_ok=True)

seen = set()
count = 0

with open(INPUT_PATH, "r", encoding="utf-8") as fin, open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
    for i, line in enumerate(fin):
        item = json.loads(line)
        passage = item.get("golden_passage", "").strip()

        if not passage:
            continue
        if passage in seen:
            continue

        seen.add(passage)
        fout.write(
            json.dumps(
                {"id": f"gold_{i}", "contents": passage},
                ensure_ascii=False,
            )
            + "\n"
        )
        count += 1

print(f"Mini corpus size: {count}")
print(f"Saved to: {OUTPUT_PATH}")