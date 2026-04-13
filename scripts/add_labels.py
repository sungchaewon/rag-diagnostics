import json

def get_question_type(question):
    q = question.lower().strip()
    if q.startswith("who"):
        return "who"
    if q.startswith("when"):
        return "when"
    if q.startswith("where"):
        return "where"
    if q.startswith("how many") or q.startswith("how much"):
        return "how_many"
    if q.startswith("what"):
        return "what"
    return "other"

INPUT_PATH = "data/nq_sample/nq_500.jsonl"
OUTPUT_PATH = "data/nq_sample/nq_500_labeled.jsonl"

with open(INPUT_PATH, "r", encoding="utf-8") as fin, open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
    for line in fin:
        item = json.loads(line)
        item["question_type"] = get_question_type(item["question"])
        fout.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Labeled file saved to {OUTPUT_PATH}")