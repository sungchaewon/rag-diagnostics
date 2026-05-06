import json
from collections import defaultdict

RESULT_PATH = "outputs/results.json"


def avg(results, cond, metric):
    if not results:
        return 0.0
    return sum(r[cond][metric] for r in results) / len(results)


with open(RESULT_PATH, "r", encoding="utf-8") as f:
    results = json.load(f)

groups = defaultdict(list)
for r in results:
    qtype = r.get("question_type", "other")
    groups[qtype].append(r)

order = ["who", "when", "where", "how_many", "what", "other"]

for qtype in order:
    subset = groups.get(qtype, [])
    if not subset:
        continue

    print(f"\n[{qtype}] n={len(subset)}")
    for cond in ["baseline", "oracle_r", "oracle_re", "oracle_g"]:
        em = avg(subset, cond, "em")
        f1 = avg(subset, cond, "f1")
        print(f"{cond:10s} | EM: {em:.4f} | F1: {f1:.4f}")