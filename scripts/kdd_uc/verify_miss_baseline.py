import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.retrieval.bm25_baseline import retrieve_bm25

with open("outputs/results.json", "r", encoding="utf-8") as f:
    results = json.load(f)

miss = []
miss_baseline_correct = []
miss_baseline_nonzero_f1 = []

for r in results:
    retrieved = retrieve_bm25(r["question"], top_k=10)

    # BM25-Miss: gold passage is not in original BM25 top-10
    if r["golden_passage"] not in retrieved:
        miss.append(r)

        if r["baseline"]["em"] > 0:
            miss_baseline_correct.append(r)

        if r["baseline"]["f1"] > 0:
            miss_baseline_nonzero_f1.append(r)

print("BM25-Miss total:", len(miss))
print("BM25-Miss with baseline EM > 0:", len(miss_baseline_correct))
print("BM25-Miss with baseline F1 > 0:", len(miss_baseline_nonzero_f1))

if miss_baseline_correct:
    print("\nExamples with baseline EM > 0:")
    for x in miss_baseline_correct[:10]:
        print("-" * 80)
        print("Q:", x["question"])
        print("Gold answers:", x["golden_answers"])
        print("Baseline pred:", x["baseline"]["pred"])
        print("Baseline EM:", x["baseline"]["em"])
        print("Baseline F1:", x["baseline"]["f1"])

if miss_baseline_nonzero_f1:
    print("\nExamples with baseline F1 > 0:")
    for x in miss_baseline_nonzero_f1[:10]:
        print("-" * 80)
        print("Q:", x["question"])
        print("Gold answers:", x["golden_answers"])
        print("Baseline pred:", x["baseline"]["pred"])
        print("Baseline EM:", x["baseline"]["em"])
        print("Baseline F1:", x["baseline"]["f1"])