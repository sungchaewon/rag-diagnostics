import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.retrieval.bm25_baseline import retrieve_bm25


def avg(results, cond, metric):
    if not results:
        return 0.0
    return sum(r[cond][metric] for r in results) / len(results)


def summarize(name, subset):
    print(f"\n[{name}] n={len(subset)}")
    for cond in ["baseline", "oracle_r", "oracle_re", "oracle_g"]:
        em = avg(subset, cond, "em")
        f1 = avg(subset, cond, "f1")
        print(f"{cond:10s} | EM: {em:.4f} | F1: {f1:.4f}")


with open("outputs/results.json", "r", encoding="utf-8") as f:
    results = json.load(f)

hit, miss = [], []

for r in results:
    retrieved = retrieve_bm25(r["question"], top_k=10)
    if r["golden_passage"] in retrieved:
        hit.append(r)
    else:
        miss.append(r)

summarize("ALL", results)
summarize("BM25-HIT", hit)
summarize("BM25-MISS", miss)