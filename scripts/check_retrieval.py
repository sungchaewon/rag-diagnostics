import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.retrieval.bm25_baseline import retrieve_bm25

with open("data/nq_sample/nq_500_labeled.jsonl", "r", encoding="utf-8") as f:
    samples = [json.loads(l) for l in f]

hits = 0
ranks = []

for s in samples:
    retrieved = retrieve_bm25(s["question"], top_k=10)
    if s["golden_passage"] in retrieved:
        hits += 1
        ranks.append(retrieved.index(s["golden_passage"]) + 1)

print(f"BM25 Recall@10: {hits}/{len(samples)} = {hits/len(samples):.4f}")
if ranks:
    print(f"Average rank among hits: {sum(ranks)/len(ranks):.4f}")
    print(f"Ranks: {ranks}")