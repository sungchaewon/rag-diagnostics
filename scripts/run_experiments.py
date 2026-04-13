import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from eval.metrics import compute_em, compute_f1
from src.generation.gpt4omini import generate
from src.oracles.oracle_g import generate_oracle_g
from src.oracles.oracle_r import retrieve_oracle_r
from src.oracles.oracle_re import inject_gold_passage
from src.retrieval.bm25_baseline import retrieve_bm25
from src.retrieval.reranker import rerank, rerank_oracle_re


def summarize_results(results: list[dict]):
    conditions = ["baseline", "oracle_r", "oracle_re", "oracle_g"]

    for cond in conditions:
        avg_em = sum(r[cond]["em"] for r in results) / len(results)
        avg_f1 = sum(r[cond]["f1"] for r in results) / len(results)
        print(f"{cond:10s} | EM: {avg_em:.4f} | F1: {avg_f1:.4f}")


def run_all(data_path: str, top_k: int = 10, rerank_top_k: int = 5):
    os.makedirs("results", exist_ok=True)

    with open(data_path, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]

    results = []

    for i, sample in enumerate(samples):
        q = sample["question"]
        gold_answers = sample["golden_answers"]
        gold_passage = sample["golden_passage"]

        print(f"[{i+1}/{len(samples)}] {q[:80]}...")

        # Baseline
        retrieved = retrieve_bm25(q, top_k=top_k)
        reranked = rerank(q, retrieved, top_k=rerank_top_k)
        pred = generate(q, reranked)

        baseline_result = {
            "pred": pred,
            "em": compute_em(pred, gold_answers),
            "f1": compute_f1(pred, gold_answers),
        }

        # Oracle-R
        or_retrieved = retrieve_oracle_r(q, gold_passage, top_k=top_k)
        or_reranked = rerank(q, or_retrieved, top_k=rerank_top_k)
        or_pred = generate(q, or_reranked)

        oracle_r_result = {
            "pred": or_pred,
            "em": compute_em(or_pred, gold_answers),
            "f1": compute_f1(or_pred, gold_answers),
        }

        # Oracle-Re
        ore_retrieved = retrieve_bm25(q, top_k=top_k)
        ore_candidates = inject_gold_passage(ore_retrieved, gold_passage, top_k=top_k)
        ore_reranked = rerank_oracle_re(q, ore_candidates, gold_passage, top_k=rerank_top_k)
        ore_pred = generate(q, ore_reranked)

        oracle_re_result = {
            "pred": ore_pred,
            "em": compute_em(ore_pred, gold_answers),
            "f1": compute_f1(ore_pred, gold_answers),
        }

        # Oracle-G
        og_pred = generate_oracle_g(q, gold_passage)

        oracle_g_result = {
            "pred": og_pred,
            "em": compute_em(og_pred, gold_answers),
            "f1": compute_f1(og_pred, gold_answers),
        }

        results.append({
            "id": sample.get("id", i),
            "question": q,
            "golden_answers": gold_answers,
            "golden_passage": gold_passage,
            "split_answer_type": sample.get("split_answer_type", "unknown"),
            "split_retrieval": sample.get("split_retrieval", "unknown"),
            "baseline": baseline_result,
            "oracle_r": oracle_r_result,
            "oracle_re": oracle_re_result,
            "oracle_g": oracle_g_result,
        })

        if (i + 1) % 10 == 0:
            with open("results/results_checkpoint.json", "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Checkpoint saved at {i+1}")

    with open("results/results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    summarize_results(results)


if __name__ == "__main__":
    run_all("data/nq_sample/nq_100.jsonl")