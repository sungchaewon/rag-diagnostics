import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from eval.metrics import compute_em, compute_f1
from src.generation.gpt4omini import generate
from src.oracles.oracle_g import generate_oracle_g
from src.oracles.oracle_r import retrieve_oracle_r
from src.oracles.oracle_re import inject_gold_passage
from src.retrieval.bm25_baseline import retrieve_bm25, get_bm25_rank_info
from src.retrieval.reranker import rerank, rerank_oracle_re


def summarize_results(results: list[dict]):
    conditions = ["baseline", "oracle_r", "oracle_re", "oracle_g"]

    for cond in conditions:
        avg_em = sum(r[cond]["em"] for r in results) / len(results)
        avg_f1 = sum(r[cond]["f1"] for r in results) / len(results)
        print(f"{cond:10s} | EM: {avg_em:.4f} | F1: {avg_f1:.4f}")



def infer_question_type(question: str) -> str:
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
    if q.startswith("which"):
        return "which"
    if q.startswith("why"):
        return "why"
    if q.startswith("how"):
        return "how"

    return "other"


def get_best_oracle_stage(baseline_result, oracle_r_result, oracle_re_result, oracle_g_result):
    """
    Choose the minimal oracle intervention that improves over baseline.
    If no oracle improves over baseline, return baseline.
    Ties among oracle stages are broken toward the weaker intervention:
    oracle_r -> oracle_re -> oracle_g.
    """
    baseline_score = (baseline_result["f1"], baseline_result["em"])

    oracle_candidates = {
        "oracle_r": oracle_r_result,
        "oracle_re": oracle_re_result,
        "oracle_g": oracle_g_result,
    }

    improved = {
        name: result
        for name, result in oracle_candidates.items()
        if (result["f1"], result["em"]) > baseline_score
    }

    if not improved:
        return "baseline"

    priority = {
        "oracle_r": 2,
        "oracle_re": 1,
        "oracle_g": 0,
    }

    best = max(
        improved.items(),
        key=lambda x: (x[1]["f1"], x[1]["em"], priority[x[0]])
    )[0]

    return best



def run_all(
    data_path: str,
    top_k: int = 10,
    rerank_top_k: int = 5,
    output_dir: str = "outputs/cikm/triviaqa_5_v4_smoke",
):
    os.makedirs(output_dir, exist_ok=True)

    with open(data_path, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]

    results = []

    for i, sample in enumerate(samples):
        q = sample["question"]
        gold_answers = sample["golden_answers"]
        gold_passage = sample["golden_passage"]
        bm25_info = get_bm25_rank_info(q, gold_passage, top_k=top_k)

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
            "question_type": sample.get("question_type") or infer_question_type(q),
            "golden_answers": gold_answers,
            "golden_passage": gold_passage,
            "split_answer_type": sample.get("split_answer_type", "unknown"),
            "split_retrieval": "BM25-Hit" if bm25_info["bm25_hit"] else "BM25-Miss",
            "bm25_hit": bm25_info["bm25_hit"],
            "bm25_gold_rank": bm25_info["bm25_gold_rank"],
            "rank_bucket": bm25_info["rank_bucket"],
            "best_oracle_stage": get_best_oracle_stage(
                baseline_result,
                oracle_r_result,
                oracle_re_result,
                oracle_g_result,
            ),
            "baseline": baseline_result,
            "oracle_r": oracle_r_result,
            "oracle_re": oracle_re_result,
            "oracle_g": oracle_g_result,
        })

        if (i + 1) % 50 == 0:
            checkpoint_path = os.path.join(output_dir, "results_checkpoint.json")
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Checkpoint saved at {i+1}: {checkpoint_path}")

    result_path = os.path.join(output_dir, "results.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved final results to: {result_path}")


if __name__ == "__main__":
    run_all(
        "data/triviaqa_sample/triviaqa_5_short_v4.jsonl",
        output_dir="outputs/cikm/triviaqa_5_v4_smoke",
    )