import json
import argparse
from pathlib import Path

from src.retrieval.bm25_baseline import retrieve_bm25
from src.generation.gpt4omini import generate
from src.repairs.retrieval_repair import retrieval_repair
from src.repairs.generation_repair import generation_repair
from eval.metrics import compute_em, compute_f1


def normalize_contexts(contexts):
    normalized = []
    for c in contexts:
        if isinstance(c, dict):
            normalized.append(c.get("contents") or c.get("text") or str(c))
        else:
            normalized.append(str(c))
    return normalized


def get_gold_answers(sample):
    gold = (
        sample.get("golden_answers")
        or sample.get("gold_answers")
        or sample.get("answers")
        or sample.get("answer")
    )

    if isinstance(gold, str):
        return [gold]
    return gold



def load_done(output_path):
    p = Path(output_path)
    if not p.exists():
        return []
    try:
        data = json.load(open(p))
        if isinstance(data, list):
            return data
    except Exception:
        return []
    return []


def save_results(output_path, results):
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def run_repair_experiments(data_path, output_path, n_samples=None, resume=True):
    with open(data_path) as f:
        samples = [json.loads(l) for l in f]

    if n_samples:
        samples = samples[:n_samples]

    results = load_done(output_path) if resume else []
    start = len(results)

    class SimpleRetriever:
        def retrieve(self, query, top_k=10):
            contexts = retrieve_bm25(query, top_k=top_k)
            return normalize_contexts(contexts)

    class SimpleGenerator:
        def generate(self, question, contexts):
            return generate(question, contexts)

    retriever = SimpleRetriever()
    generator = SimpleGenerator()

    if start > 0:
        print(f"Resume from {start}/{len(samples)}", flush=True)

    for i in range(start, len(samples)):
        sample = samples[i]
        q = sample["question"]
        gold_ans = get_gold_answers(sample)

        rank_bucket = sample.get("rank_bucket", "unknown")
        question_type = sample.get("question_type", "other")
        best_oracle = sample.get("best_oracle_stage", "baseline")

        print(f"[{i+1}/{len(samples)}] {q[:50]}...", flush=True)

        # 1. Baseline
        baseline_contexts = retriever.retrieve(q, top_k=10)
        baseline_answer = generate(q, baseline_contexts)

        # 2. Retrieval Repair
        rr = retrieval_repair(
            q,
            retriever,
            generator,
            top_k=10
        )
        retrieval_repair_answer = rr["repaired_answer"]

        # 3. Generation Repair
        context_str = "\n\n".join(baseline_contexts[:5])
        generation_repair_answer = generation_repair(
            q,
            context_str,
            baseline_answer
        )

        result = {
            "id": sample.get("id", i),
            "question": q,
            "golden_answers": gold_ans,
            "rank_bucket": rank_bucket,
            "question_type": question_type,
            "best_oracle_stage": best_oracle,
            "baseline": {
                "pred": baseline_answer,
                "em": compute_em(baseline_answer, gold_ans),
                "f1": compute_f1(baseline_answer, gold_ans)
            },
            "retrieval_repair": {
                "rewritten_query": rr["rewritten_query"],
                "pred": retrieval_repair_answer,
                "em": compute_em(retrieval_repair_answer, gold_ans),
                "f1": compute_f1(retrieval_repair_answer, gold_ans)
            },
            "generation_repair": {
                "pred": generation_repair_answer,
                "em": compute_em(generation_repair_answer, gold_ans),
                "f1": compute_f1(generation_repair_answer, gold_ans)
            }
        }

        results.append(result)

        if (i + 1) % 100 == 0:
            save_results(output_path, results)
            print(f"Checkpoint saved at {i+1}")

    save_results(output_path, results)

    print_summary(results)


def print_summary(results):
    for cond in ["baseline", "retrieval_repair", "generation_repair"]:
        avg_em = sum(r[cond]["em"] for r in results) / len(results)
        avg_f1 = sum(r[cond]["f1"] for r in results) / len(results)
        print(f"{cond:20s} | EM: {avg_em:.4f} | F1: {avg_f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--no_resume", action="store_true")
    args = parser.parse_args()

    run_repair_experiments(args.data, args.output, args.n_samples, resume=not args.no_resume)
