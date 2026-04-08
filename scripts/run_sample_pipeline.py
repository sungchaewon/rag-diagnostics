import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.retrieval.bm25_baseline import retrieve_top_k
from src.generation.gpt4omini import generate_answer
from src.oracles.oracle_g import build_oracle_generator_context
from eval.metrics import exact_match_score, f1_score

def main():
    path = "data/nq_sample/sample.jsonl"
    baseline_results = []
    oracle_g_results = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            question = item["question"]
            gold_answer = item["gold_answer"]
            gold_evidence = item["gold_evidence"]
            corpus = item["corpus"]

            # Baseline
            top_docs = retrieve_top_k(question, corpus, k=3)
            baseline_context = "\n\n".join([doc for doc, _ in top_docs])
            baseline_pred = generate_answer(question, baseline_context)

            baseline_results.append({
                "question": question,
                "prediction": baseline_pred,
                "gold_answer": gold_answer,
                "em": exact_match_score(baseline_pred, gold_answer),
                "f1": f1_score(baseline_pred, gold_answer),
            })

            # Oracle-G
            oracle_context = build_oracle_generator_context(gold_evidence)
            oracle_pred = generate_answer(question, oracle_context)

            oracle_g_results.append({
                "question": question,
                "prediction": oracle_pred,
                "gold_answer": gold_answer,
                "em": exact_match_score(oracle_pred, gold_answer),
                "f1": f1_score(oracle_pred, gold_answer),
            })

    baseline_em = sum(x["em"] for x in baseline_results) / len(baseline_results)
    baseline_f1 = sum(x["f1"] for x in baseline_results) / len(baseline_results)
    oracle_em = sum(x["em"] for x in oracle_g_results) / len(oracle_g_results)
    oracle_f1 = sum(x["f1"] for x in oracle_g_results) / len(oracle_g_results)

    print("Baseline EM:", baseline_em)
    print("Baseline F1:", baseline_f1)
    print("Oracle-G EM:", oracle_em)
    print("Oracle-G F1:", oracle_f1)

if __name__ == "__main__":
    main()