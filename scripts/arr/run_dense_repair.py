import json
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.retrieval.dense_baseline import retrieve_dense
from src.generation.gpt4omini import generate
from src.repairs.retrieval_repair import retrieval_repair
from src.repairs.generation_repair import generation_repair
from src.repairs.generation_repair_v2 import generation_repair_v2
from src.repairs.generation_repair_v3 import generation_repair_v3
from eval.metrics import compute_em, compute_f1


def normalize_contexts(contexts):
    out = []
    for c in contexts:
        if isinstance(c, dict):
            out.append(c.get("contents") or c.get("text") or str(c))
        else:
            out.append(str(c))
    return out


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


class DenseRetriever:
    def retrieve(self, query, top_k=10):
        return normalize_contexts(retrieve_dense(query, top_k=top_k))


class SimpleGenerator:
    def generate(self, question, contexts):
        return generate(question, contexts)


def load_samples(diagnosis_path):
    raw = json.loads(Path(diagnosis_path).read_text())
    rows = raw["results"] if isinstance(raw, dict) and "results" in raw else raw
    if not isinstance(rows, list):
        sys.exit(f"[run_dense_repair] unexpected diagnosis structure in "
                 f"{diagnosis_path}")
    return rows


def load_done(path):
    p = Path(path)
    if not p.exists():
        return []
    try:
        data = json.load(open(p))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save(path, results):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def run(diagnosis_path, output_path, n_samples=None, resume=True):
    samples = load_samples(diagnosis_path)
    if n_samples:
        samples = samples[:n_samples]

    results = load_done(output_path) if resume else []
    done_ids = {str(r["id"]) for r in results}
    if results:
        print(f"Resume: {len(results)}/{len(samples)} already done", flush=True)

    retriever = DenseRetriever()
    generator = SimpleGenerator()

    for i, sample in enumerate(samples):
        sid = str(sample.get("id", i))
        if sid in done_ids:
            continue

        q = sample["question"]
        gold = get_gold_answers(sample)
        rank_bucket = sample.get("rank_bucket", "unknown")
        qtype = sample.get("question_type", "other")
        best_oracle = sample.get("best_oracle_stage", "baseline")

        print(f"[{i + 1}/{len(samples)}] {q[:60]}", flush=True)

        # baseline (dense retrieval)
        base_ctx = retriever.retrieve(q, top_k=10)
        base_answer = generate(q, base_ctx)

        # retrieval repair (dense retriever passed through)
        rr = retrieval_repair(q, retriever, generator, top_k=10)
        rr_answer = rr["repaired_answer"]

        # generation repair V1 (uses baseline's dense context, top-5)
        context_str = "\n\n".join(base_ctx[:5])
        v1_answer = generation_repair(q, context_str, base_answer)

        # generation repair V2 / V3 (same dense context)
        v2_answer = generation_repair_v2(q, context_str, base_answer)
        v3_answer = generation_repair_v3(q, context_str, base_answer,
                                         question_type=qtype)

        result = {
            "id": sid,
            "question": q,
            "golden_answers": gold,
            "rank_bucket": rank_bucket,
            "question_type": qtype,
            "best_oracle_stage": best_oracle,
            "baseline": {
                "pred": base_answer,
                "em": compute_em(base_answer, gold),
                "f1": compute_f1(base_answer, gold),
            },
            "retrieval_repair": {
                "rewritten_query": rr.get("rewritten_query"),
                "pred": rr_answer,
                "em": compute_em(rr_answer, gold),
                "f1": compute_f1(rr_answer, gold),
            },
            "generation_repair": {
                "pred": v1_answer,
                "em": compute_em(v1_answer, gold),
                "f1": compute_f1(v1_answer, gold),
            },
            "generation_repair_v2": {
                "pred": v2_answer,
                "em": compute_em(v2_answer, gold),
                "f1": compute_f1(v2_answer, gold),
            },
            "generation_repair_v3": {
                "pred": v3_answer,
                "em": compute_em(v3_answer, gold),
                "f1": compute_f1(v3_answer, gold),
            },
        }
        results.append(result)

        if len(results) % 50 == 0:
            save(output_path, results)
            print(f"  checkpoint @ {len(results)}", flush=True)

    save(output_path, results)
    summarize(results)


def summarize(results):
    n = len(results)
    actions = ["baseline", "retrieval_repair", "generation_repair",
              "generation_repair_v2", "generation_repair_v3"]
    print(f"\n=== Dense retriever, uniform application (n={n}) ===")
    for a in actions:
        if not all(a in r for r in results):
            continue
        em = sum(r[a]["em"] for r in results) / n
        f1 = sum(r[a]["f1"] for r in results) / n
        helped = sum(1 for r in results if r[a]["em"] > r["baseline"]["em"])
        harmed = sum(1 for r in results if r[a]["em"] < r["baseline"]["em"])
        print(f"{a:24s} EM {em:.4f}  F1 {f1:.4f}  "
              f"(+{helped} / -{harmed} vs baseline)")

    avail = [a for a in actions if all(a in r for r in results)]
    orc_em = sum(max(r[a]["em"] for a in avail) for r in results) / n
    best_uni = max(sum(r[a]["em"] for r in results) / n for a in avail)
    print(f"\nOracle routing over {{{', '.join(avail)}}}")
    print(f"  EM {orc_em:.4f}  (gap vs best uniform: {orc_em - best_uni:+.4f})")

    print("\n(compare these numbers directly against the BM25 log's "
          "'Uniform application' summary to check retriever generality)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--diagnosis", required=True,
                    help="CIKM diagnosis json, e.g. outputs/cikm/nq_1500/results.json")
    ap.add_argument("--output", required=True)
    ap.add_argument("--n_samples", type=int, default=None)
    ap.add_argument("--no_resume", action="store_true")
    args = ap.parse_args()
    run(args.diagnosis, args.output, args.n_samples, resume=not args.no_resume)