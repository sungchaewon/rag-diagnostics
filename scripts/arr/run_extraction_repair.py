import json
import argparse
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.retrieval.bm25_baseline import retrieve_bm25
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


def build_context(question, top_k=10, use_top=5):
    """Reproduce the original generator context (top-5 of BM25 top-10)."""
    ctxs = normalize_contexts(retrieve_bm25(question, top_k=top_k))
    return "\n\n".join(ctxs[:use_top])


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


def run(log_path, output_path, n_samples=None, resume=True):
    with open(log_path) as f:
        records = json.load(f)
    if n_samples:
        records = records[:n_samples]

    results = load_done(output_path) if resume else []
    done_ids = {r["id"] for r in results}
    start = len(results)
    if start:
        print(f"Resume: {start}/{len(records)} already done", flush=True)

    for i, rec in enumerate(records):
        if rec["id"] in done_ids:
            continue
        q = rec["question"]
        gold = rec["golden_answers"]
        base_pred = rec["baseline"]["pred"]
        qtype = rec.get("question_type")

        print(f"[{i + 1}/{len(records)}] {q[:60]}", flush=True)

        ctx = build_context(q)

        v2 = generation_repair_v2(q, ctx, base_pred)
        v3 = generation_repair_v3(q, ctx, base_pred, question_type=qtype)

        rec = dict(rec)  # keep original blocks untouched
        rec["generation_repair_v2"] = {
            "pred": v2, "em": compute_em(v2, gold), "f1": compute_f1(v2, gold)
        }
        rec["generation_repair_v3"] = {
            "pred": v3, "em": compute_em(v3, gold), "f1": compute_f1(v3, gold)
        }
        results.append(rec)

        if len(results) % 50 == 0:
            save(output_path, results)
            print(f"  checkpoint @ {len(results)}", flush=True)

    save(output_path, results)
    summarize(results)


ACTIONS = [
    "baseline",
    "retrieval_repair",
    "generation_repair",      # V1 (from the original log)
    "generation_repair_v2",
    "generation_repair_v3",
]


def summarize(results):
    n = len(results)
    print(f"\n=== Uniform application (n={n}) ===")
    for a in ACTIONS:
        if not all(a in r for r in results):
            continue
        em = sum(r[a]["em"] for r in results) / n
        f1 = sum(r[a]["f1"] for r in results) / n
        helped = sum(1 for r in results if r[a]["em"] > r["baseline"]["em"])
        harmed = sum(1 for r in results if r[a]["em"] < r["baseline"]["em"])
        print(f"{a:24s} EM {em:.4f}  F1 {f1:.4f}  "
              f"(+{helped} / -{harmed} vs baseline)")

    avail = [a for a in ACTIONS if all(a in r for r in results)]
    orc_em = sum(max(r[a]["em"] for a in avail) for r in results) / n
    orc_f1 = sum(max(r[a]["f1"] for a in avail) for r in results) / n
    best_uni_em = max(sum(r[a]["em"] for r in results) / n for a in avail)
    print(f"\nOracle routing over {{{', '.join(avail)}}}")
    print(f"  EM {orc_em:.4f}  F1 {orc_f1:.4f}  "
          f"(gap vs best uniform: {orc_em - best_uni_em:+.4f} EM)")

    # unique fixes: baseline wrong, only this action fixes it
    print("\nUnique fixes among failed-baseline queries:")
    repair_actions = [a for a in avail if a != "baseline"]
    for a in repair_actions:
        others = [b for b in repair_actions if b != a]
        uniq = sum(
            1 for r in results
            if r["baseline"]["em"] == 0 and r[a]["em"] == 1
            and all(r[b]["em"] == 0 for b in others)
        )
        print(f"  {a:24s} {uniq}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True,
                    help="existing *_repair_1500_v4.json log")
    ap.add_argument("--output", required=True)
    ap.add_argument("--n_samples", type=int, default=None)
    ap.add_argument("--no_resume", action="store_true")
    args = ap.parse_args()
    run(args.log, args.output, args.n_samples, resume=not args.no_resume)
