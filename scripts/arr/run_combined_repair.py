import json
import argparse
import inspect
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.retrieval.bm25_baseline import retrieve_bm25
from src.repairs.generation_repair_v3 import generation_repair_v3
from src.repairs import retrieval_repair as rr_mod
from eval.metrics import compute_em, compute_f1

ACTION = "combined_repair"


def normalize_contexts(contexts):
    out = []
    for c in contexts:
        if isinstance(c, dict):
            out.append(c.get("contents") or c.get("text") or str(c))
        else:
            out.append(str(c))
    return out


def build_context(question, top_k=10, use_top=5):
    ctxs = normalize_contexts(retrieve_bm25(question, top_k=top_k))
    return "\n\n".join(ctxs[:use_top])


def repaired_context(question, use_top=5):
    try:
        rewrite = None
        for name in ("rewrite_query", "expand_query", "make_queries"):
            if hasattr(rr_mod, name):
                rewrite = getattr(rr_mod, name)
                break

        if rewrite is not None:
            rq = rewrite(question)
            queries = rq if isinstance(rq, (list, tuple)) else [rq]
        else:
            queries = []

        base = normalize_contexts(retrieve_bm25(question, top_k=10))
        pools = [base]
        for q in queries:
            if q and q != question:
                pools.append(normalize_contexts(retrieve_bm25(q, top_k=10)))

        if len(pools) > 1 and hasattr(rr_mod, "mix_ctx"):
            sig = inspect.signature(rr_mod.mix_ctx)
            if len(sig.parameters) >= 3:
                mixed = rr_mod.mix_ctx(pools[0], pools[1], use_top)
            else:
                mixed = rr_mod.mix_ctx(pools[0], pools[1])
            mixed = normalize_contexts(mixed)
        else:
            seen, mixed = set(), []
            for pool in pools:
                for c in pool:
                    if c not in seen:
                        seen.add(c)
                        mixed.append(c)

        return "\n\n".join(mixed[:use_top]), True
    except Exception as e:
        print(f"    [warn] context reconstruction failed ({e}), "
              f"using plain BM25 context", flush=True)
        return build_context(question, use_top=use_top), False


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
    if isinstance(records, dict):
        records = records.get("results", records)
    if n_samples:
        records = records[:n_samples]

    results = load_done(output_path) if resume else []
    done_ids = {r["id"] for r in results}
    if results:
        print(f"Resume: {len(results)}/{len(records)} already done", flush=True)

    fallbacks = 0
    for i, rec in enumerate(records):
        if rec["id"] in done_ids:
            continue
        q = rec["question"]
        gold = rec["golden_answers"]
        base_pred = rec["baseline"]["pred"]
        qtype = rec.get("question_type")

        print(f"[{i + 1}/{len(records)}] {q[:60]}", flush=True)

        ctx, ok = repaired_context(q)
        if not ok:
            fallbacks += 1

        pred = generation_repair_v3(q, ctx, base_pred, question_type=qtype)

        rec = dict(rec)
        rec[ACTION] = {
            "pred": pred,
            "em": compute_em(pred, gold),
            "f1": compute_f1(pred, gold),
        }
        results.append(rec)

        if len(results) % 50 == 0:
            save(output_path, results)
            print(f"  checkpoint @ {len(results)}", flush=True)

    save(output_path, results)
    if fallbacks:
        print(f"\n[warn] {fallbacks} queries used the plain BM25 context; "
              f"check retrieval_repair internals before trusting these",
              flush=True)
    summarize(results)


def summarize(results):
    """Does combined repair cover anything the single turns miss?"""
    n = len(results)
    singles = ["retrieval_repair", "generation_repair_v3"]
    avail = [a for a in singles if all(a in r for r in results)]

    print(f"\n=== combined repair (n={n}) ===")
    em = sum(r[ACTION]["em"] for r in results) / n
    f1 = sum(r[ACTION]["f1"] for r in results) / n
    helped = sum(1 for r in results
                 if r[ACTION]["em"] > r["baseline"]["em"])
    harmed = sum(1 for r in results
                 if r[ACTION]["em"] < r["baseline"]["em"])
    print(f"EM {em:.4f}  F1 {f1:.4f}  (+{helped} / -{harmed} vs baseline)")

    # the decisive number: fixes no single turn achieves
    uniq = sum(
        1 for r in results
        if r["baseline"]["em"] == 0 and r[ACTION]["em"] == 1
        and all(r[a]["em"] == 0 for a in avail)
    )
    print(f"\nunique fixes vs {avail}: {uniq}")
    print("  (>0 means multi-turn covers cases one turn cannot,")
    print("   0 means single-turn suffices and this stays future work)")

    # oracle with and without the new action
    def oracle(actions):
        return sum(max(r[a]["em"] for a in actions if a in r)
                   for r in results) / n

    all_acts = ["baseline", "retrieval_repair", "generation_repair",
                "generation_repair_v2", "generation_repair_v3"]
    have = [a for a in all_acts if all(a in r for r in results)]
    print(f"\noracle routing without combined: {oracle(have):.4f}")
    print(f"oracle routing with combined:    {oracle(have + [ACTION]):.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n_samples", type=int, default=None)
    ap.add_argument("--no_resume", action="store_true")
    args = ap.parse_args()
    run(args.log, args.output, args.n_samples, resume=not args.no_resume)