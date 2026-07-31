import re
import csv
import json
import argparse
from pathlib import Path
from collections import Counter
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

UNK = {
    "", "unknown", "i don't know", "i do not know",
    "not enough information", "not enough evidence",
    "cannot be determined", "unanswerable", "none", "n/a",
}
ACTION_PRIORITY = [
    "baseline",
    "retrieval_repair",
    "generation_repair",
    "generation_repair_v2",
    "generation_repair_v3",
]


def norm(x):
    x = str(x or "").strip().lower()
    x = re.sub(r"^(final answer|answer)\s*:\s*", "", x)
    x = x.strip(" \n\t\"'`.")
    return re.sub(r"\s+", " ", x)


def tokens(x):
    return re.findall(r"[a-z0-9]+", norm(x))


def token_recall(ans, ctx_tokens):
    a = tokens(ans)
    if not a:
        return 0.0
    covered = sum(min(n, ctx_tokens[t]) for t, n in Counter(a).items())
    return covered / len(a)


def best_action(rec, actions):
    scored = [(rec[a]["em"], rec[a]["f1"], -ACTION_PRIORITY.index(a), a)
              for a in actions]
    return max(scored)[3]


def main(log_path, output_path, use_dense=True):
    with open(log_path) as f:
        records = json.load(f)
    if isinstance(records, dict) and "results" in records:
        records = records["results"]

    actions = [a for a in ACTION_PRIORITY if all(a in r for r in records)]
    print(f"n={len(records)}, actions={actions}")

    if use_dense:
        import src.retrieval.dense_baseline as db
        db._load_corpus()
        model = db._load_model()

    rows = []
    for i, rec in enumerate(records):
        q = rec["question"]
        pred = rec["baseline"]["pred"]
        row = {
            "id": rec["id"],
            "qtype": rec.get("question_type", "other"),
            "is_unknown": int(norm(pred) in UNK),
            "answer_len": len(str(pred).split()),
            "is_numeric_q": int(bool(re.search(r"\bhow (many|much)\b",
                                               q.lower()))),
            "ans_has_digit": int(bool(re.search(r"\d", str(pred)))),
        }
        if use_dense:
            q_emb = model.encode([f"query: {q}"], normalize_embeddings=True,
                                 convert_to_numpy=True)[0]
            scores = db._embeddings @ q_emb  # cosine sim (both normalized)
            order = scores.argsort()[::-1]
            top10 = scores[order[:10]]
            row["bm25_top1"] = round(float(top10[0]), 4)
            row["bm25_margin"] = round(float(top10[0] - top10[4]), 4) \
                if len(top10) >= 5 else 0.0
            row["bm25_mean10"] = round(float(top10.mean()), 4)
            top5_idx = order[:5]
            ctx = " ".join(db._passages[j] for j in top5_idx)
            row["ctx_overlap"] = round(
                token_recall(pred, Counter(tokens(ctx))), 4)
        row["best_action"] = best_action(rec, actions)
        row["baseline_em"] = rec["baseline"]["em"]
        rows.append(row)
        if (i + 1) % 300 == 0:
            print(f"  {i + 1}/{len(records)}", flush=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {output_path}")

    print("\nbest_action distribution:")
    for a, c in Counter(r["best_action"] for r in rows).most_common():
        print(f"  {a:24s} {c}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--no_dense", action="store_true")
    args = ap.parse_args()
    main(args.log, args.output, use_dense=not args.no_dense)