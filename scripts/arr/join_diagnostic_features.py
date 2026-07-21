import argparse
import json
import sys
from pathlib import Path

import pandas as pd

DIAG_NUMERIC = ["bm25_hit", "bm25_gold_rank"]
DIAG_CATEG = ["rank_bucket", "split_retrieval", "split_answer_type"]
MISS_RANK_CAP = 11  # top-10 retrieval, so a miss is "rank 11+"


def load_diagnosis(path):
    raw = json.loads(Path(path).read_text())
    rows = raw["results"] if isinstance(raw, dict) and "results" in raw else raw
    if not isinstance(rows, list):
        sys.exit(f"[join_diag] unexpected diagnosis structure in {path}")

    recs = []
    for r in rows:
        rec = {"__id": str(r["id"])}
        for k in DIAG_NUMERIC + DIAG_CATEG:
            v = r.get(k)
            if k == "bm25_hit":
                v = int(bool(v))
            rec[k] = v
        recs.append(rec)
    df = pd.DataFrame(recs)
    df["gold_rank_capped"] = df["bm25_gold_rank"].apply(
        lambda x: MISS_RANK_CAP if (x is None or x < 0) else int(x))
    return df


def main(args):
    feat = pd.read_csv(args.features)
    id_col = next((c for c in ["qid", "id", "query_id", "idx"]
                   if c in feat.columns), None)
    if id_col is None:
        sys.exit(f"[join_diag] no id column found in {args.features}; "
                 f"columns={list(feat.columns)}")
    feat["__id"] = feat[id_col].astype(str)

    diag = load_diagnosis(args.diagnosis)

    overlap = len(set(feat["__id"]) & set(diag["__id"]))
    print(f"features: {len(feat)}, diagnosis: {len(diag)}, overlap: {overlap}")
    if overlap < len(feat):
        print(f"WARNING: {len(feat) - overlap} feature rows have no "
              f"diagnostic match and will get NaN")

    merged = feat.merge(diag, on="__id", how="left").drop(columns=["__id"])

    # fill any unmatched rows conservatively (treated as miss)
    merged["bm25_hit"] = merged["bm25_hit"].fillna(0).astype(int)
    merged["bm25_gold_rank"] = merged["bm25_gold_rank"].fillna(-1).astype(int)
    merged["gold_rank_capped"] = (merged["gold_rank_capped"]
                                  .fillna(MISS_RANK_CAP).astype(int))
    for c in DIAG_CATEG:
        merged[c] = merged[c].fillna("unknown")

    merged.to_csv(args.output, index=False)
    print(f"wrote {args.output}")

    # sanity: diagnostic signal vs best_action
    if "best_action" in merged.columns:
        print("\nbest_action by split_retrieval:")
        ct = pd.crosstab(merged["split_retrieval"], merged["best_action"])
        print(ct.to_string())
        print("\nbest_action by rank_bucket:")
        ct2 = pd.crosstab(merged["rank_bucket"], merged["best_action"])
        print(ct2.to_string())
        # repair rate per bucket = how much signal the gate can get
        print("\nrepair-needed rate by rank_bucket:")
        merged["_rep"] = (merged["best_action"] != "baseline").astype(int)
        print(merged.groupby("rank_bucket")["_rep"]
              .agg(["mean", "count"]).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--diagnosis", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    main(args)