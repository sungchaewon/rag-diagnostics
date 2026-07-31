import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from router_common import load_action_scores

THRESHOLDS = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9]

FEATURE_PROFILES = {
    # pre-retrieval: question text only, nothing about retrieval or the
    # generated answer is visible yet
    "adaptive_rag": {
        "numeric": ["is_numeric_q"],
        "categ": ["qtype"],
    },
    # post-retrieval: retrieval-quality signals only, nothing about the
    # generated answer
    "crag": {
        "numeric": ["bm25_top1", "bm25_margin", "bm25_mean10"],
        "categ": [],
        "optional_numeric": ["bm25_hit", "gold_rank_capped"],
        "optional_categ": ["rank_bucket"],
    },
}


def resolve_features(df, mode):
    prof = FEATURE_PROFILES[mode]
    numeric = [c for c in prof["numeric"] if c in df.columns]
    categ = [c for c in prof["categ"] if c in df.columns]
    missing_core = [c for c in prof["numeric"] + prof["categ"]
                   if c not in df.columns]
    if missing_core:
        sys.exit(f"[baselines] {mode} needs columns {missing_core}, "
                 f"missing from {list(df.columns)}")
    for c in prof.get("optional_numeric", []):
        if c in df.columns:
            numeric.append(c)
    for c in prof.get("optional_categ", []):
        if c in df.columns:
            categ.append(c)
    return numeric, categ


def make_model(numeric, categ):
    transformers = []
    if numeric:
        transformers.append(("num", StandardScaler(), numeric))
    if categ:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categ))
    pre = ColumnTransformer(transformers)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    return Pipeline([("pre", pre), ("clf", clf)])


def binary_label(scores, qid):
    """1 if retrieval_repair beats baseline EM for this query, else 0."""
    v = scores[qid]
    return int(v.get("retrieval_repair", {}).get("em", 0) > v["baseline"]["em"])


def routed_scores(decisions, scores):
    em = f1 = 0.0
    harmed = fixed = repairs = 0
    n = 0
    for qid, use_repair in decisions.items():
        v = scores.get(qid)
        if v is None:
            continue
        act = "retrieval_repair" if use_repair else "baseline"
        em += v[act]["em"]; f1 += v[act]["f1"]; n += 1
        if use_repair:
            repairs += 1
            if v[act]["em"] > v["baseline"]["em"]:
                fixed += 1
            elif v[act]["em"] < v["baseline"]["em"]:
                harmed += 1
    net = fixed - harmed
    return {"em": em / n, "f1": f1 / n, "n": n, "repair_rate": repairs / n,
            "fixed": fixed, "harmed": harmed, "net_gain": net,
            "gain_per_100_repairs": 100.0 * net / repairs if repairs else 0.0,
            "over_repair_rate": (repairs - fixed) / repairs if repairs else 0.0}


def oof_proba(df, id_col, scores, numeric, categ, seed=0):
    X = df[numeric + categ]
    y = np.array([binary_label(scores, q) for q in df[id_col]])
    proba = pd.Series(index=df.index, dtype=float)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        m = make_model(numeric, categ)
        m.fit(X.iloc[tr], y[tr])
        proba.iloc[te] = m.predict_proba(X.iloc[te])[:, 1]
    return dict(zip(df[id_col].values, proba.values))


def fit_full(df, id_col, scores, numeric, categ, seed=None):
    if seed is not None:
        df = df.sample(n=len(df), replace=True,
                       random_state=seed).reset_index(drop=True)
    X = df[numeric + categ]
    y = np.array([binary_label(scores, q) for q in df[id_col]])
    m = make_model(numeric, categ)
    m.fit(X, y)
    return m


def decide(proba, thr):
    return {qid: (p >= thr) for qid, p in proba.items()}


def uniform_rows(scores):
    rows = []
    for a, label in [("baseline", "uniform:baseline"),
                     ("retrieval_repair", "uniform:retrieval_repair")]:
        dec = {qid: (a == "retrieval_repair") for qid in scores}
        rows.append((label, routed_scores(dec, scores)))
    oracle = {qid: (v["retrieval_repair"]["em"] > v["baseline"]["em"]
                    if "retrieval_repair" in v else False)
              for qid, v in scores.items()}
    rows.append(("oracle (2-way)", routed_scores(oracle, scores)))
    return rows


def report(title, rows):
    print(f"\n=== {title} ===")
    print(f"{'setting':<28}{'EM':>8}{'F1':>8}{'rep%':>7}"
          f"{'fixed':>7}{'harmed':>7}{'net':>6}{'g/100r':>8}{'overR%':>8}")
    for name, r in rows:
        print(f"{name:<28}{r['em']:>8.4f}{r['f1']:>8.4f}"
              f"{r['repair_rate']:>7.1%}{r['fixed']:>7}{r['harmed']:>7}"
              f"{r['net_gain']:>6}{r['gain_per_100_repairs']:>8.1f}"
              f"{r['over_repair_rate']:>8.1%}")


def _mean_std(vals):
    arr = np.array(vals, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def multi_seed(df, id_col, scores, numeric, categ, thr, seeds,
               edf=None, eid=None, escores=None):
    in_em, in_harm = [], []
    tr_em, tr_harm = [], []
    for s in seeds:
        proba = oof_proba(df, id_col, scores, numeric, categ, seed=s)
        r = routed_scores(decide(proba, thr), scores)
        in_em.append(r["em"]); in_harm.append(r["harmed"])
        if edf is not None:
            m = fit_full(df, id_col, scores, numeric, categ, seed=s)
            X = edf[numeric + categ]
            qids = edf[eid].values
            p = dict(zip(qids, m.predict_proba(X)[:, 1]))
            tr = routed_scores(decide(p, thr), escores)
            tr_em.append(tr["em"]); tr_harm.append(tr["harmed"])
    out = {"in_em": _mean_std(in_em), "in_harm": _mean_std(in_harm),
           "in_em_raw": in_em}
    if edf is not None:
        out.update({"tr_em": _mean_std(tr_em), "tr_harm": _mean_std(tr_harm),
                    "tr_em_raw": tr_em})
    return out


def verdict(delta, std):
    if std == 0:
        return "N/A (zero variance)"
    sigma = delta / std
    return ("SIGNIFICANT" if delta > 2 * std else
            "MARGINAL" if delta > std else "NOT DISTINGUISHABLE"), sigma


def main(args):
    df = pd.read_csv(args.features)
    id_col = next((c for c in ["qid", "id", "query_id", "idx"]
                   if c in df.columns), None)
    if id_col is None:
        df["qid"] = df.index.astype(str)
        id_col = "qid"
    df[id_col] = df[id_col].astype(str)

    numeric, categ = resolve_features(df, args.mode)
    print(f"mode: {args.mode}")
    print(f"features: numeric={numeric}, categorical={categ}")

    scores = load_action_scores(args.log)

    rows = uniform_rows(scores)
    report(f"uniform / oracle baselines ({args.mode}, 2-way action space)",
          rows)
    best_uniform_em = max(v["em"] for k, v in rows
                          if k != "oracle (2-way)")

    proba = oof_proba(df, id_col, scores, numeric, categ)
    sweep = [(f"thr={t}", routed_scores(decide(proba, t), scores))
             for t in THRESHOLDS]
    report(f"{args.mode} classifier, 5-fold OOF, threshold sweep", sweep)
    best = max(sweep, key=lambda t: t[1]["em"])
    print(f"\nbest by routed EM: {best[0]} (EM {best[1]['em']:.4f}, "
          f"harmed {best[1]['harmed']})")

    if args.seeds:
        seeds = [int(x) for x in args.seeds.split(",")]
        edf = eid = escores = None
        tr_best_uniform = None
        if args.eval_features and args.eval_log:
            edf = pd.read_csv(args.eval_features)
            eid = next((c for c in ["qid", "id", "query_id", "idx"]
                       if c in edf.columns), None)
            edf[eid] = edf[eid].astype(str)
            escores = load_action_scores(args.eval_log)
            erows = uniform_rows(escores)
            report(f"uniform / oracle baselines (eval, {args.mode})", erows)
            tr_best_uniform = max(v["em"] for k, v in erows
                                  if k != "oracle (2-way)")

        res = multi_seed(df, id_col, scores, numeric, categ, args.fix_thr,
                         seeds, edf=edf, eid=eid, escores=escores)
        m, s = res["in_em"]
        print(f"\n=== multi-seed ({len(seeds)} seeds, thr={args.fix_thr}) ===")
        print(f"in-dataset EM: {m:.4f} +/- {s:.4f}")
        v, sigma = verdict(m - best_uniform_em, s) if s > 0 else \
            (verdict(m - best_uniform_em, s), None)
        print(f"  vs best uniform {best_uniform_em:.4f}: "
              f"delta {m - best_uniform_em:+.4f} -> {v}")
        print(f"  per-seed EM: {[round(x, 4) for x in res['in_em_raw']]}")
        if "tr_em" in res:
            tm, ts = res["tr_em"]
            print(f"transfer EM: {tm:.4f} +/- {ts:.4f}")
            v2, sigma2 = verdict(tm - tr_best_uniform, ts) if ts > 0 else \
                (verdict(tm - tr_best_uniform, ts), None)
            print(f"  vs eval best uniform {tr_best_uniform:.4f}: "
                  f"delta {tm - tr_best_uniform:+.4f} -> {v2}")
            print(f"  per-seed EM: {[round(x, 4) for x in res['tr_em_raw']]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["adaptive_rag", "crag"])
    ap.add_argument("--features", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--eval_features")
    ap.add_argument("--eval_log")
    ap.add_argument("--seeds", default="",
                    help="comma list, e.g. 0,1,2,3,4 (empty = skip)")
    ap.add_argument("--fix_thr", type=float, default=0.3)
    args = ap.parse_args()
    main(args)