"""Learning curve and repair-cost table for the fixed router configuration.

Learning curve: the same stratified 5-fold partitions as oof_two_stage (same seed), so
every training fraction is evaluated on identical held-out questions. Only the training
side of each fold is subsampled (stratified on the gate label); h_w and tau stay at the
paper's operating point. At fraction 1.0 this reproduces the fixed-config router.

Cost table (fraction 1.0): how often each repair is executed and the extra LLM calls
per question, compared with the strongest uniform repair. Call counts per action are
--calls (default: retrieval repair = rewrite + answer = 2, generation repairs = 1).

Usage (from the repo root):
    python scripts/arr/learning_curve.py --features <router_features.csv> --log <repair_log.json> \
        --out_dir outputs/arr/learning_curve/bm25_tqa --fixed_thr 0.3
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_router as trmod  # noqa: E402
from router_common import load_action_scores  # noqa: E402

DEFAULT_CALLS = "retrieval_repair=2,generation_repair=1,generation_repair_v2=1,generation_repair_v3=1"


def fast_bootstrap_ci(diffs, n_boot, seed=12345, chunk=500):
    """Percentile CI of the mean, vectorized in chunks (same estimator as reviewer_defense.bootstrap_ci)."""
    diffs = np.asarray(diffs, dtype=float)
    rng = np.random.RandomState(seed)
    means = np.concatenate([diffs[rng.randint(0, len(diffs), size=(min(chunk, n_boot - i), len(diffs)))].mean(axis=1)
                            for i in range(0, n_boot, chunk)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(diffs.mean()), float(lo), float(hi)


def fold_decisions(df, id_col, model_kind, hw, thr, seed, frac):
    """Router decisions on every question, trained on `frac` of each outer-training fold."""
    X = df[trmod.NUMERIC + trmod.CATEG]
    y_gate = (df["best_action"] != "baseline").astype(int).values
    qids = df[id_col].values
    decisions = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)  # identical to oof_two_stage
    for fold, (tr, te) in enumerate(skf.split(X, y_gate)):
        if frac < 1.0:
            tr, _ = train_test_split(tr, train_size=frac, stratify=y_gate[tr], random_state=seed * 100 + fold)
        sub = df.iloc[tr]
        gate = trmod.make_model(model_kind)
        gate.fit(X.iloc[tr], y_gate[tr], clf__sample_weight=trmod.gate_sample_weights(sub, hw))
        sel_mask = sub["best_action"].isin(trmod.SELECTOR_ACTIONS).values
        sel = trmod.make_model(model_kind)
        sel.fit(X.iloc[tr][sel_mask], sub["best_action"].values[sel_mask])
        gate_p = gate.predict_proba(X.iloc[te])[:, 1]
        sel_p = sel.predict(X.iloc[te])
        for q, p, a in zip(qids[te], gate_p, sel_p):
            decisions[q] = a if p >= thr else "baseline"
    return decisions


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model", default="logreg", choices=["logreg", "gbt"])
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--fixed_hw", type=float, default=5.0)
    ap.add_argument("--fixed_thr", type=float, required=True, help="paper operating threshold (.4 BM25 NQ, .3 otherwise)")
    ap.add_argument("--fractions", default="0.1,0.25,0.5,0.75,1.0")
    ap.add_argument("--calls", default=DEFAULT_CALLS, help="LLM calls per execution of each repair")
    ap.add_argument("--bootstrap", type=int, default=20000)
    args = ap.parse_args()

    df, id_col = trmod.load_features(args.features)
    trmod.configure_features(df, use_diag=False)
    scores = load_action_scores(args.log)
    df = df[df[id_col].astype(str).isin(scores)].reset_index(drop=True)
    qids = df[id_col].astype(str).tolist()
    n = len(qids)
    seeds = [int(x) for x in args.seeds.split(",")]
    fractions = [float(x) for x in args.fractions.split(",")]
    calls = {k: float(v) for k, v in (item.split("=") for item in args.calls.split(","))}
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    uniform = {a: trmod.routed_scores({q: a for q in qids}, scores) for a in sorted({a for q in qids for a in scores[q]})}
    best = max((a for a in uniform if a != "baseline"), key=lambda a: uniform[a]["em"])
    uniform_vec = np.array([scores[q][best]["em"] for q in qids])

    curve_rows, full_decisions = [], []
    for frac in fractions:
        per_seed, vecs = [], []
        for seed in seeds:
            dec = fold_decisions(df, id_col, args.model, args.fixed_hw, args.fixed_thr, seed, frac)
            per_seed.append(trmod.routed_scores(dec, scores))
            vecs.append(np.array([scores[q][dec[q] if dec[q] in scores[q] else "baseline"]["em"] for q in qids]))
            if frac == 1.0:
                full_decisions.append(dec)
        d, lo, hi = fast_bootstrap_ci(np.mean(vecs, axis=0) - uniform_vec, args.bootstrap)
        em = np.array([r["em"] for r in per_seed])
        curve_rows.append({"train_fraction": frac, "train_size_per_fold": int(round(0.8 * n * frac)),
                           "em": em.mean(), "em_sd": em.std(ddof=1) if len(em) > 1 else 0.0,
                           "gain_vs_uniform_pp": 100 * d, "ci_lo_pp": 100 * lo, "ci_hi_pp": 100 * hi,
                           **{k: float(np.mean([r[k] for r in per_seed])) for k in ["repair_rate", "fixed", "harmed", "net_gain"]}})
        print(f"[learning_curve] fraction {frac:g}: EM {em.mean():.4f} ({100 * d:+.2f} pp vs {best})", flush=True)
    curve = pd.DataFrame(curve_rows)
    curve.to_csv(out / "learning_curve.csv", index=False)

    # Cost table at fraction 1.0, averaged over seeds.
    cost_rows = []
    for dec in full_decisions:
        acts = pd.Series([dec[q] for q in qids])
        repairs = acts[acts != "baseline"]
        r = trmod.routed_scores(dec, scores)
        cost_rows.append({"repair_rate": len(repairs) / n,
                          "extra_llm_calls_per_query": repairs.map(lambda a: calls.get(a, 1.0)).sum() / n,
                          "gain_per_100_repairs": r["gain_per_100_repairs"], "over_repair_rate": r["over_repair_rate"],
                          **{f"share_{a}": float((repairs == a).sum() / n) for a in trmod.SELECTOR_ACTIONS}})
    cost = pd.DataFrame(cost_rows).mean().to_frame("router").T
    cost.insert(0, "policy", "router")
    u = uniform[best]
    cost = pd.concat([cost, pd.DataFrame([{"policy": f"uniform:{best}", "repair_rate": 1.0,
                                           "extra_llm_calls_per_query": calls.get(best, 1.0),
                                           "gain_per_100_repairs": u["gain_per_100_repairs"],
                                           "over_repair_rate": u["over_repair_rate"]}])], ignore_index=True)
    cost.to_csv(out / "cost.csv", index=False)

    (out / "summary.json").write_text(json.dumps({
        "n": n, "features": args.features, "log": args.log, "model": args.model, "seeds": seeds,
        "fixed_hw": args.fixed_hw, "fixed_thr": args.fixed_thr, "best_uniform_action": best,
        "best_uniform_em": u["em"], "calls_per_repair": calls,
        "notes": "learning curve subsamples only the training side of identical 5-fold partitions; "
                 "gain CI = query-level paired bootstrap of seed-averaged correctness vs best uniform"}, indent=2))

    with pd.option_context("display.width", None, "display.max_columns", None, "display.float_format", "{:.4f}".format):
        print(f"\nn={n}  best uniform: {best} EM {u['em']:.4f}  h_w={args.fixed_hw:g} tau={args.fixed_thr:g}")
        print("\n== learning curve (same held-out folds; fraction of each training fold) ==")
        print(curve.to_string(index=False))
        print("\n== repair cost (fraction 1.0, mean over seeds) ==")
        print(cost.to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
