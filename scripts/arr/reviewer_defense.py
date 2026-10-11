import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent
# Prefer running from repo root; otherwise let caller pass --repo_root.


def bootstrap_ci(diffs, n_boot=20000, seed=12345):
    diffs = np.asarray(diffs, dtype=float)
    rng = np.random.RandomState(seed)
    n = len(diffs)
    vals = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        vals[b] = diffs[idx].mean()
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(diffs.mean()), float(lo), float(hi)


def exact_mcnemar(router_y, uniform_y):
    router_y = np.asarray(router_y, dtype=int)
    uniform_y = np.asarray(uniform_y, dtype=int)
    b = int(np.sum((router_y == 1) & (uniform_y == 0)))
    c = int(np.sum((router_y == 0) & (uniform_y == 1)))
    n = b + c
    p = 1.0 if n == 0 else float(binomtest(b, n=n, p=0.5, alternative="two-sided").pvalue)
    return {"router_only_correct": b, "uniform_only_correct": c, "discordant": n, "p": p}


def fit_plain(tr, model_kind, harm_weight, make_model, gate_sample_weights, SELECTOR_ACTIONS, NUMERIC, CATEG):
    X = tr[NUMERIC + CATEG]
    y_gate = (tr["best_action"] != "baseline").astype(int).values
    weights = gate_sample_weights(tr, harm_weight)
    gate = make_model(model_kind)
    gate.fit(X, y_gate, clf__sample_weight=weights)

    sel_mask = tr["best_action"].isin(SELECTOR_ACTIONS)
    sel = make_model(model_kind)
    sel.fit(X.loc[sel_mask, :], tr.loc[sel_mask, "best_action"])
    return gate, sel


def nested_predictions(df, id_col, scores, model_kind, seed, harm_weights, thresholds,
                       make_model, gate_sample_weights, routed_scores,
                       two_stage_decide, oof_two_stage,
                       SELECTOR_ACTIONS, NUMERIC, CATEG):
    """Outer 5-fold test; (h_w,tau) chosen only within outer-train via inner OOF."""
    X = df[NUMERIC + CATEG]
    y_gate = (df["best_action"] != "baseline").astype(int).values
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    decisions = {}
    selected = []

    for fold, (tr_idx, te_idx) in enumerate(outer.split(X, y_gate)):
        tr = df.iloc[tr_idx].reset_index(drop=True)
        te = df.iloc[te_idx]
        tr_id = id_col

        best = None
        # Inner OOF selection on outer-training partition only.
        for hw in harm_weights:
            gp, sp = oof_two_stage(tr, tr_id, model_kind, harm_weight=hw, seed=seed + 1000 + fold)
            for thr in thresholds:
                dec = two_stage_decide(gp, sp, thr)
                r = routed_scores(dec, scores)
                key = (r["em"], -r["harmed"], -r["repair_rate"], -hw, -thr)
                if best is None or key > best[0]:
                    best = (key, hw, thr)

        _, hw_star, thr_star = best
        gate, sel = fit_plain(tr, model_kind, hw_star, make_model, gate_sample_weights,
                              SELECTOR_ACTIONS, NUMERIC, CATEG)
        Xte = te[NUMERIC + CATEG]
        qids = te[id_col].astype(str).values
        gp = gate.predict_proba(Xte)[:, 1]
        sp = sel.predict(Xte)
        for qid, p, act in zip(qids, gp, sp):
            decisions[qid] = act if p >= thr_star else "baseline"
        selected.append({"fold": fold, "harm_weight": hw_star, "threshold": thr_star})

    return decisions, selected


def per_query_correct(decisions, scores, qids):
    return np.array([scores[q][decisions.get(q, "baseline")]["em"] for q in qids], dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", default=".")
    ap.add_argument("--features", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model", default="logreg", choices=["logreg", "gbt"])
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--harm_weights", default="1,3,5,10")
    ap.add_argument("--thresholds", default="0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.7,0.8,0.9")
    ap.add_argument("--fixed_hw", type=float, default=5.0)
    ap.add_argument("--fixed_thr", type=float, default=0.4)
    ap.add_argument("--bootstrap", type=int, default=20000)
    ap.add_argument("--random_trials", type=int, default=200)
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    sys.path.insert(0, str(repo / "scripts" / "arr"))
    sys.path.insert(0, str(repo))

    import train_router as trmod
    from router_common import load_action_scores

    df, id_col = trmod.load_features(args.features)
    trmod.configure_features(df, use_diag=False)
    scores = load_action_scores(args.log)
    df[id_col] = df[id_col].astype(str)
    qids = [q for q in df[id_col].tolist() if q in scores]

    seeds = [int(x) for x in args.seeds.split(",")]
    hws = [float(x) for x in args.harm_weights.split(",")]
    thrs = [float(x) for x in args.thresholds.split(",")]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Best uniform action, query-wise vector.
    actions = sorted({a for q in qids for a in scores[q]})
    uni_metrics = {}
    for a in actions:
        dec = {q: a for q in qids}
        uni_metrics[a] = trmod.routed_scores(dec, scores)
    best_uniform_action = max(uni_metrics, key=lambda a: uni_metrics[a]["em"])
    uniform_vec = np.array([scores[q][best_uniform_action]["em"] for q in qids], dtype=float)

    # Fixed-config OOF predictions across seeds, plus grid and h_w ablation.
    fixed_vecs = []
    mcnemar_rows = []
    fixed_rows = []
    grid_acc = {(hw, t): [] for hw in hws for t in thrs}
    hw_rows = []
    heur_rows = []

    for seed in seeds:
        # Selector predictions are independent of harm weight; get from fixed-hw call.
        gp_fixed, sp_fixed = trmod.oof_two_stage(df, id_col, args.model, args.fixed_hw, seed=seed)
        dec_fixed = trmod.two_stage_decide(gp_fixed, sp_fixed, args.fixed_thr)
        r_fixed = trmod.routed_scores(dec_fixed, scores)
        vec = per_query_correct(dec_fixed, scores, qids)
        fixed_vecs.append(vec)
        mc = exact_mcnemar(vec, uniform_vec)
        mcnemar_rows.append({"seed": seed, **mc})
        fixed_rows.append({"seed": seed, **r_fixed})

        # h_w x threshold sensitivity and h_w ablation.
        for hw in hws:
            gp, sp = trmod.oof_two_stage(df, id_col, args.model, hw, seed=seed)
            for t in thrs:
                rr = trmod.routed_scores(trmod.two_stage_decide(gp, sp, t), scores)
                grid_acc[(hw, t)].append(rr["em"])
            rr_fixed_t = trmod.routed_scores(trmod.two_stage_decide(gp, sp, args.fixed_thr), scores)
            hw_rows.append({"seed": seed, "harm_weight": hw, "threshold": args.fixed_thr, **rr_fixed_t})

        # Simple gate baselines using the same OOF selector predictions.
        # no gate: always execute selector action
        no_gate = {q: sp_fixed[q] for q in qids}
        rn = trmod.routed_scores(no_gate, scores)
        heur_rows.append({"seed": seed, "baseline": "no_gate_selector", **rn})

        # unknown-only gate: repair only when baseline answer was 'unknown' feature.
        isunk = dict(zip(df[id_col].astype(str), df["is_unknown"].astype(int)))
        unk_dec = {q: (sp_fixed[q] if isunk.get(q, 0) == 1 else "baseline") for q in qids}
        ru = trmod.routed_scores(unk_dec, scores)
        heur_rows.append({"seed": seed, "baseline": "unknown_only_gate", **ru})

        # matched-rate random gate: same repair count as fixed router, selector action if admitted.
        n_repairs = sum(1 for q, a in dec_fixed.items() if a != "baseline" and q in scores)
        rng = np.random.RandomState(10000 + seed)
        random_metrics = []
        qarr = np.array(qids, dtype=object)
        for _ in range(args.random_trials):
            chosen = set(rng.choice(qarr, size=n_repairs, replace=False).tolist()) if n_repairs else set()
            rd = {q: (sp_fixed[q] if q in chosen else "baseline") for q in qids}
            random_metrics.append(trmod.routed_scores(rd, scores))
        for key in ["em", "f1", "repair_rate", "fixed", "harmed", "net_gain", "gain_per_100_repairs", "over_repair_rate"]:
            vals = [m[key] for m in random_metrics]
            # write mean only; SD added as separate fields for key metrics
            pass
        hr = {"seed": seed, "baseline": "random_matched_rate"}
        for key in ["em", "f1", "repair_rate", "fixed", "harmed", "net_gain", "gain_per_100_repairs", "over_repair_rate"]:
            vals = np.array([m[key] for m in random_metrics], dtype=float)
            hr[key] = float(vals.mean())
            hr[key + "_sd"] = float(vals.std(ddof=1))
        heur_rows.append(hr)

    fixed_mat = np.vstack(fixed_vecs)
    mean_router_per_q = fixed_mat.mean(axis=0)
    delta, lo, hi = bootstrap_ci(mean_router_per_q - uniform_vec, n_boot=args.bootstrap)

    # Nested CV: select h_w and tau strictly within each outer-training partition.
    nested_rows = []
    nested_vecs = []
    nested_selection = {}
    for seed in seeds:
        dec, selected = nested_predictions(
            df, id_col, scores, args.model, seed, hws, thrs,
            trmod.make_model, trmod.gate_sample_weights, trmod.routed_scores,
            trmod.two_stage_decide, trmod.oof_two_stage,
            trmod.SELECTOR_ACTIONS, trmod.NUMERIC, trmod.CATEG)
        rr = trmod.routed_scores(dec, scores)
        vec = per_query_correct(dec, scores, qids)
        nested_vecs.append(vec)
        mc = exact_mcnemar(vec, uniform_vec)
        nested_rows.append({"seed": seed, **rr, **{f"mcnemar_{k}": v for k, v in mc.items()}})
        nested_selection[str(seed)] = selected

    nested_mat = np.vstack(nested_vecs)
    ndelta, nlo, nhi = bootstrap_ci(nested_mat.mean(axis=0) - uniform_vec, n_boot=args.bootstrap, seed=54321)

    # Grid dataframe averaged over seeds.
    grid_rows = []
    for (hw, t), vals in grid_acc.items():
        arr = np.array(vals, dtype=float)
        grid_rows.append({"harm_weight": hw, "threshold": t,
                          "em_mean": float(arr.mean()), "em_sd": float(arr.std(ddof=1)),
                          "delta_vs_uniform": float(arr.mean() - uni_metrics[best_uniform_action]["em"])})
    grid_df = pd.DataFrame(grid_rows)
    grid_df.to_csv(out / "grid.csv", index=False)
    pd.DataFrame(fixed_rows).to_csv(out / "fixed_config_per_seed.csv", index=False)
    pd.DataFrame(mcnemar_rows).to_csv(out / "mcnemar_per_seed.csv", index=False)
    pd.DataFrame(nested_rows).to_csv(out / "nested_per_seed.csv", index=False)
    pd.DataFrame(hw_rows).to_csv(out / "ablation_hw.csv", index=False)
    pd.DataFrame(heur_rows).to_csv(out / "heuristic_baselines.csv", index=False)

    # Heatmap, paper-friendly and editable SVG.
    try:
        import matplotlib.pyplot as plt
        piv = grid_df.pivot(index="harm_weight", columns="threshold", values="delta_vs_uniform")
        fig, ax = plt.subplots(figsize=(7.2, 2.8))
        im = ax.imshow(piv.values, aspect="auto")
        ax.set_xticks(range(len(piv.columns)), [f"{x:g}" for x in piv.columns])
        ax.set_yticks(range(len(piv.index)), [f"{x:g}" for x in piv.index])
        ax.set_xlabel("Gate threshold τ")
        ax.set_ylabel("Harm weight $h_w$")
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                ax.text(j, i, f"{100*piv.values[i,j]:+.2f}", ha="center", va="center", fontsize=7)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("ΔEM vs best uniform")
        fig.tight_layout()
        fig.savefig(out / "grid.svg")
        plt.close(fig)
    except Exception as e:
        print(f"[warn] could not render grid.svg: {e}")

    summary = {
        "n": len(qids),
        "best_uniform_action": best_uniform_action,
        "best_uniform": uni_metrics[best_uniform_action],
        "fixed_config": {"harm_weight": args.fixed_hw, "threshold": args.fixed_thr,
                         "paired_bootstrap_delta": delta,
                         "paired_bootstrap_95ci": [lo, hi],
                         "per_seed_mcnemar": mcnemar_rows,
                         "per_seed_metrics": fixed_rows},
        "nested_cv": {"paired_bootstrap_delta": ndelta,
                      "paired_bootstrap_95ci": [nlo, nhi],
                      "per_seed_metrics": nested_rows,
                      "selected_per_outer_fold": nested_selection},
        "notes": {
            "bootstrap_unit": "query; router correctness averaged across seeds before resampling",
            "mcnemar": "two-sided exact binomial test on discordant query outcomes, reported per seed",
            "random_gate": f"matched to fixed router repair count; {args.random_trials} random trials per seed"
        }
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"\nWrote reviewer-defense outputs to {out}")


if __name__ == "__main__":
    main()

