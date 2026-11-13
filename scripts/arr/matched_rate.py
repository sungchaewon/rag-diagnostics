"""Harm weight vs. threshold: compare gate weights at matched repair budgets.

A larger harm weight lowers gate scores, so at a fixed threshold it also repairs
fewer queries; fewer harmed answers at a fixed tau can therefore come from
intervening less rather than from choosing better. This script removes that
confound. For every budget k it admits the k queries with the highest gate
score under each harm weight and applies the (harm-weight-independent) selector
action, so policies differ only in *which* queries the gate ranks first.

The selector and the OOF folds do not depend on the harm weight (the folds are
stratified on the gate label, which is the same for every weight), so at a
matched budget any difference in fixed / harmed / EM is due to the gate ranking.

Budgets: fractions of the dataset, plus "op" = the number of repairs the
reference weight (--ref_hw) makes at the paper's operating threshold (--fixed_thr),
recomputed per seed. Per-query outcomes are averaged over seeds before
query-level paired bootstrap, as in reviewer_defense.py.

Usage (from the repo root):
    python scripts/arr/matched_rate.py --features <router_features.csv> \
        --log <repair_log.json> --out_dir outputs/arr/matched_rate/bm25_tqa --fixed_thr 0.3
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_router as trmod  # noqa: E402
from reviewer_defense import bootstrap_ci  # noqa: E402
from router_common import load_action_scores  # noqa: E402

DEFAULT_FRACTIONS = "0.02,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.8,1.0"


def outcome_vectors(decisions, scores, qids):
    """Per-query EM, fixed, harmed and repaired indicators (same rules as routed_scores)."""
    n = len(qids)
    em, fixed, harmed, repaired = (np.zeros(n) for _ in range(4))
    for i, q in enumerate(qids):
        v = scores[q]
        a = decisions.get(q, "baseline")
        a = a if a in v else "baseline"
        base, new = v["baseline"]["em"], v[a]["em"]
        em[i] = new
        if a != "baseline":
            repaired[i] = 1
            fixed[i] = new > base
            harmed[i] = new < base
    return {"em": em, "fixed": fixed, "harmed": harmed, "repaired": repaired}


def top_k_decisions(gate_p, sel_p, qids, k):
    """Admit the k queries with the highest gate score; ties keep dataset order."""
    p = np.array([gate_p[q] for q in qids])
    admitted = set(np.asarray(qids, dtype=object)[np.argsort(-p, kind="stable")[:k]])
    return {q: (sel_p[q] if q in admitted else "baseline") for q in qids}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model", default="logreg", choices=["logreg", "gbt"])
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--harm_weights", default="1,3,5,10")
    ap.add_argument("--ref_hw", type=float, default=5.0, help="weight whose operating point defines the 'op' budget")
    ap.add_argument("--base_hw", type=float, default=1.0, help="weight every other weight is compared against")
    ap.add_argument("--fixed_thr", type=float, required=True, help="paper operating threshold (.4 BM25 NQ, .3 otherwise)")
    ap.add_argument("--fractions", default=DEFAULT_FRACTIONS)
    ap.add_argument("--bootstrap", type=int, default=20000)
    args = ap.parse_args()

    df, id_col = trmod.load_features(args.features)
    trmod.configure_features(df, use_diag=False)
    scores = load_action_scores(args.log)
    qids = [q for q in df[id_col].astype(str).tolist() if q in scores]
    n = len(qids)
    seeds = [int(x) for x in args.seeds.split(",")]
    hws = sorted({float(x) for x in args.harm_weights.split(",")} | {args.ref_hw, args.base_hw})
    fractions = [float(x) for x in args.fractions.split(",")]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    uniform = {a: trmod.routed_scores({q: a for q in qids}, scores) for a in sorted({a for q in qids for a in scores[q]})}
    best_uniform = max(uniform, key=lambda a: uniform[a]["em"])

    # vecs[(hw, budget)] -> list over seeds of outcome dicts
    vecs, op_counts = {}, []
    for seed in seeds:
        preds = {hw: trmod.oof_two_stage(df, id_col, args.model, harm_weight=hw, seed=seed) for hw in hws}
        ref_gate, ref_sel = preds[args.ref_hw]
        op_k = int(sum(a != "baseline" for q, a in trmod.two_stage_decide(ref_gate, ref_sel, args.fixed_thr).items()
                       if q in scores))
        op_counts.append(op_k)
        budgets = [("op", op_k)] + [(f"{f:g}", int(round(f * n))) for f in fractions]
        for hw, (gate_p, sel_p) in preds.items():
            # Selector must not depend on the weight, otherwise the comparison is not gate-only.
            assert sel_p == ref_sel, "selector predictions differ across harm weights"
            for label, k in budgets:
                vecs.setdefault((hw, label), []).append(outcome_vectors(top_k_decisions(gate_p, sel_p, qids, k), scores, qids))

    def seed_mean(hw, label, key):
        return np.mean([v[key] for v in vecs[(hw, label)]], axis=0)

    rows, curve = [], []
    for (hw, label), runs in vecs.items():
        row = {"harm_weight": hw, "budget": label,
               "repair_rate": float(np.mean([r["repaired"].mean() for r in runs])),
               "em": float(np.mean([r["em"].mean() for r in runs])),
               "fixed": float(np.mean([r["fixed"].sum() for r in runs])),
               "harmed": float(np.mean([r["harmed"].sum() for r in runs]))}
        row["net"] = row["fixed"] - row["harmed"]
        row["em_vs_best_uniform_pp"] = 100 * (row["em"] - uniform[best_uniform]["em"])
        if hw != args.base_hw:
            for key, scale, name in [("em", 100.0, "d_em_pp"), ("harmed", n, "d_harmed"), ("fixed", n, "d_fixed")]:
                d, lo, hi = bootstrap_ci(seed_mean(hw, label, key) - seed_mean(args.base_hw, label, key), n_boot=args.bootstrap)
                row.update({name: d * scale, f"{name}_lo": lo * scale, f"{name}_hi": hi * scale})
        (rows if label == "op" else curve).append(row)

    order = lambda frame: frame.sort_values(["harm_weight", "repair_rate"]).reset_index(drop=True)  # noqa: E731
    op_df, curve_df = order(pd.DataFrame(rows)), order(pd.DataFrame(curve))
    op_df.to_csv(out / "matched_op.csv", index=False)
    curve_df.to_csv(out / "matched_curve.csv", index=False)

    summary = {"n": n, "features": args.features, "log": args.log, "model": args.model, "seeds": seeds,
               "best_uniform_action": best_uniform, "best_uniform_em": uniform[best_uniform]["em"],
               "ref_hw": args.ref_hw, "base_hw": args.base_hw, "fixed_thr": args.fixed_thr,
               "op_repairs_per_seed": op_counts,
               "notes": "differences are (hw - base_hw) at the same repair budget; CIs are query-level paired bootstrap "
                        "over seed-averaged outcomes; d_harmed/d_fixed in query counts, d_em_pp in percentage points"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
        for hw, g in curve_df.groupby("harm_weight"):
            axes[0].plot(100 * g["repair_rate"], g["harmed"], marker="o", ms=3, label=f"$h_w$={hw:g}")
            axes[1].plot(100 * g["repair_rate"], g["net"], marker="o", ms=3, label=f"$h_w$={hw:g}")
        for ax, ylabel in zip(axes, ["Harmed queries", "Net (fixed - harmed)"]):
            ax.axvline(100 * np.mean(op_counts) / n, color="grey", ls=":", lw=1)
            ax.set_xlabel("Repair rate (%)")
            ax.set_ylabel(ylabel)
        axes[1].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "matched_curve.svg")
        plt.close(fig)
    except Exception as e:  # plotting is optional on the server
        print(f"[warn] could not render matched_curve.svg: {e}")

    cols = ["harm_weight", "budget", "repair_rate", "em", "fixed", "harmed", "net",
            "d_em_pp", "d_em_pp_lo", "d_em_pp_hi", "d_harmed", "d_harmed_lo", "d_harmed_hi"]
    with pd.option_context("display.width", None, "display.max_columns", None, "display.float_format", "{:.4f}".format):
        print(f"n={n}  best uniform: {best_uniform} EM {uniform[best_uniform]['em']:.4f}  "
              f"op budget (h_w={args.ref_hw:g}, tau={args.fixed_thr:g}) repairs per seed: {op_counts}")
        print(f"\n== matched at the operating-point budget (differences vs h_w={args.base_hw:g}) ==")
        print(op_df.reindex(columns=cols).to_string(index=False))
        print("\n== matched-budget curve ==")
        print(curve_df.reindex(columns=cols).to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
