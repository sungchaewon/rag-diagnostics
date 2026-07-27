import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from router_common import load_action_scores

BASE_NUMERIC = ["is_unknown", "answer_len", "ctx_overlap", "is_numeric_q",
                "ans_has_digit", "bm25_top1", "bm25_margin", "bm25_mean10"]
BASE_CATEG = ["qtype"]

# diagnostic features (present only after join_diagnostic_features.py)
DIAG_NUMERIC = ["bm25_hit", "gold_rank_capped"]
DIAG_CATEG = ["rank_bucket", "split_retrieval", "split_answer_type"]

# populated by configure_features()
NUMERIC = list(BASE_NUMERIC)
CATEG = list(BASE_CATEG)


def configure_features(df, use_diag):
    """Set the active feature set; returns a label describing it."""
    global NUMERIC, CATEG
    NUMERIC, CATEG = list(BASE_NUMERIC), list(BASE_CATEG)
    if not use_diag:
        return "surface-only"
    have = [c for c in DIAG_NUMERIC + DIAG_CATEG if c in df.columns]
    if not have:
        print("[train_router] --use_diag set but no diagnostic columns found; "
              "run join_diagnostic_features.py first. Falling back.")
        return "surface-only"
    NUMERIC += [c for c in DIAG_NUMERIC if c in df.columns]
    CATEG += [c for c in DIAG_CATEG if c in df.columns]
    return f"surface+diagnostic ({', '.join(have)})"
SELECTOR_ACTIONS = ["retrieval_repair", "generation_repair_v2",
                    "generation_repair_v3"]
THRESHOLDS = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9]


# ---------------------------------------------------------------- data
def load_features(path):
    df = pd.read_csv(path)
    id_col = next((c for c in ["qid", "id", "query_id", "idx"]
                   if c in df.columns), None)
    if id_col is None:
        df["qid"] = df.index.astype(str)
        id_col = "qid"
    df[id_col] = df[id_col].astype(str)
    missing = [c for c in BASE_NUMERIC + BASE_CATEG + ["best_action"]
               if c not in df.columns]
    if missing:
        sys.exit(f"[train_router] missing columns in {path}: {missing}")
    return df, id_col


def make_model(kind):
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEG),
    ])
    if kind == "gbt":
        clf = GradientBoostingClassifier(random_state=0)
    else:
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    return Pipeline([("pre", pre), ("clf", clf)])


# ------------------------------------------------------------- routing
def routed_scores(decisions, scores):
    """decisions: qid -> action. Returns EM, F1, harm/fix counts."""
    em = f1 = 0.0
    harmed = fixed = repairs = 0
    n = 0
    for qid, act in decisions.items():
        v = scores.get(qid)
        if v is None:
            continue
        a = act if act in v else "baseline"
        em += v[a]["em"]
        f1 += v[a]["f1"]
        n += 1
        if a != "baseline":
            repairs += 1
            if v[a]["em"] > v["baseline"]["em"]:
                fixed += 1
            elif v[a]["em"] < v["baseline"]["em"]:
                harmed += 1
    net = fixed - harmed
    return {"em": em / n, "f1": f1 / n, "n": n,
            "repair_rate": repairs / n,
            "fixed": fixed, "harmed": harmed,
            "net_gain": net,
            "gain_per_100_repairs": 100.0 * net / repairs if repairs else 0.0,
            "over_repair_rate": (repairs - fixed) / repairs if repairs else 0.0}


def two_stage_decide(gate_proba, sel_pred, thr):
    """gate_proba: qid -> P(repair); sel_pred: qid -> action."""
    return {qid: (sel_pred[qid] if p >= thr else "baseline")
            for qid, p in gate_proba.items()}


# ------------------------------------------------- out-of-fold training
def gate_sample_weights(df, harm_weight):
    """
    Harm-aware weighting for the gate. Misclassifying a baseline-success
    query as repairable is the only path to harm (FIXED/HARMED cross-tab),
    so those negatives get weight `harm_weight`; everything else weight 1.
    harm_weight=1.0 reproduces the original unweighted training.
    """
    w = np.ones(len(df))
    if harm_weight != 1.0 and "baseline_em" in df.columns:
        harm_risk = ((df["best_action"] == "baseline")
                     & (df["baseline_em"] == 1)).values
        w[harm_risk] = harm_weight
    return w


def oof_two_stage(df, id_col, model_kind, harm_weight=1.0, seed=0):
    """5-fold OOF gate probabilities + selector predictions."""
    X = df[NUMERIC + CATEG]
    y_gate = (df["best_action"] != "baseline").astype(int).values
    weights = gate_sample_weights(df, harm_weight)
    gate_proba = pd.Series(index=df.index, dtype=float)
    sel_pred = pd.Series(index=df.index, dtype=object)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y_gate):
        gate = make_model(model_kind)
        gate.fit(X.iloc[tr], y_gate[tr], clf__sample_weight=weights[tr])
        gate_proba.iloc[te] = gate.predict_proba(X.iloc[te])[:, 1]

        sel_mask = df.iloc[tr]["best_action"].isin(SELECTOR_ACTIONS)
        sel = make_model(model_kind)
        sel.fit(X.iloc[tr][sel_mask.values],
                df.iloc[tr]["best_action"][sel_mask.values])
        sel_pred.iloc[te] = sel.predict(X.iloc[te])

    qids = df[id_col].values
    return (dict(zip(qids, gate_proba.values)),
            dict(zip(qids, sel_pred.values)))


def oof_one_stage(df, id_col, model_kind, seed=0):
    """Ablation: single 4-way classifier."""
    X = df[NUMERIC + CATEG]
    y = df["best_action"].values
    pred = pd.Series(index=df.index, dtype=object)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        m = make_model(model_kind)
        m.fit(X.iloc[tr], y[tr])
        pred.iloc[te] = m.predict(X.iloc[te])
    return dict(zip(df[id_col].values, pred.values))


def fit_full(df, model_kind, harm_weight=1.0):
    """Fit gate + selector on the full dataframe (for transfer)."""
    X = df[NUMERIC + CATEG]
    weights = gate_sample_weights(df, harm_weight)
    gate = make_model(model_kind)
    gate.fit(X, (df["best_action"] != "baseline").astype(int),
             clf__sample_weight=weights)
    sel_mask = df["best_action"].isin(SELECTOR_ACTIONS)
    sel = make_model(model_kind)
    sel.fit(X[sel_mask.values], df["best_action"][sel_mask.values])
    return gate, sel


def feature_importance(gate, sel):
    """Readable importance for logreg (|coef|) or gbt."""
    out = {}
    for name, model in [("gate", gate), ("selector", sel)]:
        pre = model.named_steps["pre"]
        clf = model.named_steps["clf"]
        names = list(NUMERIC) + list(
            pre.named_transformers_["cat"].get_feature_names_out(CATEG))
        if hasattr(clf, "coef_"):
            w = np.abs(clf.coef_).mean(axis=0)
        else:
            w = clf.feature_importances_
        top = sorted(zip(names, w), key=lambda t: -t[1])[:8]
        out[name] = [(n, round(float(v), 4)) for n, v in top]
    return out


# ---------------------------------------------------------------- main
def _mean_std(vals):
    arr = np.array(vals, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def multi_seed_eval(df, id_col, scores, model_kind, harm_weight, thr, seeds,
                    edf=None, eid=None, escores=None):
    """
    Fix (harm_weight, threshold) and vary only the seed. Returns
    mean/std of routed EM, F1, and harmed for in-dataset (OOF) and,
    if eval data is given, for transfer. This is what decides whether
    a small in-dataset gap (e.g. NQ +1%p) survives seed noise.
    """
    in_em, in_f1, in_harm = [], [], []
    tr_em, tr_f1, tr_harm = [], [], []
    for s in seeds:
        gate_p, sel_p = oof_two_stage(df, id_col, model_kind,
                                      harm_weight=harm_weight, seed=s)
        dec = two_stage_decide(gate_p, sel_p, thr)
        r = routed_scores(dec, scores)
        in_em.append(r["em"]); in_f1.append(r["f1"]); in_harm.append(r["harmed"])

        if edf is not None:
            # transfer: refit gate/selector on full train at this seed
            g, sl = fit_full(df, model_kind, harm_weight=harm_weight)
            X = edf[NUMERIC + CATEG]
            qids = edf[eid].values
            gp = dict(zip(qids, g.predict_proba(X)[:, 1]))
            sp = dict(zip(qids, sl.predict(X)))
            tdec = two_stage_decide(gp, sp, thr)
            tr = routed_scores(tdec, escores)
            tr_em.append(tr["em"]); tr_f1.append(tr["f1"])
            tr_harm.append(tr["harmed"])

    out = {"n_seeds": len(seeds), "harm_weight": harm_weight, "threshold": thr,
           "in_em": _mean_std(in_em), "in_f1": _mean_std(in_f1),
           "in_harmed": _mean_std(in_harm), "in_em_raw": in_em}
    if edf is not None:
        out.update({"tr_em": _mean_std(tr_em), "tr_f1": _mean_std(tr_f1),
                    "tr_harmed": _mean_std(tr_harm), "tr_em_raw": tr_em})
    return out


def report_seed_block(res, best_uniform_em, tr_uniform_em=None):
    m, s = res["in_em"]
    fm, fs = res["in_f1"]
    hm, hs = res["in_harmed"]
    print(f"\n=== multi-seed ({res['n_seeds']} seeds, "
          f"hw={res['harm_weight']}, thr={res['threshold']}) ===")
    print(f"in-dataset routed EM : {m:.4f} +/- {s:.4f}  "
          f"(F1 {fm:.4f} +/- {fs:.4f}, harmed {hm:.1f} +/- {hs:.1f})")
    delta = m - best_uniform_em
    margin = delta / s if s > 0 else float("inf")
    verdict = ("SIGNIFICANT" if delta > 2 * s else
               "MARGINAL" if delta > s else "NOT DISTINGUISHABLE")
    print(f"  vs best uniform {best_uniform_em:.4f}: "
          f"delta {delta:+.4f} ({margin:.1f} sigma) -> {verdict}")
    print(f"  per-seed EM: {[round(x, 4) for x in res['in_em_raw']]}")
    if "tr_em" in res:
        tm, ts = res["tr_em"]
        thm, ths = res["tr_harmed"]
        print(f"transfer routed EM  : {tm:.4f} +/- {ts:.4f}  "
              f"(harmed {thm:.1f} +/- {ths:.1f})")
        if tr_uniform_em is not None:
            d = tm - tr_uniform_em
            mg = d / ts if ts > 0 else float("inf")
            v = ("SIGNIFICANT" if d > 2 * ts else
                 "MARGINAL" if d > ts else "NOT DISTINGUISHABLE")
            print(f"  vs eval best uniform {tr_uniform_em:.4f}: "
                  f"delta {d:+.4f} ({mg:.1f} sigma) -> {v}")
        print(f"  per-seed EM: {[round(x, 4) for x in res['tr_em_raw']]}")


def report_block(title, rows):
    print(f"\n=== {title} ===")
    print(f"{'setting':<34}{'EM':>8}{'F1':>8}{'rep%':>7}"
          f"{'fixed':>7}{'harmed':>7}{'net':>6}{'g/100r':>8}{'overR%':>8}")
    for name, r in rows:
        print(f"{name:<34}{r['em']:>8.4f}{r['f1']:>8.4f}"
              f"{r['repair_rate']:>7.1%}{r['fixed']:>7}{r['harmed']:>7}"
              f"{r['net_gain']:>6}{r['gain_per_100_repairs']:>8.1f}"
              f"{r['over_repair_rate']:>8.1%}")


def uniform_rows(scores):
    actions = sorted({a for v in scores.values() for a in v})
    rows = []
    for a in actions:
        dec = {qid: a for qid in scores}
        rows.append((f"uniform:{a}", routed_scores(dec, scores)))
    oracle = {qid: max(v, key=lambda a: v[a]["em"]) for qid, v in scores.items()}
    rows.append(("oracle_routing", routed_scores(oracle, scores)))
    return rows


def main(args):
    df, id_col = load_features(args.features)
    scores = load_action_scores(args.log)
    tag = Path(args.features).stem
    feat_label = configure_features(df, args.use_diag)
    results = {"train": tag, "feature_set": feat_label,
               "numeric": list(NUMERIC), "categorical": list(CATEG)}

    print(f"train features: {args.features} ({len(df)} rows)")
    print(f"feature set: {feat_label}")
    print(f"best_action distribution: "
          f"{df['best_action'].value_counts().to_dict()}")

    # baselines
    rows = uniform_rows(scores)
    report_block("uniform / oracle baselines (train dataset)", rows)
    results["baselines"] = {k: v for k, v in rows}

    # two-stage OOF, harm-weight x threshold sweep
    harm_weights = [float(x) for x in args.harm_weights.split(",")]
    best_overall = None
    results["two_stage"] = {}
    for hw in harm_weights:
        gate_p, sel_p = oof_two_stage(df, id_col, args.model, harm_weight=hw)
        sweep = []
        for thr in THRESHOLDS:
            dec = two_stage_decide(gate_p, sel_p, thr)
            sweep.append((f"hw={hw} thr={thr}", routed_scores(dec, scores)))
        report_block(f"two-stage router, 5-fold OOF, harm_weight={hw}", sweep)
        results["two_stage"][f"hw={hw}"] = {k: v for k, v in sweep}
        top = max(sweep, key=lambda t: t[1]["em"])
        if best_overall is None or top[1]["em"] > best_overall[1]["em"]:
            best_overall = top
    print(f"\nbest overall by routed EM: {best_overall[0]} "
          f"(EM {best_overall[1]['em']:.4f}, harmed {best_overall[1]['harmed']})")

    # one-stage ablation
    one = oof_one_stage(df, id_col, args.model)
    r1 = routed_scores(one, scores)
    report_block("ablation: 1-stage 4-way", [("1-stage 4-way", r1)])
    results["one_stage"] = r1

    # feature importance (full fit)
    best_hw = float(best_overall[0].split()[0].split("=")[1])
    gate, sel = fit_full(df, args.model, harm_weight=best_hw)
    print(f"\n(full fit uses best harm_weight={best_hw})")
    fi = feature_importance(gate, sel)
    print("\n=== feature importance (top 8, full fit) ===")
    for part, items in fi.items():
        print(f"[{part}] " + ", ".join(f"{n}={v}" for n, v in items))
    results["feature_importance"] = fi

    # cross-dataset transfer
    if args.eval_features and args.eval_log:
        edf, eid = load_features(args.eval_features)
        escores = load_action_scores(args.eval_log)
        print(f"\ntransfer eval: {args.eval_features} ({len(edf)} rows)")
        erows = uniform_rows(escores)
        report_block("uniform / oracle baselines (eval dataset)", erows)
        results["eval_baselines"] = {k: v for k, v in erows}

        X = edf[NUMERIC + CATEG]
        qids = edf[eid].values
        results["transfer"] = {}
        for hw in harm_weights:
            tgate, tsel = fit_full(df, args.model, harm_weight=hw)
            gate_pe = dict(zip(qids, tgate.predict_proba(X)[:, 1]))
            sel_pe = dict(zip(qids, tsel.predict(X)))
            tsweep = []
            for thr in THRESHOLDS:
                dec = two_stage_decide(gate_pe, sel_pe, thr)
                tsweep.append((f"hw={hw} thr={thr}",
                               routed_scores(dec, escores)))
            report_block(f"cross-dataset transfer, harm_weight={hw}", tsweep)
            results["transfer"][f"hw={hw}"] = {k: v for k, v in tsweep}

    # multi-seed verification at a fixed (harm_weight, threshold)
    if args.seeds:
        seeds = [int(x) for x in args.seeds.split(",")]
        bu_em = max(v["em"] for k, v in results["baselines"].items()
                    if k != "oracle_routing")
        edf2 = eid2 = escores2 = None
        tr_bu = None
        if args.eval_features and args.eval_log:
            edf2, eid2 = load_features(args.eval_features)
            escores2 = load_action_scores(args.eval_log)
            tr_bu = max(v["em"] for k, v in results["eval_baselines"].items()
                        if k != "oracle_routing")
        seed_res = multi_seed_eval(
            df, id_col, scores, args.model,
            harm_weight=args.fix_hw, thr=args.fix_thr, seeds=seeds,
            edf=edf2, eid=eid2, escores=escores2)
        report_seed_block(seed_res, bu_em, tr_bu)
        results["multi_seed"] = seed_res

    out = Path("outputs/arr")
    out.mkdir(parents=True, exist_ok=True)
    suffix = "diag" if args.use_diag and "diagnostic" in feat_label else "surface"
    out_path = out / f"router_results_{tag}_{args.model}_{suffix}.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--eval_features")
    ap.add_argument("--eval_log")
    ap.add_argument("--model", default="logreg", choices=["logreg", "gbt"])
    ap.add_argument("--harm_weights", default="1,3,5,10",
                    help="comma list of gate sample weights for "
                         "baseline-success negatives (1=unweighted)")
    ap.add_argument("--use_diag", action="store_true",
                    help="include CIKM diagnostic features if present")
    ap.add_argument("--seeds", default="",
                    help="comma list of seeds for multi-seed verification, "
                         "e.g. 0,1,2,3,4 (empty = skip)")
    ap.add_argument("--fix_hw", type=float, default=5.0,
                    help="harm_weight fixed during multi-seed run")
    ap.add_argument("--fix_thr", type=float, default=0.4,
                    help="gate threshold fixed during multi-seed run")
    args = ap.parse_args()
    main(args)