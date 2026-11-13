"""Human audit of the semantic judge used by semantic_relabel.py.

Three steps, all without API calls:

1. sample   — draw judged answers (EM-incorrect, non-abstaining) from the relabeled logs,
              stratified by dataset (nq / tqa) x judge verdict, one item per unique
              (question, normalized answer). Writes items.csv WITHOUT the judge verdict,
              shuffled, plus key.csv with the verdicts, so annotation is blind.
2. annotate — show each unannotated item in the terminal and record c (correct),
              i (incorrect) or u (unsure) into items.csv; safe to stop and resume.
3. score    — compare human labels with the judge: agreement, Cohen's kappa, and the
              precision of each judge verdict (how often "correct" is really correct)
              with 95% Wilson intervals, overall and per dataset.

Usage (from the repo root):
    python scripts/arr/judge_audit.py sample --logs outputs/arr/semantic/*_log.json --out_dir outputs/arr/semantic/audit
    python scripts/arr/judge_audit.py annotate --out_dir outputs/arr/semantic/audit
    python scripts/arr/judge_audit.py score --out_dir outputs/arr/semantic/audit
"""
import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eval.metrics import normalize_answer  # noqa: E402
from extract_router_features import ACTION_PRIORITY, UNK, norm  # noqa: E402

LABELS = {"c": "correct", "i": "incorrect", "u": "unsure"}


def dataset_of(config):
    return "tqa" if "tqa" in config else "nq"


def judged_items(log_path):
    """Unique (question, answer) pairs whose label came from the judge, with its verdict."""
    config = Path(log_path).stem.removesuffix("_log")
    items = {}
    for rec in json.loads(Path(log_path).read_text()):
        gold = rec.get("golden_answers", rec.get("gold_answers", []))
        for a in ACTION_PRIORITY:
            v = rec.get(a)
            if not v or "em_exact" not in v or v["em_exact"] == 1 or norm(v["pred"]) in UNK:
                continue
            key = (rec["question"], normalize_answer(str(v["pred"])))
            items.setdefault(key, {"config": config, "dataset": dataset_of(config), "action": a,
                                   "question": rec["question"], "gold": " | ".join(map(str, gold)),
                                   "answer": v["pred"], "judge": "correct" if v["em"] == 1 else "incorrect"})
    return list(items.values())


def cmd_sample(args):
    out = Path(args.out_dir)
    if (out / "items.csv").exists() and not args.force:
        sys.exit(f"{out / 'items.csv'} exists; pass --force to resample (this discards annotations)")
    pool, seen = [], set()
    for path in args.logs:
        for it in judged_items(path):
            k = (it["question"], normalize_answer(str(it["answer"])))
            if k not in seen:  # the same answer can appear under several configurations
                seen.add(k)
                pool.append(it)
    rng = random.Random(args.seed)
    picked = []
    for ds in ["nq", "tqa"]:
        for verdict in ["correct", "incorrect"]:
            cell = [it for it in pool if it["dataset"] == ds and it["judge"] == verdict]
            rng.shuffle(cell)
            picked += cell[:args.n_per_cell]
            print(f"{ds:4s} judge={verdict:9s}: {len(cell):5d} available, {min(len(cell), args.n_per_cell)} sampled")
    rng.shuffle(picked)
    df = pd.DataFrame(picked)
    df.insert(0, "item", range(1, len(df) + 1))
    out.mkdir(parents=True, exist_ok=True)
    df[["item", "question", "gold", "answer"]].assign(human="", note="").to_csv(out / "items.csv", index=False)
    df[["item", "config", "dataset", "action", "judge"]].to_csv(out / "key.csv", index=False)
    print(f"wrote {len(df)} items to {out / 'items.csv'} (blind) and verdicts to {out / 'key.csv'}")


def cmd_annotate(args):
    path = Path(args.out_dir) / "items.csv"
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    todo = df.index[df["human"] == ""].tolist()
    print(f"{len(df) - len(todo)}/{len(df)} done. Is the answer correct for the question, given the gold answers?\n"
          "c = correct (same fact, any wording)   i = incorrect   u = unsure   n = add a note first   q = quit\n")
    for idx in todo:
        row = df.loc[idx]
        print(f"[{row['item']}/{len(df)}] Q: {row['question']}\n   gold:   {row['gold']}\n   answer: {row['answer']}")
        note = ""
        while True:
            key = input("   > ").strip().lower()
            if key == "q":
                df.to_csv(path, index=False)
                print("saved; rerun annotate to continue")
                return
            if key == "n":
                note = input("   note: ").strip()
                continue
            if key in LABELS:
                df.loc[idx, ["human", "note"]] = [LABELS[key], note]
                df.to_csv(path, index=False)
                print()
                break
    print("all items annotated; run score")


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((list(a).count(c) / n) * (list(b).count(c) / n) for c in ["correct", "incorrect"])
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def cmd_score(args):
    out = Path(args.out_dir)
    df = pd.read_csv(out / "items.csv", dtype=str, keep_default_na=False).merge(
        pd.read_csv(out / "key.csv", dtype=str), on="item")
    done = df[df["human"].isin(["correct", "incorrect"])]
    print(f"annotated: {(df['human'] != '').sum()}/{len(df)}  usable (not unsure): {len(done)}")
    rows = []
    for name, g in [("all", done)] + [(ds, done[done["dataset"] == ds]) for ds in ["nq", "tqa"]]:
        if g.empty:
            continue
        agree = int((g["judge"] == g["human"]).sum())
        row = {"subset": name, "n": len(g), "agreement": agree / len(g),
               "agree_ci": wilson(agree, len(g)), "kappa": kappa(g["judge"].tolist(), g["human"].tolist())}
        for verdict in ["correct", "incorrect"]:
            s = g[g["judge"] == verdict]
            k = int((s["human"] == verdict).sum())
            row[f"judge_{verdict}_precision"] = k / len(s) if len(s) else float("nan")
            row[f"judge_{verdict}_ci"] = wilson(k, len(s))
            row[f"judge_{verdict}_n"] = len(s)
        rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(out / "score.csv", index=False)
    fmt = lambda ci: f"[{ci[0]:.2f}, {ci[1]:.2f}]"  # noqa: E731
    for r in rows:
        print(f"\n== {r['subset']} (n={r['n']}) ==\n"
              f"agreement {r['agreement']:.2f} {fmt(r['agree_ci'])}  kappa {r['kappa']:.2f}\n"
              f"judge says correct   -> human agrees {r['judge_correct_precision']:.2f} {fmt(r['judge_correct_ci'])} (n={r['judge_correct_n']})\n"
              f"judge says incorrect -> human agrees {r['judge_incorrect_precision']:.2f} {fmt(r['judge_incorrect_ci'])} (n={r['judge_incorrect_n']})")
    disagree = df[df["human"].isin(["correct", "incorrect"]) & (df["judge"] != df["human"])]
    disagree[["item", "config", "action", "question", "gold", "answer", "judge", "human", "note"]].to_csv(
        out / "disagreements.csv", index=False)
    print(f"\n{len(disagree)} disagreements written to {out / 'disagreements.csv'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--logs", nargs="+", required=True, help="relabeled logs written by semantic_relabel.py")
    s.add_argument("--out_dir", required=True)
    s.add_argument("--n_per_cell", type=int, default=25, help="items per dataset x judge-verdict cell")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--force", action="store_true")
    for name in ["annotate", "score"]:
        sub.add_parser(name).add_argument("--out_dir", required=True)
    args = ap.parse_args()
    {"sample": cmd_sample, "annotate": cmd_annotate, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
