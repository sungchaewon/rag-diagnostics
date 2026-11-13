"""Two-sided semantic re-scoring of every action's answer, for router re-training.

Section 7.3 re-judged only EM-harmed answers. Here every answer that EM marks wrong is
judged with the same GPT-4o-mini prompt (llm_judge_harm.JUDGE_PROMPT), so fixed and
harmed transitions are both re-scored and router labels can be rebuilt:

- EM-correct answers stay correct (no call).
- Empty / abstaining answers ("unknown", "I don't know", ...) stay incorrect (no call).
- Every other EM-incorrect answer is judged once per (question, normalized answer);
  verdicts go to a JSONL cache shared across logs, so reruns and other configurations
  reuse them and an interrupted run resumes.

Outputs:
- --out_log: the log with each action's "em" replaced by the semantic label
  (original kept as "em_exact"; "f1" unchanged). Readable by router_common.
- --out_features: the router feature CSV with best_action / baseline_em recomputed
  from semantic labels. The rule is extract_router_features.best_action (correctness,
  then token F1, then action priority), except that a semantically correct baseline is
  always labeled "baseline". Without that exception, a correct baseline tied with a
  correct repair would lose on lexical F1 to the repair that copies the gold spelling,
  so surface-form rewrites of correct answers would be labeled as interventions.

Evaluate with the existing script, e.g.
    # trained and evaluated on semantic labels
    python scripts/arr/reviewer_defense.py --features <out_features> --log <out_log> --out_dir ... --fixed_thr 0.3
    # trained on EM labels, evaluated on semantic labels
    python scripts/arr/reviewer_defense.py --features <original features> --log <out_log> --out_dir ... --fixed_thr 0.3

Usage (from the repo root; needs OPENAI_API_KEY):
    python scripts/arr/semantic_relabel.py --log outputs/arr/triviaqa_repair_1500_v5.json \
        --features outputs/arr/tqa_router_features.csv \
        --out_log outputs/arr/semantic/tqa_bm25_log.json --out_features outputs/arr/semantic/tqa_bm25_features.csv --dry_run
"""
import argparse
import hashlib
import json
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eval.metrics import normalize_answer  # noqa: E402
from extract_router_features import ACTION_PRIORITY, UNK, best_action, norm  # noqa: E402

DEFAULT_CACHE = "outputs/arr/semantic/judge_cache.jsonl"


def load_records(path):
    raw = json.loads(Path(path).read_text())
    return raw["results"] if isinstance(raw, dict) and "results" in raw else raw


def cache_key(question, pred):
    return hashlib.sha1(f"{question}\x1f{normalize_answer(str(pred or ''))}".encode()).hexdigest()


def load_cache(path):
    cache = {}
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                cache[row["key"]] = row["correct"]
    return cache


def pending_judgments(records, actions, cache):
    """(key, question, gold, pred) for EM-wrong, non-abstaining answers not yet judged."""
    todo = {}
    for rec in records:
        gold = rec.get("golden_answers", rec.get("gold_answers", []))
        for a in actions:
            pred, em = rec[a]["pred"], rec[a]["em"]
            if em == 1 or norm(pred) in UNK:
                continue
            key = cache_key(rec["question"], pred)
            if key not in cache:
                todo.setdefault(key, (key, rec["question"], gold, pred))
    return list(todo.values())


def semantic_label(rec, action, cache):
    pred, em = rec[action]["pred"], rec[action]["em"]
    if em == 1:
        return 1
    if norm(pred) in UNK:
        return 0
    return int(cache[cache_key(rec["question"], pred)])


def semantic_best_action(rec, actions):
    """Preserve a semantically correct baseline; otherwise the original lexicographic rule."""
    return "baseline" if rec["baseline"]["em"] == 1 else best_action(rec, actions)


def run_judge(todo, cache, cache_path, workers):
    from llm_judge_harm import judge  # imports the OpenAI client; only needed when calling

    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    with open(cache_path, "a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(judge, q, gold, pred): (key, q, pred) for key, q, gold, pred in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            key, q, pred = futures[fut]
            correct = bool(fut.result())
            with lock:
                cache[key] = correct
                f.write(json.dumps({"key": key, "question": q, "pred": pred, "correct": correct}, ensure_ascii=False) + "\n")
                f.flush()
            if i % 100 == 0 or i == len(todo):
                print(f"  judged {i}/{len(todo)}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--out_log", required=True)
    ap.add_argument("--out_features", required=True)
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry_run", action="store_true", help="only count the judge calls this log still needs")
    args = ap.parse_args()

    records = load_records(args.log)
    actions = [a for a in ACTION_PRIORITY if all(a in r for r in records)]
    cache = load_cache(args.cache)
    todo = pending_judgments(records, actions, cache)
    print(f"log: {args.log}  n={len(records)}  actions={actions}")
    print(f"cached verdicts: {len(cache)}  judge calls needed: {len(todo)}")
    if args.dry_run:
        return
    if todo:
        run_judge(todo, cache, args.cache, args.workers)

    relabeled = []
    for rec in records:
        new = dict(rec)
        for a in actions:
            new[a] = {**rec[a], "em_exact": rec[a]["em"], "em": semantic_label(rec, a, cache)}
        relabeled.append(new)
    Path(args.out_log).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_log).write_text(json.dumps(relabeled, indent=1, ensure_ascii=False))

    by_id = {str(r["id"]): r for r in relabeled}
    feats = pd.read_csv(args.features)
    feats["id"] = feats["id"].astype(str)
    missing = set(feats["id"]) - set(by_id)
    if missing:
        sys.exit(f"[semantic_relabel] {len(missing)} feature ids not in the log, e.g. {sorted(missing)[:3]}")
    old_best = feats["best_action"].copy()
    feats["best_action"] = [semantic_best_action(by_id[i], actions) for i in feats["id"]]
    feats["baseline_em"] = [by_id[i]["baseline"]["em"] for i in feats["id"]]
    Path(args.out_features).parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(args.out_features, index=False)

    rows = []
    for a in actions:
        for kind, key in [("exact", "em_exact"), ("semantic", "em")]:
            base = [r["baseline"][key] for r in relabeled]
            act = [r[a][key] for r in relabeled]
            fixed = sum(b == 0 and x == 1 for b, x in zip(base, act))
            harmed = sum(b == 1 and x == 0 for b, x in zip(base, act))
            rows.append({"action": a, "labels": kind, "acc": sum(act) / len(act),
                         "fixed": fixed, "harmed": harmed, "net": fixed - harmed})
    with pd.option_context("display.width", None, "display.float_format", "{:.4f}".format):
        print("\n== exact match vs semantic labels ==")
        print(pd.DataFrame(rows).to_string(index=False))
    changed = int((old_best != feats["best_action"]).sum())
    print(f"\nbest_action changed for {changed}/{len(feats)} questions")
    print(f"semantic best_action distribution: {dict(Counter(feats['best_action']))}")
    print(f"wrote {args.out_log} and {args.out_features}")


if __name__ == "__main__":
    main()
