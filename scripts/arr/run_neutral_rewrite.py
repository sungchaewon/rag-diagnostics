import json
import re
import time
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import openai
from src.generation.gpt4omini import generate as _generate
from src.repairs.retrieval_repair import retrieval_repair
from eval.metrics import compute_em, compute_f1

_WAIT_RE = re.compile(r"try again in ([\d.]+)\s*s", re.IGNORECASE)


def normalize_contexts(contexts):
    out = []
    for c in contexts:
        if isinstance(c, dict):
            out.append(c.get("contents") or c.get("text") or str(c))
        else:
            out.append(str(c))
    return out


def make_retriever(kind):
    if kind == "bm25":
        from src.retrieval.bm25_baseline import retrieve_bm25 as fn
    else:
        from src.retrieval.dense_baseline import retrieve_dense as fn

    class R:
        def retrieve(self, query, top_k=10):
            return normalize_contexts(fn(query, top_k=top_k))
    return R()


def make_generator(model):
    class G:
        def generate(self, question, contexts):
            return _generate(question, contexts, model=model)
    return G()


def call_with_retry(fn, max_retries=200):
    attempt = 0
    while True:
        try:
            return fn()
        except openai.RateLimitError as e:
            msg = str(e)
            if "insufficient_quota" in msg or "credit_balance" in msg:
                raise  # out of credits: retrying cannot help
            attempt += 1
            if attempt > max_retries:
                raise
            m = _WAIT_RE.search(msg)
            wait = float(m.group(1)) + 1.0 if m else min(30 * attempt, 300)
            print(f"[rate-limit] attempt {attempt}, sleeping {wait:.1f}s", flush=True)
            time.sleep(wait)


def load_records(path, n):
    raw = json.loads(Path(path).read_text())
    rows = raw["results"] if isinstance(raw, dict) and "results" in raw else raw
    return rows[:n] if n else rows


def load_done(path):
    p = Path(path)
    if not p.exists():
        return []
    try:
        d = json.load(open(p))
        return d if isinstance(d, list) else []
    except Exception:
        return []


def save(path, results):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def summarize(results):
    n = len(results)
    if not n:
        return
    def net(key):
        fixed = sum(1 for r in results if r["baseline"]["em"] == 0 and r[key]["em"] == 1)
        harmed = sum(1 for r in results if r["baseline"]["em"] == 1 and r[key]["em"] == 0)
        em = sum(r[key]["em"] for r in results) / n
        return em, fixed, harmed, fixed - harmed
    base_em = sum(r["baseline"]["em"] for r in results) / n
    print(f"\n n={n}  baseline EM={base_em:.4f}")
    for key in ("retrieval_repair_legacy", "retrieval_repair"):
        if key in results[0]:
            em, f, h, nt = net(key)
            label = "neutral" if key == "retrieval_repair" else "legacy "
            print(f" {label}: EM={em:.4f} fixed={f} harmed={h} net={nt:+d}")


def run(args):
    records = load_records(args.input, args.n_samples)
    results = load_done(args.output) if not args.no_resume else []
    done = {str(r["id"]) for r in results}
    if results:
        print(f"Resume: {len(results)}/{len(records)} already done", flush=True)

    retriever = make_retriever(args.retriever)
    generator = make_generator(args.model)

    for i, rec in enumerate(records):
        sid = str(rec["id"])
        if sid in done:
            continue
        q = rec["question"]
        gold = rec["golden_answers"]
        print(f"[{i + 1}/{len(records)}] {q[:60]}", flush=True)

        rr = call_with_retry(lambda: retrieval_repair(
            q, retriever, generator, top_k=10, prompt_style="neutral"))
        ans = rr["repaired_answer"]

        out = {
            "id": rec["id"],
            "question": q,
            "golden_answers": gold,
            "question_type": rec.get("question_type"),
            "retriever": args.retriever,
            "model": args.model,
            "baseline": rec["baseline"],
            "retrieval_repair_legacy": rec.get("retrieval_repair"),
            "retrieval_repair": {
                "rewritten_query": rr.get("rewritten_query"),
                "pred": ans,
                "em": compute_em(ans, gold),
                "f1": compute_f1(ans, gold),
            },
        }
        results.append(out)
        if len(results) % 25 == 0:
            save(args.output, results)
            print(f"  checkpoint @ {len(results)}", flush=True)

    save(args.output, results)
    summarize(results)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="existing mini repair log (baseline + legacy retrieval_repair)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--retriever", choices=["bm25", "dense"], required=True)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--n_samples", type=int, default=None)
    ap.add_argument("--no_resume", action="store_true")
    run(ap.parse_args())