import json
import argparse
import re
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.retrieval.bm25_baseline import retrieve_bm25
from src.generation.gpt4omini import generate as _raw_generate
from src.repairs.retrieval_repair import retrieval_repair
from src.repairs.generation_repair import generation_repair
from src.repairs.generation_repair_v2 import generation_repair_v2
from src.repairs.generation_repair_v3 import generation_repair_v3
from eval.metrics import compute_em, compute_f1

MODEL = "gpt-4o"

# rough per-1M-token USD pricing for cost estimate only
PRICE_IN = 2.5
PRICE_OUT = 10.0

_WAIT_RE = re.compile(r"try again in ([\d.]+)\s*s", re.IGNORECASE)
_MAX_RETRIES = 2000
_usage = {"in_tokens": 0, "out_tokens": 0, "calls": 0}


def _estimate_tokens(text):
    # rough heuristic: ~1.3 tokens per word, good enough for a cost
    # order-of-magnitude check, not an exact count
    return max(1, int(len(str(text).split()) * 1.3))


def generate(question, contexts, model=MODEL):
    """Retry-wrapped generate() with rough token/cost tracking."""
    import openai

    ctx_text = "\n\n".join(contexts) if isinstance(contexts, list) else contexts
    in_est = _estimate_tokens(question) + _estimate_tokens(ctx_text) + 60

    attempt = 0
    while True:
        try:
            ans = _raw_generate(question, contexts, model=model)
            _usage["in_tokens"] += in_est
            _usage["out_tokens"] += _estimate_tokens(ans)
            _usage["calls"] += 1
            return ans
        except openai.RateLimitError as e:
            attempt += 1
            if attempt > _MAX_RETRIES:
                raise
            msg = str(e)
            m = _WAIT_RE.search(msg)
            wait_s = float(m.group(1)) + 1.0 if m else min(30 * attempt, 300)
            print(f"[rate-limit] attempt {attempt}, sleeping {wait_s:.1f}s",
                  flush=True)
            time.sleep(wait_s)


def normalize_contexts(contexts):
    out = []
    for c in contexts:
        if isinstance(c, dict):
            out.append(c.get("contents") or c.get("text") or str(c))
        else:
            out.append(str(c))
    return out


def get_gold_answers(sample):
    gold = (
        sample.get("golden_answers")
        or sample.get("gold_answers")
        or sample.get("answers")
        or sample.get("answer")
    )
    if isinstance(gold, str):
        return [gold]
    return gold


class BM25Retriever:
    def retrieve(self, query, top_k=10):
        return normalize_contexts(retrieve_bm25(query, top_k=top_k))


class ModelGenerator:
    def generate(self, question, contexts):
        return generate(question, contexts, model=MODEL)


def load_samples(diagnosis_path, n_samples):
    raw = json.loads(Path(diagnosis_path).read_text())
    rows = raw["results"] if isinstance(raw, dict) and "results" in raw else raw
    if not isinstance(rows, list):
        sys.exit(f"[backbone] unexpected diagnosis structure in {diagnosis_path}")
    return rows[:n_samples] if n_samples else rows


def load_done(path):
    p = Path(path)
    if not p.exists():
        return []
    try:
        data = json.load(open(p))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save(path, results):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def run(diagnosis_path, output_path, n_samples, resume=True):
    samples = load_samples(diagnosis_path, n_samples)
    results = load_done(output_path) if resume else []
    done_ids = {str(r["id"]) for r in results}
    if results:
        print(f"Resume: {len(results)}/{len(samples)} already done", flush=True)

    retriever = BM25Retriever()
    generator = ModelGenerator()

    for i, sample in enumerate(samples):
        sid = str(sample.get("id", i))
        if sid in done_ids:
            continue

        q = sample["question"]
        gold = get_gold_answers(sample)
        qtype = sample.get("question_type", "other")

        print(f"[{i + 1}/{len(samples)}] {q[:60]}", flush=True)

        base_ctx = retriever.retrieve(q, top_k=10)
        base_answer = generate(q, base_ctx, model=MODEL)

        rr = retrieval_repair(q, retriever, generator, top_k=10)
        rr_answer = rr["repaired_answer"]

        context_str = "\n\n".join(base_ctx[:5])
        v1_answer = generation_repair(q, context_str, base_answer, model=MODEL)
        v2_answer = generation_repair_v2(q, context_str, base_answer, model=MODEL)
        v3_answer = generation_repair_v3(q, context_str, base_answer,
                                         question_type=qtype, model=MODEL)

        result = {
            "id": sid,
            "question": q,
            "golden_answers": gold,
            "question_type": qtype,
            "model": MODEL,
            "baseline": {"pred": base_answer, "em": compute_em(base_answer, gold),
                        "f1": compute_f1(base_answer, gold)},
            "retrieval_repair": {"rewritten_query": rr.get("rewritten_query"),
                                 "pred": rr_answer,
                                 "em": compute_em(rr_answer, gold),
                                 "f1": compute_f1(rr_answer, gold)},
            "generation_repair": {"pred": v1_answer, "em": compute_em(v1_answer, gold),
                                  "f1": compute_f1(v1_answer, gold)},
            "generation_repair_v2": {"pred": v2_answer, "em": compute_em(v2_answer, gold),
                                     "f1": compute_f1(v2_answer, gold)},
            "generation_repair_v3": {"pred": v3_answer, "em": compute_em(v3_answer, gold),
                                     "f1": compute_f1(v3_answer, gold)},
        }
        results.append(result)

        if len(results) % 25 == 0:
            save(output_path, results)
            cost = (_usage["in_tokens"] / 1e6 * PRICE_IN +
                   _usage["out_tokens"] / 1e6 * PRICE_OUT)
            print(f"  checkpoint @ {len(results)}  "
                  f"(est. cost so far: ${cost:.2f}, {_usage['calls']} calls)",
                  flush=True)

    save(output_path, results)
    summarize(results)


def summarize(results):
    n = len(results)
    if n == 0:
        print("no results")
        return
    actions = ["baseline", "retrieval_repair", "generation_repair",
              "generation_repair_v2", "generation_repair_v3"]
    print(f"\n=== Backbone={MODEL}, uniform application (n={n}) ===")
    for a in actions:
        em = sum(r[a]["em"] for r in results) / n
        f1 = sum(r[a]["f1"] for r in results) / n
        helped = sum(1 for r in results if r[a]["em"] > r["baseline"]["em"])
        harmed = sum(1 for r in results if r[a]["em"] < r["baseline"]["em"])
        print(f"{a:24s} EM {em:.4f}  F1 {f1:.4f}  "
              f"(+{helped} / -{harmed} vs baseline)")
    orc_em = sum(max(r[a]["em"] for a in actions) for r in results) / n
    best_uni = max(sum(r[a]["em"] for r in results) / n for a in actions)
    print(f"\nOracle routing EM {orc_em:.4f}  "
          f"(gap vs best uniform: {orc_em - best_uni:+.4f})")

    cost = (_usage["in_tokens"] / 1e6 * PRICE_IN +
           _usage["out_tokens"] / 1e6 * PRICE_OUT)
    print(f"\n[cost estimate] {_usage['calls']} calls, "
          f"~{_usage['in_tokens']:,} in / ~{_usage['out_tokens']:,} out tokens "
          f"(word-count heuristic, not exact)")
    print(f"[cost estimate] approx ${cost:.2f} USD at "
          f"${PRICE_IN}/1M in + ${PRICE_OUT}/1M out")
    print("(compare this summary directly against the mini-backbone BM25 "
          "log's summary for the same n to see if the retriever-conditional "
          "pattern holds across backbones too)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--diagnosis", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n_samples", type=int, default=300)
    ap.add_argument("--no_resume", action="store_true")
    args = ap.parse_args()
    run(args.diagnosis, args.output, args.n_samples, resume=not args.no_resume)