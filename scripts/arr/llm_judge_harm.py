import argparse
import json
import re
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.generation.gpt4omini import client

MODEL = "gpt-4o-mini"

JUDGE_PROMPT = """You are judging whether a candidate answer is factually \
correct given a set of accepted gold answers, ignoring surface-form \
differences.

Question: {question}
Gold answers (any one counts as correct): {gold}
Candidate answer: {pred}

Rules:
- Answer CORRECT if the candidate expresses the same fact as any gold \
answer, even if wording, abbreviation, capitalization, extra/missing \
articles, or formatting differs (e.g. "USA" vs "United States", \
"May 18, 2018" vs "18 May 2018").
- Answer INCORRECT if the candidate states a different fact, a wrong \
entity, a wrong date/number, or is empty/unresponsive (e.g. "I don't \
know" when a gold answer exists).
- Respond with exactly one word: CORRECT or INCORRECT. No explanation.
"""

_WAIT_RE = re.compile(r"try again in ([\d.]+)\s*s", re.IGNORECASE)
_MAX_RETRIES = 500


def judge(question, gold, pred):
    """Returns True if the LLM judges pred correct, False otherwise."""
    import openai

    gold_str = " | ".join(str(g) for g in gold) if isinstance(gold, list) \
        else str(gold)
    prompt = JUDGE_PROMPT.format(question=question, gold=gold_str, pred=pred)

    attempt = 0
    while True:
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=5,
            )
            verdict = resp.choices[0].message.content.strip().upper()
            return verdict.startswith("CORRECT")
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


def run(log_path, action, output_path, resume=True):
    with open(log_path) as f:
        raw = json.load(f)
    records = raw["results"] if isinstance(raw, dict) and "results" in raw \
        else raw

    harm_cases = [
        r for r in records
        if action in r and r["baseline"]["em"] == 1 and r[action]["em"] == 0
    ]
    print(f"log: {log_path}")
    print(f"action: {action}")
    print(f"harm cases found (baseline correct, {action} wrong by EM): "
          f"{len(harm_cases)}")

    results = load_done(output_path) if resume else []
    done_ids = {str(r["id"]) for r in results}
    if results:
        print(f"Resume: {len(results)}/{len(harm_cases)} already judged",
              flush=True)

    for i, rec in enumerate(harm_cases):
        rid = str(rec.get("id", i))
        if rid in done_ids:
            continue
        q = rec["question"]
        gold = rec.get("golden_answers", rec.get("gold_answers", []))
        pred = rec[action]["pred"]

        print(f"[{i + 1}/{len(harm_cases)}] {q[:60]}", flush=True)
        is_correct = judge(q, gold, pred)

        results.append({
            "id": rid,
            "question": q,
            "golden_answers": gold,
            "pred": pred,
            "em_says_harmed": True,
            "llm_judge_correct": is_correct,
        })

        if len(results) % 25 == 0:
            save(output_path, results)
            print(f"  checkpoint @ {len(results)}", flush=True)

    save(output_path, results)
    summarize(results, records, action)


def summarize(results, all_records, action):
    n = len(results)
    if n == 0:
        print("\nno harm cases to summarize")
        return

    surface_mismatch = sum(1 for r in results if r["llm_judge_correct"])
    genuine_harm = n - surface_mismatch

    print(f"\n=== LLM-judge re-scoring of {action} harm cases ===")
    print(f"total EM-harmed: {n}")
    print(f"  surface-form mismatch (LLM says CORRECT): {surface_mismatch} "
          f"({surface_mismatch / n:.1%})")
    print(f"  genuine harm (LLM says INCORRECT): {genuine_harm} "
          f"({genuine_harm / n:.1%})")

    # recompute net gain for this action under LLM-judge scoring
    fixed = sum(
        1 for r in all_records
        if action in r and r["baseline"]["em"] == 0 and r[action]["em"] == 1
    )
    net_em = fixed - n
    net_judge = fixed - genuine_harm
    print(f"\nnet gain under EM: {fixed} fixed - {n} harmed = {net_em}")
    print(f"net gain under LLM-judge: {fixed} fixed - {genuine_harm} "
          f"corrected-harmed = {net_judge}")
    print(f"(fixed count is EM-based and not re-judged here -- judging "
          f"harm cases only, since that's the count driving the uniform-"
          f"application net loss)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--action", default="generation_repair_v3")
    ap.add_argument("--output", required=True)
    ap.add_argument("--no_resume", action="store_true")
    args = ap.parse_args()
    run(args.log, args.action, args.output, resume=not args.no_resume)