import json
import csv
from pathlib import Path

CONFIGS = [
    {
        "dataset": "nq_1500",
        "oracle_file": "outputs/cikm/nq_1500/results.json",
        "priority_file": "outputs/cikm/nq_1500/analysis/cscod_priority_by_split.csv",
        "repair_file": "outputs/cikm/nq_repair_50_v2.json",
        "out_file": "outputs/cikm/nq_repair_selector_summary.csv",
    },
    {
        "dataset": "triviaqa_1500_v4",
        "oracle_file": "outputs/cikm/triviaqa_1500_v4/results.json",
        "priority_file": "outputs/cikm/triviaqa_1500_v4/analysis/cscod_priority_by_split.csv",
        "repair_file": "outputs/cikm/triviaqa_repair_100_v2.json",
        "out_file": "outputs/cikm/triviaqa_repair_selector_summary.csv",
    },
]

def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0

def load_priority_map(path):
    priority = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            priority[(row["group"], row["split"])] = row["priority"]
    return priority

def pick_cscod(row, oracle_meta, priority_map):
    # Retrieval split has priority because BM25-Miss has a special incidence map.
    split_retrieval = oracle_meta.get("split_retrieval", row.get("split_retrieval", "unknown"))
    question_type = oracle_meta.get("question_type", row.get("question_type", "unknown"))

    retrieval_priority = priority_map.get(("split_retrieval", split_retrieval), "")
    qtype_priority = priority_map.get(("question_type", question_type), "")

    if retrieval_priority == "Access":
        return "retrieval_repair"

    if qtype_priority == "Access":
        return "retrieval_repair"

    if qtype_priority == "Extraction":
        return "generation_repair"

    # Ordering repair is not implemented in current preliminary repair files.
    # Mixed, LowGain, LowSupport, and Ordering all abstain to baseline.
    return "baseline"

def summarize_strategy(rows, strategy, oracle_by_id=None, priority_map=None):
    ems, f1s = [], []
    chosen_counts = {"baseline": 0, "retrieval_repair": 0, "generation_repair": 0}

    for row in rows:
        if strategy == "no_repair":
            chosen = "baseline"
        elif strategy == "always_retrieval_repair":
            chosen = "retrieval_repair"
        elif strategy == "always_generation_repair":
            chosen = "generation_repair"
        elif strategy == "cscod_selector":
            oracle_meta = oracle_by_id.get(str(row["id"]), {})
            chosen = pick_cscod(row, oracle_meta, priority_map)
        else:
            raise ValueError(strategy)

        chosen_counts[chosen] += 1
        ems.append(float(row[chosen]["em"]))
        f1s.append(float(row[chosen]["f1"]))

    return {
        "strategy": strategy,
        "n": len(rows),
        "em": mean(ems),
        "f1": mean(f1s),
        "chosen_baseline": chosen_counts["baseline"],
        "chosen_retrieval_repair": chosen_counts["retrieval_repair"],
        "chosen_generation_repair": chosen_counts["generation_repair"],
    }

def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {}
            for k, v in row.items():
                if isinstance(v, float):
                    out[k] = f"{v:.4f}"
                else:
                    out[k] = v
            writer.writerow(out)

def main():
    all_rows = []

    for cfg in CONFIGS:
        oracle = json.load(open(cfg["oracle_file"]))
        repair = json.load(open(cfg["repair_file"]))
        priority_map = load_priority_map(cfg["priority_file"])

        oracle_by_id = {str(r["id"]): r for r in oracle}

        rows = []
        for strategy in [
            "no_repair",
            "always_retrieval_repair",
            "always_generation_repair",
            "cscod_selector",
        ]:
            summary = summarize_strategy(
                repair,
                strategy,
                oracle_by_id=oracle_by_id,
                priority_map=priority_map,
            )
            summary["dataset"] = cfg["dataset"]
            rows.append(summary)
            all_rows.append(summary)

        write_csv(cfg["out_file"], rows)
        print(f"[OK] wrote {cfg['out_file']}")

    write_csv("outputs/cikm/repair_selector_summary_all.csv", all_rows)
    print("[OK] wrote outputs/cikm/repair_selector_summary_all.csv")

if __name__ == "__main__":
    main()
