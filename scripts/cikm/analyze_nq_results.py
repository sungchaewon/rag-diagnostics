import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


INPUT_PATH = Path("outputs/cikm/nq_1500/results.json")
OUTPUT_DIR = Path("outputs/cikm/nq_1500/analysis")

CONDITIONS = ["baseline", "oracle_r", "oracle_re", "oracle_g"]


def load_results(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def avg_metric(items: list[dict], condition: str, metric: str) -> float:
    if not items:
        return 0.0
    return sum(x[condition][metric] for x in items) / len(items)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {path}")


def make_summary_rows(data: list[dict], group_key: str, group_name: str) -> list[dict]:
    groups = defaultdict(list)

    if group_key == "__overall__":
        groups["overall"] = data
    else:
        for x in data:
            groups[x[group_key]].append(x)

    rows = []

    for group_value, items in sorted(groups.items(), key=lambda kv: (-len(kv[1]), str(kv[0]))):
        baseline_em = avg_metric(items, "baseline", "em")
        baseline_f1 = avg_metric(items, "baseline", "f1")

        for cond in CONDITIONS:
            em = avg_metric(items, cond, "em")
            f1 = avg_metric(items, cond, "f1")

            rows.append(
                {
                    "group_name": group_name,
                    "group_value": group_value,
                    "n": len(items),
                    "condition": cond,
                    "em": round(em, 4),
                    "f1": round(f1, 4),
                    "em_gain_vs_baseline": round(em - baseline_em, 4),
                    "f1_gain_vs_baseline": round(f1 - baseline_f1, 4),
                }
            )

    return rows


def get_max_performance_stage(sample: dict) -> str:
    """
    Stage label based on actual best F1/EM among baseline and oracle conditions.
    If tied, choose baseline first, then weaker oracle intervention.
    """
    priority = {
        "baseline": 3,
        "oracle_r": 2,
        "oracle_re": 1,
        "oracle_g": 0,
    }

    return max(
        CONDITIONS,
        key=lambda cond: (
            sample[cond]["f1"],
            sample[cond]["em"],
            priority[cond],
        ),
    )


def rule_diagnostic_minimal(sample: dict) -> str:
    """
    Diagnosis-oriented rule:
    choose the smallest intervention likely to expose the bottleneck.
    """
    if sample["split_retrieval"] == "BM25-Miss":
        return "oracle_r"

    if sample["rank_bucket"] == "rank_4_10":
        return "oracle_re"

    if sample["question_type"] in {"when", "where", "how_many"}:
        return "oracle_g"

    return "baseline"


def rule_performance_oriented(sample: dict) -> str:
    """
    Performance-oriented rule:
    choose the intervention expected to maximize EM/F1 based on NQ split trends.
    """
    if sample["rank_bucket"] == "miss":
        return "oracle_g"

    if sample["rank_bucket"] == "rank_4_10":
        return "oracle_re"

    if sample["rank_bucket"] == "rank_2_3":
        return "oracle_g"

    if sample["question_type"] in {"when", "where", "how_many"}:
        return "oracle_g"

    return "baseline"


def evaluate_strategy(data: list[dict], name: str, selector) -> dict:
    selected_stages = []
    em_values = []
    f1_values = []
    minimal_label_hits = 0
    max_label_hits = 0

    for x in data:
        stage = selector(x)
        selected_stages.append(stage)

        em_values.append(x[stage]["em"])
        f1_values.append(x[stage]["f1"])

        if stage == x["best_oracle_stage"]:
            minimal_label_hits += 1

        if stage == get_max_performance_stage(x):
            max_label_hits += 1

    stage_counts = Counter(selected_stages)
    n = len(data)

    return {
        "strategy": name,
        "n": n,
        "em": round(sum(em_values) / n, 4),
        "f1": round(sum(f1_values) / n, 4),
        "accuracy_vs_minimal_label": round(minimal_label_hits / n, 4),
        "accuracy_vs_max_performance_label": round(max_label_hits / n, 4),
        "baseline_count": stage_counts.get("baseline", 0),
        "oracle_r_count": stage_counts.get("oracle_r", 0),
        "oracle_re_count": stage_counts.get("oracle_re", 0),
        "oracle_g_count": stage_counts.get("oracle_g", 0),
    }


def evaluate_random_oracle(data: list[dict], runs: int = 100, seed: int = 42) -> dict:
    rng = random.Random(seed)
    rows = []

    for _ in range(runs):
        rows.append(
            evaluate_strategy(
                data,
                "random_oracle",
                lambda x: rng.choice(["oracle_r", "oracle_re", "oracle_g"]),
            )
        )

    return {
        "strategy": f"random_oracle_mean_{runs}",
        "n": len(data),
        "em": round(sum(r["em"] for r in rows) / runs, 4),
        "f1": round(sum(r["f1"] for r in rows) / runs, 4),
        "accuracy_vs_minimal_label": round(sum(r["accuracy_vs_minimal_label"] for r in rows) / runs, 4),
        "accuracy_vs_max_performance_label": round(sum(r["accuracy_vs_max_performance_label"] for r in rows) / runs, 4),
        "baseline_count": 0,
        "oracle_r_count": "",
        "oracle_re_count": "",
        "oracle_g_count": "",
    }


def make_stage_distribution_rows(data: list[dict], group_key: str, group_name: str) -> list[dict]:
    groups = defaultdict(list)

    if group_key == "__overall__":
        groups["overall"] = data
    else:
        for x in data:
            groups[x[group_key]].append(x)

    rows = []

    for group_value, items in sorted(groups.items(), key=lambda kv: (-len(kv[1]), str(kv[0]))):
        counter = Counter(x["best_oracle_stage"] for x in items)
        n = len(items)

        for stage in CONDITIONS:
            count = counter.get(stage, 0)
            rows.append(
                {
                    "group_name": group_name,
                    "group_value": group_value,
                    "n": n,
                    "stage": stage,
                    "count": count,
                    "ratio": round(count / n, 4),
                }
            )

    return rows


def main():
    data = load_results(INPUT_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(data)} samples from {INPUT_PATH}")

    summary_fields = [
        "group_name",
        "group_value",
        "n",
        "condition",
        "em",
        "f1",
        "em_gain_vs_baseline",
        "f1_gain_vs_baseline",
    ]

    write_csv(
        OUTPUT_DIR / "overall_summary.csv",
        make_summary_rows(data, "__overall__", "overall"),
        summary_fields,
    )

    write_csv(
        OUTPUT_DIR / "rank_bucket_summary.csv",
        make_summary_rows(data, "rank_bucket", "rank_bucket"),
        summary_fields,
    )

    write_csv(
        OUTPUT_DIR / "retrieval_split_summary.csv",
        make_summary_rows(data, "split_retrieval", "split_retrieval"),
        summary_fields,
    )

    write_csv(
        OUTPUT_DIR / "question_type_summary.csv",
        make_summary_rows(data, "question_type", "question_type"),
        summary_fields,
    )

    predictor_rows = [
        evaluate_strategy(data, "always_baseline", lambda x: "baseline"),
        evaluate_strategy(data, "always_oracle_r", lambda x: "oracle_r"),
        evaluate_strategy(data, "always_oracle_re", lambda x: "oracle_re"),
        evaluate_strategy(data, "always_oracle_g", lambda x: "oracle_g"),
        evaluate_random_oracle(data, runs=100, seed=42),
        evaluate_strategy(data, "rule_diagnostic_minimal", rule_diagnostic_minimal),
        evaluate_strategy(data, "rule_performance_oriented", rule_performance_oriented),
    ]

    predictor_fields = [
        "strategy",
        "n",
        "em",
        "f1",
        "accuracy_vs_minimal_label",
        "accuracy_vs_max_performance_label",
        "baseline_count",
        "oracle_r_count",
        "oracle_re_count",
        "oracle_g_count",
    ]

    write_csv(
        OUTPUT_DIR / "predictor_eval.csv",
        predictor_rows,
        predictor_fields,
    )

    stage_rows = []
    stage_rows += make_stage_distribution_rows(data, "__overall__", "overall")
    stage_rows += make_stage_distribution_rows(data, "rank_bucket", "rank_bucket")
    stage_rows += make_stage_distribution_rows(data, "split_retrieval", "split_retrieval")
    stage_rows += make_stage_distribution_rows(data, "question_type", "question_type")

    stage_fields = [
        "group_name",
        "group_value",
        "n",
        "stage",
        "count",
        "ratio",
    ]

    write_csv(
        OUTPUT_DIR / "stage_distribution.csv",
        stage_rows,
        stage_fields,
    )

    print("\nPredictor evaluation:")
    for row in predictor_rows:
        print(
            f"{row['strategy']:28s} | "
            f"EM: {row['em']:.4f} | "
            f"F1: {row['f1']:.4f} | "
            f"acc_min: {row['accuracy_vs_minimal_label']:.4f} | "
            f"acc_max: {row['accuracy_vs_max_performance_label']:.4f}"
        )


if __name__ == "__main__":
    main()