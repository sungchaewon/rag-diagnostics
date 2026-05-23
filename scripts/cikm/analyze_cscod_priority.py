import json
import csv
from pathlib import Path
from collections import defaultdict

N_MIN = 30
GAMMA = 0.03
EPSILON = 0.02

DATASETS = {
    "nq_1500": "outputs/cikm/nq_1500/results.json",
    "triviaqa_1500_v4": "outputs/cikm/triviaqa_1500_v4/results.json",
}

GROUP_COLS = [
    "split_retrieval",
    "rank_bucket",
    "question_type",
    "split_answer_type",
]

def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0

def metric(row, stage, name):
    return float(row[stage].get(name, 0.0))

def decide_priority(n, d_r, d_re, d_g, group_col, split_value):
    is_miss_split = (
        split_value in {"BM25-Miss", "miss"}
        or (group_col == "split_retrieval" and split_value == "BM25-Miss")
        or (group_col == "rank_bucket" and split_value == "miss")
    )

    if is_miss_split:
        scores = {
            "Access": max(d_r, d_re, d_g)
        }
    else:
        scores = {
            "Access": d_r,
            "Ordering": d_re,
            "Extraction": d_g,
        }

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_label, best_score = sorted_scores[0]

    if len(sorted_scores) == 1:
        margin = float("inf")
    else:
        margin = best_score - sorted_scores[1][1]

    if n < N_MIN:
        priority = "LowSupport"
    elif best_score <= GAMMA:
        priority = "LowGain"
    elif margin <= EPSILON:
        priority = "Mixed"
    else:
        priority = best_label

    if priority in {"Access", "Ordering", "Extraction"}:
        if margin == float("inf"):
            rho = best_score - GAMMA
        else:
            rho = min(best_score - GAMMA, (margin - EPSILON) / 2)
    else:
        rho = ""

    return best_label, best_score, margin, priority, rho

def summarize_group(rows, dataset_name, group_col, split_value):
    n = len(rows)

    base_em = mean([metric(r, "baseline", "em") for r in rows])
    r_em = mean([metric(r, "oracle_r", "em") for r in rows])
    re_em = mean([metric(r, "oracle_re", "em") for r in rows])
    g_em = mean([metric(r, "oracle_g", "em") for r in rows])

    base_f1 = mean([metric(r, "baseline", "f1") for r in rows])
    r_f1 = mean([metric(r, "oracle_r", "f1") for r in rows])
    re_f1 = mean([metric(r, "oracle_re", "f1") for r in rows])
    g_f1 = mean([metric(r, "oracle_g", "f1") for r in rows])

    d_r_em = r_em - base_em
    d_re_em = re_em - base_em
    d_g_em = g_em - base_em

    d_r_f1 = r_f1 - base_f1
    d_re_f1 = re_f1 - base_f1
    d_g_f1 = g_f1 - base_f1

    best_label, best_score, margin, priority, rho = decide_priority(
        n, d_r_em, d_re_em, d_g_em, group_col, split_value
    )

    return {
        "dataset": dataset_name,
        "group": group_col,
        "split": split_value,
        "n": n,
        "base_em": base_em,
        "oracle_r_em": r_em,
        "oracle_re_em": re_em,
        "oracle_g_em": g_em,
        "delta_r_em": d_r_em,
        "delta_re_em": d_re_em,
        "delta_g_em": d_g_em,
        "base_f1": base_f1,
        "oracle_r_f1": r_f1,
        "oracle_re_f1": re_f1,
        "oracle_g_f1": g_f1,
        "delta_r_f1": d_r_f1,
        "delta_re_f1": d_re_f1,
        "delta_g_f1": d_g_f1,
        "best_label": best_label,
        "best_score": best_score,
        "margin": margin,
        "priority": priority,
        "rho": rho,
    }

def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {}
            for k, v in row.items():
                if isinstance(v, float):
                    if v == float("inf"):
                        out[k] = "inf"
                    else:
                        out[k] = f"{v:.4f}"
                else:
                    out[k] = v
            writer.writerow(out)

def main():
    all_rows = []

    for dataset_name, file_path in DATASETS.items():
        data = json.load(open(file_path))

        analysis_dir = Path(file_path).parent / "analysis"
        dataset_rows = []

        # overall
        dataset_rows.append(summarize_group(data, dataset_name, "overall", "all"))

        # split groups
        for group_col in GROUP_COLS:
            groups = defaultdict(list)
            for row in data:
                value = row.get(group_col, "unknown")
                if value is None or value == "":
                    value = "unknown"
                groups[value].append(row)

            for split_value, rows in sorted(groups.items(), key=lambda x: str(x[0])):
                dataset_rows.append(summarize_group(rows, dataset_name, group_col, split_value))

        write_csv(analysis_dir / "cscod_priority_by_split.csv", dataset_rows)
        all_rows.extend(dataset_rows)

        print(f"[OK] {dataset_name}: wrote {analysis_dir / 'cscod_priority_by_split.csv'}")

    write_csv(Path("outputs/cikm/cscod_priority_by_split_all.csv"), all_rows)
    print("[OK] wrote outputs/cikm/cscod_priority_by_split_all.csv")

if __name__ == "__main__":
    main()
