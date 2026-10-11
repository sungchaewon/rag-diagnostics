import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.retrieval.bm25_baseline import retrieve_bm25

RESULT_PATH = "outputs/results.json"


def print_case(title: str, cases: list[dict]):
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)
    for x in cases:
        print("-" * 120)
        print("Q:", x["question"])
        print("Type:", x.get("question_type", "other"))
        print("Gold:", x["golden_answers"])
        print("B :", x["baseline"]["pred"], x["baseline"]["em"], x["baseline"]["f1"])
        print("R :", x["oracle_r"]["pred"], x["oracle_r"]["em"], x["oracle_r"]["f1"])
        print("RE:", x["oracle_re"]["pred"], x["oracle_re"]["em"], x["oracle_re"]["f1"])
        print("G :", x["oracle_g"]["pred"], x["oracle_g"]["em"], x["oracle_g"]["f1"])


with open(RESULT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# 1) retrieval miss + oracle_re recovery
miss_recovery = []
for x in data:
    retrieved = retrieve_bm25(x["question"], top_k=10)
    if x["golden_passage"] not in retrieved and x["oracle_re"]["f1"] > x["baseline"]["f1"]:
        miss_recovery.append(x)

# 2) BM25-hit but baseline still fails
hit_but_fail = []
for x in data:
    retrieved = retrieve_bm25(x["question"], top_k=10)
    if x["golden_passage"] in retrieved and x["baseline"]["em"] == 0:
        hit_but_fail.append(x)

# 3) oracle-resistant count-like cases
oracle_resistant = []
for x in data:
    if x.get("question_type") == "how_many":
        if (
            x["baseline"]["em"] == 0
            and x["oracle_r"]["em"] == 0
            and x["oracle_re"]["em"] == 0
            and x["oracle_g"]["em"] == 0
        ):
            oracle_resistant.append(x)

# 4) all-IDK hard failures
all_idk = []
for x in data:
    preds = [
        x["baseline"]["pred"].lower(),
        x["oracle_r"]["pred"].lower(),
        x["oracle_re"]["pred"].lower(),
        x["oracle_g"]["pred"].lower(),
    ]
    if all("i don't know" in p for p in preds):
        all_idk.append(x)

print_case("1) Retrieval miss + Oracle-Re recovery", miss_recovery[:5])
print_case("2) BM25-hit but baseline still fails", hit_but_fail[:5])
print_case("3) Oracle-resistant how_many cases", oracle_resistant[:5])
print_case("4) All-IDK hard failures", all_idk[:5])