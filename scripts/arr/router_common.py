import json
import sys
from pathlib import Path

# canonical action names; extend if a new module is added
KNOWN_ACTIONS = [
    "baseline",
    "retrieval_repair",
    "generation_repair",      # V1 (kept for uniform-baseline comparison)
    "generation_repair_v2",
    "generation_repair_v3",
]

ID_KEYS = ["qid", "id", "query_id", "example_id", "idx", "index"]


def _get_id(rec, fallback):
    for k in ID_KEYS:
        if k in rec:
            return str(rec[k])
    return str(fallback)


def _extract_actions(rec):
    """Return {action: {"em": float, "f1": float}} found in a record."""
    out = {}

    # shape 1: nested under "actions"
    node = rec.get("actions") if isinstance(rec.get("actions"), dict) else None
    if node:
        for a, v in node.items():
            if isinstance(v, dict) and ("em" in v or "f1" in v):
                out[a] = {"em": float(v.get("em", 0.0)),
                          "f1": float(v.get("f1", 0.0))}
        if out:
            return out

    # shape 2: action name as a direct key
    for a in KNOWN_ACTIONS:
        v = rec.get(a)
        if isinstance(v, dict) and ("em" in v or "f1" in v):
            out[a] = {"em": float(v.get("em", 0.0)),
                      "f1": float(v.get("f1", 0.0))}
    if out:
        return out

    # shape 3: flat keys like baseline_em / retrieval_repair_f1
    for a in KNOWN_ACTIONS:
        if f"{a}_em" in rec or f"{a}_f1" in rec:
            out[a] = {"em": float(rec.get(f"{a}_em", 0.0)),
                      "f1": float(rec.get(f"{a}_f1", 0.0))}
    return out


def load_action_scores(log_path):
    """
    Returns: dict qid -> {action: {"em": float, "f1": float}}
    Exits with a schema dump if the structure is unrecognized.
    """
    raw = json.loads(Path(log_path).read_text())

    if isinstance(raw, dict):
        for key in ("results", "examples", "records", "data"):
            if key in raw and isinstance(raw[key], list):
                records = list(enumerate(raw[key]))
                break
        else:
            # dict keyed by qid
            records = list(raw.items())
    elif isinstance(raw, list):
        records = list(enumerate(raw))
    else:
        sys.exit(f"[router_common] unsupported top-level type: {type(raw)}")

    scores = {}
    for fallback_id, rec in records:
        if not isinstance(rec, dict):
            continue
        acts = _extract_actions(rec)
        if acts:
            scores[_get_id(rec, fallback_id)] = acts

    if not scores:
        first = records[0][1] if records else {}
        sys.exit(
            "[router_common] could not find per-action EM/F1.\n"
            f"first record keys: {list(first.keys()) if isinstance(first, dict) else first}\n"
            "fix _extract_actions() in scripts/arr/router_common.py"
        )
    return scores


def metric_table(scores, actions=None):
    """Mean EM/F1 per action + oracle routing over the given actions."""
    if actions is None:
        actions = sorted({a for v in scores.values() for a in v})
    n = len(scores)
    table = {}
    for a in actions:
        ems = [v[a]["em"] for v in scores.values() if a in v]
        f1s = [v[a]["f1"] for v in scores.values() if a in v]
        if ems:
            table[a] = {"em": sum(ems) / len(ems),
                        "f1": sum(f1s) / len(f1s),
                        "n": len(ems)}
    oracle_em = sum(max(v[a]["em"] for a in v) for v in scores.values()) / n
    oracle_f1 = sum(max(v[a]["f1"] for a in v) for v in scores.values()) / n
    table["oracle_routing"] = {"em": oracle_em, "f1": oracle_f1, "n": n}
    return table