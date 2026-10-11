import json
import re

path = "outputs/cikm/triviaqa_repair_100_v2.json"
with open(path) as f:
    data = json.load(f)

def normalize(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokens(text):
    return normalize(text).split()

def is_unknown(pred):
    p = normalize(pred)
    return p in {
        "i don t know",
        "unknown",
        "not enough information",
        "cannot be determined",
    }

def token_overlap(a, b):
    ta = set(tokens(a))
    tb = set(tokens(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(tb)

def is_question_echo(question, pred):
    q_tokens = set(tokens(question))
    p_tokens = tokens(pred)

    if not p_tokens:
        return False

    overlap = len(set(p_tokens) & q_tokens) / len(set(p_tokens))

    # "Miss Greenwich Village"처럼 baseline 답이 질문 일부를 그대로 반복하는 경우
    return overlap >= 0.8 and len(p_tokens) >= 2

def choose(ex):
    q = ex["question"]
    b = ex["baseline"]
    rr = ex["retrieval_repair"]
    gr = ex["generation_repair"]

    b_pred = b["pred"]
    rr_pred = rr["pred"]
    gr_pred = gr["pred"]

    # Rule 1: baseline이 질문 일부를 그대로 반복하면 generation repair 선택
    if is_question_echo(q, b_pred) and not is_unknown(gr_pred):
        return gr, "generation_fix_question_echo"

    # Rule 2: non-unknown baseline을 unknown repair로 바꾸지 않음
    if is_unknown(rr_pred) and not is_unknown(b_pred):
        rr_candidate = None
    else:
        rr_candidate = rr

    if is_unknown(gr_pred) and not is_unknown(b_pred):
        gr_candidate = None
    else:
        gr_candidate = gr

    b_len = len(tokens(b_pred))
    rr_len = len(tokens(rr_pred))
    gr_len = len(tokens(gr_pred))

    rr_overlap = token_overlap(b_pred, rr_pred)
    gr_overlap = token_overlap(b_pred, gr_pred)

    # Rule 3: generation repair가 더 짧은 span으로 정리하면 선택
    if gr_candidate and gr_len < b_len and gr_overlap >= 0.75 and not is_unknown(gr_pred):
        return gr, "generation_trim_verbose"

    # Rule 4: retrieval repair가 더 짧은 span으로 정리하면 선택
    if rr_candidate and rr_len < b_len and rr_overlap >= 0.75 and not is_unknown(rr_pred):
        return rr, "retrieval_trim_verbose"

    # Rule 5: baseline이 unknown이면 repair 후보 사용
    if is_unknown(b_pred):
        if gr_candidate and not is_unknown(gr_pred):
            return gr, "generation_fix_unknown"
        if rr_candidate and not is_unknown(rr_pred):
            return rr, "retrieval_fix_unknown"

    return b, "baseline_default"

def avg(rows, key):
    return sum(x[key] for x in rows) / len(rows)

selected = []
reasons = {}

for ex in data:
    chosen, reason = choose(ex)
    selected.append(chosen)
    reasons[reason] = reasons.get(reason, 0) + 1

for name in ["baseline", "retrieval_repair", "generation_repair"]:
    rows = [ex[name] for ex in data]
    print(f"{name:24s} | EM: {avg(rows, 'em'):.4f} | F1: {avg(rows, 'f1'):.4f}")

print(f"{'heuristic_selector_v3':24s} | EM: {avg(selected, 'em'):.4f} | F1: {avg(selected, 'f1'):.4f}")
print("reasons:", reasons)
