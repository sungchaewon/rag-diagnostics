import json
import os
from rank_bm25 import BM25Okapi

CORPUS_PATH = os.getenv("RAG_CORPUS_PATH", "data/corpus/nq_passage_corpus_500.jsonl")

_passages = None
_bm25 = None


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def _load_corpus():
    global _passages, _bm25

    if _passages is not None and _bm25 is not None:
        return

    passages = []
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            passages.append(item["contents"])

    tokenized_corpus = [p.lower().split() for p in passages]

    _passages = passages
    _bm25 = BM25Okapi(tokenized_corpus)


def retrieve_bm25(query: str, top_k: int = 10) -> list[str]:
    _load_corpus()

    query_tokens = query.lower().split()
    scores = _bm25.get_scores(query_tokens)

    ranked = sorted(
        zip(scores, _passages),
        key=lambda x: x[0],
        reverse=True
    )

    return [passage for _, passage in ranked[:top_k]]


def get_bm25_rank_info(query: str, golden_passage: str, top_k: int = 10) -> dict:
    """
    Return whether the gold passage appears in the BM25 top-k list
    and its 1-based rank. If not found, rank is -1.
    """
    retrieved = retrieve_bm25(query, top_k=top_k)

    norm_gold = _normalize_text(golden_passage)
    gold_rank = -1

    for idx, passage in enumerate(retrieved, start=1):
        if _normalize_text(passage) == norm_gold:
            gold_rank = idx
            break

    if gold_rank == 1:
        rank_bucket = "rank_1"
    elif 2 <= gold_rank <= 3:
        rank_bucket = "rank_2_3"
    elif 4 <= gold_rank <= top_k:
        rank_bucket = "rank_4_10"
    else:
        rank_bucket = "miss"

    return {
        "bm25_hit": gold_rank != -1,
        "bm25_gold_rank": gold_rank,
        "rank_bucket": rank_bucket,
    }
