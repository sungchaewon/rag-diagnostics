import json
from rank_bm25 import BM25Okapi

CORPUS_PATH = "data/corpus/nq_passage_corpus_500.jsonl"

_passages = None
_bm25 = None


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