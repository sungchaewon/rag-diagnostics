from sentence_transformers import CrossEncoder

model = CrossEncoder("BAAI/bge-reranker-v2-m3")


def rerank(query: str, passages: list[str], top_k: int = 5) -> list[str]:
    pairs = [[query, p] for p in passages]
    scores = model.predict(pairs)
    ranked = sorted(zip(scores, passages), key=lambda x: x[0], reverse=True)
    return [p for _, p in ranked[:top_k]]


def rerank_oracle_re(query: str, passages: list[str], golden_passage: str, top_k: int = 5) -> list[str]:
    """
    Oracle-Re:
    Candidate pool is normal retrieval result + injected golden passage.
    Reranker is oracle, so golden passage is forced to rank 1.
    """
    others = [p for p in passages if p != golden_passage]
    reranked_others = rerank(query, others, top_k=top_k - 1)
    return [golden_passage] + reranked_others