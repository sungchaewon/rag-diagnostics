def inject_gold_passage(passages: list[str], golden_passage: str, top_k: int) -> list[str]:
    """
    Ensure golden passage exists in the reranker candidate pool.
    """
    passages = passages[:top_k]

    if golden_passage in passages:
        return passages

    if len(passages) < top_k:
        return passages + [golden_passage]

    return passages[:-1] + [golden_passage]