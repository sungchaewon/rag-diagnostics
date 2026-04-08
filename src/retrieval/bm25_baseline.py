from rank_bm25 import BM25Okapi


def retrieve_top_k(question, corpus, k=3):
    """
    BM25 baseline retrieval.
    Returns top-k documents with scores.
    """
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    scores = bm25.get_scores(question.lower().split())

    ranked = sorted(
        [{"doc": doc, "score": float(score)} for doc, score in zip(corpus, scores)],
        key=lambda x: x["score"],
        reverse=True,
    )
    return ranked[:k]