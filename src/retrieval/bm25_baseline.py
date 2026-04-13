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

def retrieve_oracle_r(query: str, 
                      golden_passage: str, 
                      top_k: int = 10) -> list[str]:
    """
    Oracle-R: BM25 결과에 golden passage를 강제로 포함
    golden passage를 첫 번째로 넣고 나머지를 BM25 결과로 채움
    """
    # BM25로 top_k 검색
    bm25_results = retrieve_bm25(query, top_k=top_k-1)
    
    # golden passage를 맨 앞에 강제 삽입
    oracle_results = [golden_passage] + bm25_results
    
    return oracle_results