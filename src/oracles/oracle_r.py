from src.retrieval.bm25_baseline import retrieve_bm25


def retrieve_oracle_r(query: str, golden_passage: str, top_k: int = 10) -> list[str]:
    """
    Oracle-R:
    Retriever is oracle, so golden passage is forcibly included at rank 1.
    Remaining passages come from normal BM25 retrieval.
    """
    bm25_results = retrieve_bm25(query, top_k=top_k)
    bm25_results = [p for p in bm25_results if p != golden_passage]
    return [golden_passage] + bm25_results[: top_k - 1]