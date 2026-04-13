from sentence_transformers import CrossEncoder

model = CrossEncoder("BAAI/bge-reranker-v2-m3")

def rerank(query: str, passages: list[str], top_k: int = 5) -> list[str]:
    """일반 reranking"""
    pairs = [[query, p] for p in passages]
    scores = model.predict(pairs)
    ranked = sorted(zip(scores, passages), reverse=True)
    return [p for _, p in ranked[:top_k]]

def rerank_oracle_re(query: str, 
                     passages: list[str], 
                     golden_passage: str,
                     top_k: int = 5) -> list[str]:
    """
    Oracle-Re: reranker 결과에서 golden passage를 1위로 강제 배치
    나머지는 실제 reranker 점수대로
    """
    # golden 제외한 나머지 reranking
    others = [p for p in passages if p != golden_passage]
    reranked_others = rerank(query, others, top_k=top_k-1)
    
    # golden을 1위로
    return [golden_passage] + reranked_others