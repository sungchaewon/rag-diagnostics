def retrieval_repair(question, retriever, generator, top_k=10):
    rewritten_query = gpt(f"""
Rewrite the question into a search query that is likely to retrieve passages containing the answer.
Keep the query concise and include important entities.
Return only the rewritten query.

Question:
{question}

Search query:
""").strip()

    new_contexts = retriever.retrieve(rewritten_query, top_k=top_k)

    repaired_answer = generator.generate(question, new_contexts)

    return {
        "rewritten_query": rewritten_query,
        "retrieved_contexts": new_contexts,
        "repaired_answer": repaired_answer,
    }