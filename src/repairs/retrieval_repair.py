from src.generation.gpt4omini import gpt


def add_ctx(out, seen, x):
    key = str(x)
    if key in seen:
        return
    seen.add(key)
    out.append(x)


def mix_ctx(a, b, k):
    out = []
    seen = set()

    for i in range(max(len(a), len(b))):
        if i < len(a):
            add_ctx(out, seen, a[i])
        if i < len(b):
            add_ctx(out, seen, b[i])
        if len(out) >= k:
            break

    return out[:k]


def retrieval_repair(question, retriever, generator, top_k=10):
    q2 = gpt(f"""
Rewrite the question as a BM25 keyword query.

Rules:
- Keep the original meaning and answer type.
- Keep names, titles, places, dates, and important nouns from the question.
- Do not add any date, year, entity, or fact that is not in the question.
- Remove unnecessary question words when possible.
- Do not answer the question.
- Return only one search query.

Question:
{question}

Search query:
""").strip()

    ctx1 = retriever.retrieve(question, top_k=top_k)
    ctx2 = retriever.retrieve(q2, top_k=top_k)

    ctx = mix_ctx(ctx2, ctx1, top_k)
    ans = generator.generate(question, ctx)

    return {
        "rewritten_query": q2,
        "retrieved_contexts": ctx,
        "repaired_answer": ans,
    }
