def generation_repair(question, context, baseline_answer):
    revised = gpt(f"""
You are given a question, retrieved context, and an initial answer produced by a RAG system.

Check whether the initial answer is directly supported by the context.
If the initial answer is correct and supported, return the same answer.
If it is incorrect, incomplete, or unsupported, revise it using only the context.
If the context does not contain enough evidence to answer the question, return "unknown".

Return only the final short answer.

Question:
{question}

Context:
{context}

Initial answer:
{baseline_answer}

Final answer:
""")
    return revised.strip()