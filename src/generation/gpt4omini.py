import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set. Put it in your .env file.")

client = OpenAI(api_key=api_key)


def generate_answer(question, context, model="gpt-4o-mini"):
    prompt = f"""Answer the question using the provided context only.
If the answer is not supported by the context, say "I don't know".

Question: {question}

Context:
{context}

Answer:"""

    response = client.responses.create(
        model=model,
        input=prompt,
    )

    if hasattr(response, "output_text") and response.output_text:
        return response.output_text.strip()

    return str(response)