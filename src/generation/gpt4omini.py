import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set. Put it in your .env file.")

client = OpenAI(api_key=api_key)


def generate_answer(question, context, model="gpt-4o-mini"):
    if isinstance(context, list):
        context = "\n\n".join(context)

    prompt = f"""Answer the question using only the provided context.

    Rules:
    - Give the shortest possible answer span.
    - Do not write a full sentence.
    - If the context contains the answer, copy it as directly as possible.
    - Only say "I don't know" if the context truly does not contain the answer.
    - Return only the minimal answer span.
    - Do not include extra names, explanations, or surrounding details. 

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


def generate(question, context, model="gpt-4o-mini"):
    return generate_answer(question, context, model=model)