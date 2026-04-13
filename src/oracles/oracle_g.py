from src.generation.gpt4omini import generate


def generate_oracle_g(question: str, golden_passage: str) -> str:
    """
    Oracle-G:
    Generator receives the ideal supporting passage directly.
    """
    return generate(question, [golden_passage])