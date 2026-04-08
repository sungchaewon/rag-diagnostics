def build_oracle_generator_context(gold_evidence):
    """
    Oracle-G:
    Directly provide ideal supporting evidence to the generator.
    """
    if isinstance(gold_evidence, list):
        return "\n\n".join(str(x) for x in gold_evidence)
    return str(gold_evidence)