# Harm-Aware Multi-Action Routing for Selective RAG Repair

Anonymous code release accompanying an ACL Rolling Review submission.

This repository contains the code for studying **inference-time repair of completed RAG answers**. A repair action can fix one query while corrupting another, so we frame repair as a *selective intervention* problem and train a two-stage, harm-aware router that (1) decides whether to preserve the baseline answer and (2) selects among heterogeneous repair actions.

## Overview

Given a query, retrieved contexts, and a baseline RAG answer, the router decides:

1. **Gate (preserve vs. intervene).** A logistic-regression gate trained with weighted binary cross-entropy. Baseline-success examples whose best action is to preserve the baseline receive an extra weight `h_w` (main experiments: `h_w = 5`). Repair is invoked when the gate probability is at least a threshold `tau`.
2. **Selector.** A three-way logistic-regression selector over retrieval repair, Generation V2, and Generation V3, applied only to queries admitted by the gate.

Outcomes are reported as *Fixed* (baseline wrong, repaired right), *Harmed* (baseline right, repaired wrong), and *Net* = Fixed - Harmed.

## Experimental Setting

- **Data:** Natural Questions and TriviaQA (first 1,500 qualifying validation questions each), using controlled gold-passage-only corpora.
- **Retrievers:** BM25 (`rank_bm25.BM25Okapi`) and dense retrieval (`intfloat/e5-base-v2`).
- **Generators:** GPT-4o-mini (primary) and GPT-4o (backbone comparison on NQ / BM25).
- **Metrics:** Exact Match, token-level F1, and fixed / harmed / net counts.
- **Evaluation:** five-fold stratified out-of-fold prediction over five seeds, paired-bootstrap 95% confidence intervals, McNemar / sign tests, and nested cross-validation.

## Repair Actions

| Action | Description |
|---|---|
| Retrieval repair | Rewrite the query, retrieve with rewritten and original queries, interleave and deduplicate contexts, re-answer. |
| Generation V1 | Conservative verification with exact-substring grounding (analyzed in uniform repair only). |
| Generation V2 | Evidence-first re-answering with an answer-token grounding check (>= 0.6). |
| Generation V3 | Fresh question-type-conditioned extraction without showing the baseline answer. |

The exact repair prompts are listed in the paper appendix.

## Setup

```bash
# Python 3.10+
pip install -r requirements.txt
export OPENAI_API_KEY=<your key>   # required for generation and repair calls
```

Experiments call the OpenAI API; no API keys are stored in this repository.

## Reproducing the Experiments

> TODO (fill in before release): replace each `<...>` with the actual script name and arguments.

```bash
# 1. Prepare corpora and questions
python scripts/<prepare_script>.py

# 2. Run baseline RAG and repair actions (BM25 / dense; GPT-4o-mini / GPT-4o)
python scripts/<run_repair_script>.py --retriever <bm25|dense> --dataset <nq|triviaqa>

# 3. Train and evaluate the gate + selector router
python scripts/<run_router_script>.py --hw 5

# 4. Controls, transfer, and restricted-baseline analyses
python scripts/<controls_script>.py
python scripts/reviewer_defense.py
```

## Repository Structure

```
.
├── data/              # controlled corpora and question samples
├── figures/           # figures
├── scripts/           # data preparation, experiment, and analysis scripts
├── src/
│   ├── evaluation/
│   ├── generation/
│   └── retrieval/
└── README.md
```

## Notes

- Results in the paper use temperature 0 for all generation calls.
- Dense routing features store cosine-similarity statistics in the same interface as BM25 score statistics.
- Some analysis scripts reuse the corpora and the pre-existing retrieval / Generation-V1 pipeline from earlier diagnostic experiments on the same data; those scripts are kept for completeness.
