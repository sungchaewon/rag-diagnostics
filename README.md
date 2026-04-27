# rag-diagnostics

Oracle-based diagnosis for modular Retrieval-Augmented Generation (RAG) pipelines.

This repository contains the experimental code for a KDD Undergraduate Consortium submission on diagnosing bottlenecks in multi-stage RAG pipelines. Instead of evaluating a RAG system only with end-to-end answer quality, this project applies stage-wise oracle interventions and split-based analysis to identify where failures occur.

## Overview

Retrieval-Augmented Generation (RAG) systems commonly consist of multiple stages, including retrieval, reranking, and generation. However, aggregate metrics such as Exact Match (EM) and token-level F1 do not directly reveal whether failures are caused by missing evidence, poor ranking, or generation errors.

This project investigates a modular RAG pipeline through oracle-based interventions. By injecting gold evidence at different stages, we analyze how much each component affects final answer quality. In addition to average oracle gains, we further examine subset-level results by retrieval outcome and question type to reveal hidden bottlenecks that may be obscured by aggregate performance.

![Oracle-based diagnosis overview](figures/intro3.png)

## Experimental Setting

- Dataset: Natural Questions
- Retriever: BM25
- Reranker: BAAI/bge-reranker-v2-m3
- Generator: GPT-4o-mini
- Metrics: Exact Match (EM), token-level F1

## Pipeline

The baseline pipeline follows a standard multi-stage RAG structure:

1. Retrieve candidate passages using BM25.
2. Rerank retrieved passages using a cross-encoder reranker.
3. Generate a short answer using GPT-4o-mini based on the provided context.
4. Evaluate the generated answer against the gold answer.

## Oracle Interventions

The following experimental conditions are used to diagnose stage-level bottlenecks.

### Baseline

A standard RAG pipeline without oracle intervention.

### Oracle-R

The gold supporting passage is forced into the retrieved candidate set.

This condition tests whether retrieval failure is a major bottleneck. If performance improves substantially under Oracle-R, it suggests that the retriever often fails to include the necessary evidence.

### Oracle-Re

The gold supporting passage is forced into the candidate set and ranked at the top.

This condition tests whether reranking quality affects downstream answer generation. If Oracle-Re improves over Oracle-R, it suggests that the correct evidence may be present but not ranked highly enough to guide the generator effectively.

### Oracle-G

The gold supporting passage is directly provided to the generator.

This condition tests the remaining generation-side bottleneck when the correct evidence is available. If performance remains limited even under Oracle-G, it suggests that the generator may still fail due to answer formatting, extraction errors, or ambiguity in the evidence.

### Oracle-All

An optional upper-bound setting where all relevant oracle information is provided.

## Analysis

The experiments are designed to answer the following questions:

- How much does each stage-wise oracle intervention improve end-to-end performance?
- Do average oracle gains obscure heterogeneous subset-level bottlenecks?
- How does the dominant bottleneck vary by question type?
- How does the bottleneck vary with retrieval outcome?

To answer these questions, the project analyzes both overall performance and split-level performance.

### Retrieval Outcome Split

Questions are divided based on whether BM25 retrieves the gold passage within the top-k candidates.

- BM25-Hit: the gold passage is included in the retrieved candidates.
- BM25-Miss: the gold passage is not included in the retrieved candidates.

This split helps distinguish cases where the pipeline fails despite retrieving the correct evidence from cases where retrieval failure is the primary bottleneck.

### Question Type Split

Questions are also grouped by question type, such as:

- who
- when
- where
- what
- how many

This analysis helps identify whether different types of questions depend more heavily on retrieval, reranking, or generation quality.

## Repository Structure

```text
rag-diagnostics/
├── data/
│   ├── corpus/
│   └── nq_sample/
├── figures/
│   └── intro3.png
├── scripts/
│   ├── add_labels.py
│   ├── analyze_hit_miss.py
│   ├── analyze_question_type.py
│   ├── check_retrieval.py
│   ├── extract_cases.py
│   ├── prepare_mini_corpus_500.py
│   ├── prepare_nq_500.py
│   ├── run_experiments.py
│   └── verify_miss_baseline.py
├── src/
│   ├── evaluation/
│   ├── generation/
│   └── retrieval/
└── README.md
```

## Current Status

This repository currently supports the initial experimental setup for the KDD UC version of the study. The main focus is on validating stage-wise oracle interventions, analyzing split-level bottlenecks, and preparing results for paper writing.

## Notes

This project is intended for research and diagnostic analysis rather than deployment. Experimental results may vary depending on dataset sampling, corpus construction, retrieval depth, reranking configuration, and generation settings.