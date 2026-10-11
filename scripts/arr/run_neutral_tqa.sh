#!/bin/bash
set -e
cd ~/chaewon/cikm26/rag-diagnostics
export RAG_CORPUS_PATH=data/corpus/triviaqa_passage_corpus_1500_short_v4.jsonl

python scripts/arr/run_neutral_rewrite.py --retriever bm25 \
  --input outputs/arr/triviaqa_repair_1500_v5.json \
  --output outputs/arr/tqa_rewrite_neutral_bm25.json --n_samples 1500

python scripts/arr/run_neutral_rewrite.py --retriever dense \
  --input outputs/arr/tqa_repair_1500_dense.json \
  --output outputs/arr/tqa_rewrite_neutral_dense.json --n_samples 1500
