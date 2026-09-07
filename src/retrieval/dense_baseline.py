import json
import os
import argparse
from pathlib import Path

import numpy as np

CORPUS_PATH = os.getenv("RAG_CORPUS_PATH", "data/corpus/nq_passage_corpus_500.jsonl")
MODEL_NAME = os.getenv("RAG_DENSE_MODEL", "intfloat/e5-base-v2")

_model = None
_passages = None
_ids = None
_embeddings = None
_cache_path = None


def _load_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print(f"[dense] loading {MODEL_NAME} ...", flush=True)
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _cache_file_for(corpus_path):
    p = Path(corpus_path)
    return p.with_suffix("").with_suffix(f".{Path(MODEL_NAME).name}.npy")


def _load_corpus():
    """Load corpus + embeddings, using cache if present and fresh."""
    global _passages, _ids, _embeddings, _cache_path
    if _embeddings is not None:
        return

    ids, passages = [], []
    with open(CORPUS_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            ids.append(item["id"])
            passages.append(item["contents"])
    _ids, _passages = ids, passages

    cache = _cache_file_for(CORPUS_PATH)
    _cache_path = cache
    corpus_mtime = Path(CORPUS_PATH).stat().st_mtime

    if cache.exists() and cache.stat().st_mtime >= corpus_mtime:
        print(f"[dense] loading cached embeddings: {cache}", flush=True)
        arr = np.load(cache)
        if arr.shape[0] == len(passages):
            _embeddings = arr
            return
        print(f"[dense] cache size mismatch ({arr.shape[0]} vs "
              f"{len(passages)}), re-embedding", flush=True)

    model = _load_model()
    print(f"[dense] embedding {len(passages)} passages from {CORPUS_PATH} ...",
          flush=True)
    texts = [f"passage: {p}" for p in passages]
    emb = model.encode(texts, batch_size=64, show_progress_bar=True,
                       normalize_embeddings=True, convert_to_numpy=True)
    np.save(cache, emb)
    print(f"[dense] cached embeddings to {cache}", flush=True)
    _embeddings = emb


def retrieve_dense(query: str, top_k: int = 10) -> list[str]:
    _load_corpus()
    model = _load_model()
    q_emb = model.encode([f"query: {query}"], normalize_embeddings=True,
                         convert_to_numpy=True)[0]
    # embeddings are L2-normalized, so dot product = cosine similarity
    scores = _embeddings @ q_emb
    top_idx = np.argsort(-scores)[:top_k]
    return [_passages[i] for i in top_idx]


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def get_dense_rank_info(query: str, golden_passage: str, top_k: int = 10) -> dict:
    """
    Same contract as bm25_baseline.get_bm25_rank_info: whether the gold
    passage appears in the dense top-k and its 1-based rank (-1 if not
    found). Enables the same rank_bucket / gold_rank_capped features to
    be recomputed for the dense setting.
    """
    retrieved = retrieve_dense(query, top_k=top_k)
    norm_gold = _normalize_text(golden_passage)
    gold_rank = -1
    for idx, passage in enumerate(retrieved, start=1):
        if _normalize_text(passage) == norm_gold:
            gold_rank = idx
    return {"hit": gold_rank != -1, "gold_rank": gold_rank,
           "retrieved": retrieved}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build_cache", action="store_true",
                    help="embed the corpus and write the cache, then exit")
    ap.add_argument("--smoke_test", action="store_true",
                    help="after building the cache, run a few sanity queries")
    args = ap.parse_args()

    if args.build_cache or args.smoke_test:
        _load_corpus()
        print(f"[dense] corpus: {len(_passages)} passages, "
              f"embedding dim {_embeddings.shape[1]}")
        print(f"[dense] cache file: {_cache_path}")

    if args.smoke_test:
        q = _passages[0][:80]
        print(f"\n[dense] smoke test, query = passage[0] prefix: {q!r}")
        top = retrieve_dense(q, top_k=3)
        for i, p in enumerate(top, 1):
            print(f"  {i}. {p[:100]}")
        print("  (passage 0's own text should rank #1 if embeddings are sane)")