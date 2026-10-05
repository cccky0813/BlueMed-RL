"""本地 m3e-base 句向量：用于实体链接和多跳剪枝。带缓存，同一文本只算一次。"""

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("moka-ai/m3e-base")


_cache: dict[str, np.ndarray] = {}


def embed(texts: list[str]) -> np.ndarray:
    """返回归一化后的向量矩阵（每行一个文本），余弦相似度 = 点积。"""
    missing = [t for t in dict.fromkeys(texts) if t not in _cache]
    if missing:
        for text, vec in zip(missing, _model().encode(missing, batch_size=64, normalize_embeddings=True)):
            _cache[text] = vec
    return np.stack([_cache[t] for t in texts])


def similarity(query: str, candidates: list[str]) -> list[float]:
    """query 与每个候选的余弦相似度。"""
    if not candidates:
        return []
    vecs = embed([query] + candidates)
    return (vecs[1:] @ vecs[0]).tolist()
