import math
from functools import lru_cache
from config import CACHE_SIZE
import numpy as np


@lru_cache(maxsize=CACHE_SIZE)
def calculate_tf_idf(tf: float, idf: float) -> float:
    return tf * idf


@lru_cache(maxsize=CACHE_SIZE)
def _calculate_bm25_plus(
    tf: float,
    idf: float,
    doc_length: int,
    avg_dl: float,
    tf_query: float,
    k1: float = 1.2,
    b: float = 0.75,
    alpha: float = 0.1,
    k2: float = 1000,
) -> float:
    # Calculate the standard BM25 score
    bm25_score = calculate_bm25(tf, idf, doc_length, avg_dl, k1=k1, b=b, plus=False)

    # Calculate the BM25+ additional term
    additional_term = alpha * (tf_query * (k2 + 1)) / (tf_query + k2) * idf

    # Calculate the BM25+ score by adding the additional term to the standard BM25 score
    bm25_plus_score = bm25_score + additional_term

    return bm25_plus_score


@lru_cache(maxsize=CACHE_SIZE)
def calculate_bm25(
    tf: float,
    idf: float,
    doc_length: int,
    avg_dl: float,
    plus: bool = True,
    k1: float = 1.2,
    b: float = 0.75,
) -> float:
    return (
        _calculate_bm25_plus(tf, idf, doc_length, avg_dl, k1, b)
        if plus
        else (tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_length / avg_dl))) * idf
    )


@lru_cache(maxsize=CACHE_SIZE)
def calculate_idf(total_docs: int, docs_with_term: int, bm25: bool = False) -> float:
    if bm25:
        return math.log(
            1 + (total_docs - docs_with_term + 0.5) / (docs_with_term + 0.5)
        )

    return math.log(total_docs / (docs_with_term + 1))


@lru_cache(maxsize=CACHE_SIZE)
def calculate_tf(
    total_words: int,
    word_count: int,
) -> float:
    return word_count / total_words


def calculate_dot(vec1: np.ndarray, vec2: np.ndarray) -> int:
    return sum(a * b for a, b in zip(vec1, vec2))


def calculate_vector_norm(vec: np.ndarray) -> float:
    return math.sqrt(sum(x * x for x in vec))


# DO NOT USE NP.DOT -> do this manually
def cosine_similarity(query_vector: np.ndarray, doc_vector: np.ndarray) -> float:
    dot_product = calculate_dot(query_vector, doc_vector.T)

    query_norm = calculate_vector_norm(query_vector)
    doc_norm = calculate_vector_norm(doc_vector)

    if (product := query_norm * doc_norm) == 0:
        return 0

    return dot_product / product
