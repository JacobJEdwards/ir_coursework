import math
from functools import lru_cache
from config import CACHE_SIZE
from typing import Sequence


@lru_cache(maxsize=CACHE_SIZE)
def calculate_tf_idf(tf: float, idf: float) -> float:
    """
    Calculate TF-IDF (Term Frequency-Inverse Document Frequency).

    Args:
    - tf (float): Term frequency in a document.
    - idf (float): Inverse document frequency of a term.

    Returns:
    - float: TF-IDF score.
    """
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
    """
    Calculate BM25 Plus (Best Matching 25) scoring.

    Args:
    - tf (float): Term frequency in a document.
    - idf (float): Inverse document frequency of a term.
    - doc_length (int): Length of the document.
    - avg_dl (float): Average document length in the collection.
    - plus (bool, optional): Whether to calculate the BM25+ score. Defaults to True.
    - k1 (float, optional): BM25 parameter. Defaults to 1.2.
    - b (float, optional): BM25 parameter. Defaults to 0.75.
    - alpha (float, optional): BM25 parameter. Defaults to 0.1
    - k2 (float, optional): BM25 parameter. Defaults to 1000

    Returns:
    - float: BM25 Plus score.
    """
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
    """
    Calculate BM25 (Best Matching 25) scoring.

    Args:
    - tf (float): Term frequency in a document.
    - idf (float): Inverse document frequency of a term.
    - doc_length (int): Length of the document.
    - avg_dl (float): Average document length in the collection.
    - plus (bool, optional): Whether to calculate the BM25+ score. Defaults to True.
    - k1 (float, optional): BM25 parameter. Defaults to 1.2.
    - b (float, optional): BM25 parameter. Defaults to 0.75.

    Returns:
    - float: BM25 score.
    """
    return (
        _calculate_bm25_plus(tf, idf, doc_length, avg_dl, k1, b)
        if plus
        else (tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_length / avg_dl))) * idf
    )


@lru_cache(maxsize=CACHE_SIZE)
def calculate_idf(total_docs: int, docs_with_term: int, *, bm25: bool = False) -> float:
    """
    Calculate IDF scoring.

    Args:
    - total_docs (int): Number of documents.
    - docs_with_term (int): Number of docs containing the term.
    - bm25 (bool, optional): Where to calculate BM25 IDF. Defaults to False.

    Returns:
    - float: IDF score.
    """
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
    """
    Calculate Term Frequency (TF) for a word in a document.

    Args:
    - total_words (int): Total number of words in the document.
    - word_count (int): Number of occurrences of a specific word.

    Returns:
    - float: Calculated TF score.
    """
    if total_words == 0:
        return 0

    return word_count / total_words


def calculate_dot(vec1: Sequence[float], vec2: Sequence[float]) -> int:
    """
    Calculate the dot product of two vectors.

    Args:
    - vec1 (Sequence[float]): First vector.
    - vec2 (Sequence[float]): Second vector.

    Returns:
    - int: Dot product of the two vectors.
    """
    return sum(a * b for a, b in zip(vec1, vec2))


def calculate_vector_norm(vec: Sequence[float]) -> float:
    """
    Calculate the Euclidean norm (magnitude) of a vector.

    Args:
    - vec (Sequence[float]): Input vector.

    Returns:
    - float: Euclidean norm of the vector.
    """
    return math.sqrt(sum(x**2 for x in vec))


def cosine_similarity(
    query_vector: Sequence[float], doc_vector: Sequence[float]
) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
    - query_vector (Sequence[float]): Vector representing the query.
    - doc_vector (Sequence[float]): Vector representing the document.

    Returns:
    - float: Cosine similarity score between the query and document vectors.
    """
    dot_product = calculate_dot(query_vector, doc_vector)

    query_norm = calculate_vector_norm(query_vector)
    doc_norm = calculate_vector_norm(doc_vector)

    if (product := query_norm * doc_norm) == 0:
        return 0

    return dot_product / product
