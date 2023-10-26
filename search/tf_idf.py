import math
from functools import lru_cache
from config import CACHE_SIZE


@lru_cache(maxsize=CACHE_SIZE)
def calculate_tf_idf(tf: float, idf: float) -> float:
    return tf * idf


@lru_cache(maxsize=CACHE_SIZE)
def calculate_idf(total_docs: int, docs_with_term: int) -> float:
    return math.log(total_docs / (docs_with_term + 1))


@lru_cache(maxsize=CACHE_SIZE)
def calculate_tf(total_words: int, word_count: int) -> float:
    return word_count / total_words
