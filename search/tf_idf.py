import math


def calculate_tf_idf(tf: float, idf: float) -> float:
    return tf * idf


def calculate_idf(total_docs: int, docs_with_term: int) -> float:
    return math.log(total_docs / (docs_with_term + 1))


def calculate_tf(total_words: int, word_count: int) -> float:
    return word_count/total_words
