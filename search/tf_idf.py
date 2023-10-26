import math


def calculate_tf_idf(tf: int, idf: float) -> float:
    return tf * idf


def calculate_idf(total_docs: int, docs_with_term: int) -> float:
    return math.log(total_docs / (docs_with_term+1))



