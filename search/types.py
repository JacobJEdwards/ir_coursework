from typing import NamedTuple, Callable, TypedDict
import numpy as np


# A search result consisting of a document ID (string) and a search score (float)
class ResultType(TypedDict):
    vec: np.ndarray
    score: float


SearchResult = tuple[str, ResultType | float]

# A collection of search results
SearchResults = list[SearchResult]

# Represents a callable type that performs a search operation and returns search results
SearchFunc = Callable[..., SearchResults]


class QueryTerm(NamedTuple):
    """
    Represents a query term with a term string and its associated weight.

    Attributes:
    - term (str): The query term.
    - weight (float): The weight associated with the query term.
    """

    term: str
    weight: float
