from typing import NamedTuple

SearchResult = tuple[str, float]
SearchResults = set[SearchResult]


class QueryTerm(NamedTuple):
    term: str
    weight: float
