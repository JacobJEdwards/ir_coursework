from typing import NamedTuple

SearchResult = tuple[str, float]
SearchResults = list[SearchResult]


class QueryTerm(NamedTuple):
    term: str
    weight: float
