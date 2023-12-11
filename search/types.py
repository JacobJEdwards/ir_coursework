from typing import NamedTuple, Callable


SearchResult = tuple[str, float]
SearchResults = list[SearchResult]
SearchFunc = Callable[..., SearchResults]


class QueryTerm(NamedTuple):
    term: str
    weight: float
