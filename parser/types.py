from typing import TypedDict, Callable, Literal
import numpy as np
from dataclasses import dataclass


class Metadata(TypedDict):
    stripper: Literal["lemmatize", "stem"]
    total_docs: int
    files: dict


DocumentMatrix = np.ndarray


@dataclass
class DocOccurrences:
    filename: str
    word: str
    num_occ: int
    tf: float
    weight: float
    positions: list[int]
    tfidf: float = 0.0


# doc token represents an instance of a word in a particular document
@dataclass
class DocToken:
    count: int
    weight: float
    position: int


# token represents
@dataclass
class Token:
    word: str
    count: int
    idf: float
    occurrences: list[DocOccurrences]
    positions: list[list[int]]


InvertedIndex = dict[str, Token]
StripFunc = Callable[[str], str]
