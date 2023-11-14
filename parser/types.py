from typing import TypedDict, Callable, Literal, Mapping
from pathlib import Path
import numpy as np
from dataclasses import dataclass


class FileMetadata(TypedDict):
    last_modified: float
    file_size: int
    info: dict[str, str] | None
    word_count: int


class Metadata(TypedDict):
    stripper: Literal["lemmatize", "stem"]
    total_docs: int
    files: Mapping[str, FileMetadata]
    average_wc: float


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
    bm25: float = 0.0
    bm25_plus: float = 0.0


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
    bm25_idf: float
    occurrences: list[DocOccurrences]
    positions: list[list[int]]


InvertedIndex = dict[str, Token]
StripFunc = Callable[[str], str]


# could use class to emcapuslate parsing together documents, or genreating dm
class II(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)


class FileParseSuccess(TypedDict):
    result: list[DocOccurrences]
    word_count: int


ParsedFile = FileParseSuccess | Exception
ParsedDir = Mapping[Path, ParsedFile]
ParsedDirSuccess = Mapping[Path, FileParseSuccess]

ParsedDirResults = Mapping[Path, list[DocOccurrences]]
