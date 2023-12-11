from typing import TypedDict, Callable, Literal, Mapping
from pathlib import Path
import numpy as np
from dataclasses import dataclass
from enum import Enum


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


class Weight(Enum):
    H1 = 1.5
    H2 = 1.4
    H3 = 1.3
    H4 = 1.2
    H5 = 1.1
    H6 = 1.0
    P = 1.0
    TITLE = 1.6
    BOLD = 1.2
    ITALIC = 1.1
    STRONG = 1.2
    EM = 1.1
    A = 1.0
    META = 1.2

    @classmethod
    def get_word_weight(cls: "Weight", tag_name: str) -> float:
        match tag_name:
            case "h1":
                weight = cls.H1.value
            case "h2":
                weight = cls.H2.value
            case "h3":
                weight = cls.H3.value
            case "h4":
                weight = cls.H4.value
            case "h5":
                weight = cls.H5.value
            case "h6":
                weight = cls.H6.value
            case "p":
                weight = cls.P.value
            case "title":
                weight = cls.TITLE.value
            case "strong":
                weight = cls.STRONG.value
            case "em":
                weight = cls.EM.value
            case "b":
                weight = cls.BOLD.value
            case "i":
                weight = cls.ITALIC.value
            case "meta":
                weight = cls.META.value
            case "a":
                weight = cls.A.value
            case _:
                weight = 1.0

        return weight


class Entity(Enum):
    ORGANIZATION = 1
    GPE = 1
    PERSON = 1
    GSP = 1
    NONE = 1

    @classmethod
    def get_entity(cls: "Entity", label: str) -> "Entity":
        match label:
            case "ORGANIZATION":
                tag = cls.ORGANIZATION
            case "GPE":
                tag = cls.GPE
            case "PERSON":
                tag = cls.PERSON
            case "GSP":
                tag = cls.GSP
            case _:
                tag = cls.NONE

        return tag


# doc token represents an instance of a word in a particular document
@dataclass
class DocToken:
    count: int
    weight: float
    position: int


@dataclass
class DocEntity:
    token: str
    filename: str
    type: Entity = Entity.NONE
    position: int | None = None


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
StripperType = Literal["lemmatize", "stem"]


class FileParseSuccess(TypedDict):
    tokens: list[DocOccurrences]
    word_count: int
    entities: list[DocEntity] | None


ParsedFile = FileParseSuccess | Exception
ParsedDir = Mapping[Path, ParsedFile]
ParsedDirSuccess = dict[Path, FileParseSuccess]

ParsedDirResults = Mapping[Path, list[DocOccurrences]]
