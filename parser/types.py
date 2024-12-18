from typing import TypedDict, Callable, Literal, Mapping, Self, Type
from pathlib import Path
import numpy as np
from dataclasses import dataclass
from enum import Enum

RawTokenList = set[str]
StripFunc = Callable[[str], str]
StripperType = Literal["lemmatize", "stem", "none"]


# Defining TypedDicts to represent structured data
class FileMetadata(TypedDict):
    last_modified: float
    file_size: int
    info: dict[str, str] | None
    word_count: int


class Metadata(TypedDict):
    stripper: StripperType
    total_docs: int
    files: dict[str, FileMetadata]
    average_wc: float


# Typing for specific data structures
DocumentMatrix = np.ndarray


# Enum for defining weights based on HTML tag names
class Weight(Enum):
    # Enum members with associated weights for different HTML tags
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
    def get_word_weight(cls: Self, tag_name: str) -> float:
        # Method to retrieve weight based on the HTML tag
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


# Enum for different types of entities
class Entity(Enum):
    # Enum members representing different types of entities
    ORGANIZATION = 1
    GPE = 1
    PERSON = 1
    GSP = 1
    NONE = 1

    @classmethod
    def get_entity(cls: Self, label: str) -> Self:
        # Method to retrieve an entity based on its label
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


# Data class to represent a token within a document
@dataclass
class DocToken:
    count: int
    weight: float
    position: int


# Data class representing an entity within a document
@dataclass
class DocEntity:
    token: str
    filename: str
    type: Entity = Entity.NONE
    position: int | None = None


# Data class to represent occurrences of a word in a document
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


# Data class representing a token in a document with its occurrences and positions
@dataclass
class Token:
    word: str
    count: int
    idf: float
    bm25_idf: float
    occurrences: list[DocOccurrences]
    positions: list[list[int]]


# Alias definitions for better readability and type hints
InvertedIndex = dict[str, Token]


# TypedDicts and aliases for managing parsing results and directories
class FileParseSuccess(TypedDict):
    tokens: list[DocOccurrences]
    word_count: int
    entities: list[DocEntity] | None


ParsedFile = FileParseSuccess | Exception
ParsedDir = Mapping[Path, ParsedFile]
ParsedDirSuccess = dict[Path, FileParseSuccess]

ParsedDirResults = Mapping[Path, list[DocOccurrences]]
