import pickle
from pathlib import Path
from main import Context

from bs4 import BeautifulSoup
from bs4.element import Comment
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from config import PICKLE_FILE
from typing import (
    BinaryIO,
    NoReturn,
    Iterable,
    assert_never,
    Literal,
)
import logging
from search.tf_idf import calculate_tf, calculate_idf, calculate_tf_idf
from resources import lemmatizer, stop_words, translator, stemmer
from enum import Enum
from collections import defaultdict
import numpy as np
from parser.types import (
    InvertedIndex,
    StripFunc,
    DocumentMatrix,
    DocOccurrences,
    DocToken,
    Metadata,
    Token,
)


logger = logging.getLogger(__name__)


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
    def get_word_weight(cls, tag_name: str) -> float:
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


def filter_text(strip_func: StripFunc, text: str) -> list[str]:
    return filter_tokens(strip_func, word_tokenize(text))


def filter_tokens(strip_func: StripFunc, tokens: Iterable[str]) -> list[str]:
    return [strip_func(word) for word in tokens if word not in stop_words]


def get_strip_func(strip_type: Literal["lemmatize", "stem"]) -> StripFunc:
    match strip_type:
        case "lemmatize":
            return lemmatizer.lemmatize
        case "stem":
            return stemmer.stem
        case _:
            assert_never("Unreachable")


def parse_contents(ctx: Context, file: BinaryIO, parser="lxml") -> list[DocOccurrences]:
    soup = BeautifulSoup(file.read(), features=parser)
    # remove unneeded info
    comments = soup.find_all(string=lambda element: isinstance(element, Comment))
    [s.extract() for s in soup(["script", "style", "iframe", "a", "img"])]
    [c.extract() for c in comments]

    occurrences: list[DocOccurrences] = []

    all_text = soup.get_text().translate(translator).lower()

    filtered_all_text = filter_text(get_strip_func(ctx.stripper), all_text)

    tokens = defaultdict(list[DocToken])
    total_words = len(filtered_all_text)

    for i, el in enumerate(soup.find_all()):
        text = el.get_text().translate(translator).lower()

        filtered_text = filter_text(get_strip_func(ctx.stripper), text)

        weight = Weight.get_word_weight(el.name)
        counts: FreqDist = get_count(filtered_text)

        for name, count in counts.items():
            tokens[name].append(
                DocToken(
                    count=count,
                    weight=weight,
                    position=i,
                )
            )

    for word, occur in tokens.items():
        total_weight = 1
        total_count = 0
        positions = []
        for occ in occur:
            total_weight *= occ.weight
            total_count += occ.count
            positions.append(occ.position)

        tf = calculate_tf(total_words, total_count)
        occurrences.append(
            DocOccurrences(
                filename=file.name,
                word=word,
                num_occ=total_count,
                tf=tf,
                weight=total_weight,
                positions=positions,
            )
        )

    return occurrences


def get_count(words: list[str]) -> FreqDist:
    return FreqDist(words)


def merge_word_count_dicts(
    ctx: Context, doc_dict: dict[Path, list[DocOccurrences]], metadata: Metadata
) -> tuple[InvertedIndex, DocumentMatrix]:
    merged_dict: InvertedIndex = {}

    doc_matrix: DocumentMatrix = np.zeros((metadata["total_docs"], len(doc_dict)))

    for name, occurrences in doc_dict.items():
        for occ in occurrences:
            if occ.word in merged_dict:
                merged_dict[occ.word].count += occ.num_occ
                merged_dict[occ.word].occurrences.append(occ)
                merged_dict[occ.word].positions.extend([occ.positions])
            else:
                merged_dict[occ.word] = Token(
                    occ.word, occ.num_occ, 0, [occ], [occ.positions]
                )

    for word, token in merged_dict.items():
        doc_count = len(token.occurrences)
        idf = calculate_idf(metadata["total_docs"], doc_count)
        token.idf = idf

        for i, occ in enumerate(token.occurrences):
            tf_idf = calculate_tf_idf(occ.tf, idf)
            occ.tfidf = tf_idf
            doc_matrix[i][occ.positions] = tf_idf

    logger.debug("Generated inverted index")

    return merged_dict, doc_matrix


# TODO: pickle multiples files based on ctx
def pickle_obj(ctx: Context, data: InvertedIndex) -> None | NoReturn:
    if ctx.verbose:
        logger.info("Pickling index")
    try:
        with open(PICKLE_FILE, "wb") as f:
            pickle.dump(data, f)
            return
    except Exception as e:
        logger.critical(e)
        exit(1)


def depickle_obj(ctx: Context) -> InvertedIndex | NoReturn:
    if ctx.verbose:
        logger.info("Depickling index")
    try:
        with open(PICKLE_FILE, "rb") as f:
            data = pickle.load(f)
            return data
    except Exception as e:
        logger.critical(e)
        exit(1)
