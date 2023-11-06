import pickle

from bs4 import BeautifulSoup
from bs4.element import Comment
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from dataclasses import dataclass
from config import PICKLE_FILE
from typing import List, Dict, BinaryIO, NoReturn, Union, Tuple
import logging
from search.tf_idf import calculate_tf, calculate_idf, calculate_tf_idf
from resources import lemmatizer, stop_words, translator
from enum import Enum
from collections import defaultdict
import numpy as np


logger = logging.getLogger(__name__)


DocumentMatrix = np.ndarray


@dataclass
class DocOccurrences:
    filename: str
    word: str
    num_occ: int
    tf: float
    weight: float
    positions: List[int]
    tfidf: float = 0.0


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
    occurrences: List[DocOccurrences]
    positions: List[List[int]]


InvertedIndex = Dict[str, Token]


# add weight as well
def parse_contents(file: BinaryIO, parser="lxml") -> List[DocOccurrences]:
    soup = BeautifulSoup(file.read(), features=parser)
    # remove unneeded info
    comments = soup.find_all(string=lambda element: isinstance(element, Comment))
    [s.extract() for s in soup(["script", "style", "iframe", "a", "img"])]
    [c.extract() for c in comments]

    occurrences: List[DocOccurrences] = []

    all_text = soup.get_text().translate(translator)

    filtered_all_text = [
        # stemmer.stem(word.lower())
        lemmatizer.lemmatize(word.lower())
        for word in word_tokenize(all_text)
        if word not in stop_words
    ]

    tokens = defaultdict(list[DocToken])

    total_words = len(filtered_all_text)

    for i, el in enumerate(soup.find_all()):
        text = el.get_text().translate(translator)

        filtered_text = [
            # stemmer.stem(word.lower())
            lemmatizer.lemmatize(word.lower())
            for word in word_tokenize(text)
            if word not in stop_words
        ]

        match el.name:
            case "h1":
                weight = Weight.H1.value
            case "h2":
                weight = Weight.H2.value
            case "h3":
                weight = Weight.H3.value
            case "h4":
                weight = Weight.H4.value
            case "h5":
                weight = Weight.H5.value
            case "h6":
                weight = Weight.H6.value
            case "p":
                weight = Weight.P.value
            case "title":
                weight = Weight.TITLE.value
            case "strong":
                weight = Weight.STRONG.value
            case "em":
                weight = Weight.EM.value
            case "b":
                weight = Weight.BOLD.value
            case "i":
                weight = Weight.ITALIC.value
            case "meta":
                weight = Weight.META.value
            case "a":
                weight = Weight.A.value
            case _:
                weight = 1.0

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
        for occ in occur:
            total_weight *= occ.weight
            total_count += occ.count

        tf = calculate_tf(total_words, total_count)
        occurrences.append(
            DocOccurrences(
                filename=file.name,
                word=word,
                num_occ=total_count,
                tf=tf,
                weight=total_weight,
                positions=[occ.position for occ in occur],
            )
        )

    # logger.debug(occurrences)

    return occurrences


def get_count(words: List[str]) -> FreqDist:
    return FreqDist(words)


def merge_word_count_dicts(
    doc_dict: Dict[str, List[DocOccurrences]], total_docs: int
) -> Tuple[InvertedIndex, DocumentMatrix]:
    merged_dict: InvertedIndex = {}

    doc_matrix: DocumentMatrix = np.zeros((total_docs, len(doc_dict)))

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
        idf = calculate_idf(total_docs, doc_count)
        token.idf = idf

        for i, occ in enumerate(token.occurrences):
            tf_idf = calculate_tf_idf(occ.tf, idf)
            occ.tfidf = tf_idf
            doc_matrix[i][occ.positions] = tf_idf

    logger.debug("Generated inverted index")

    return merged_dict, doc_matrix


def pickle_obj(data: InvertedIndex) -> Union[None, NoReturn]:
    try:
        with open(PICKLE_FILE, "wb") as f:
            pickle.dump(data, f)
            return
    except Exception as e:
        logger.critical(e)
        exit(1)


def depickle_obj() -> Union[InvertedIndex, NoReturn]:
    try:
        with open(PICKLE_FILE, "rb") as f:
            data = pickle.load(f)
            return data
    except Exception as e:
        logger.critical(e)
        exit(1)
