import pickle

from bs4 import BeautifulSoup
from bs4.element import Comment
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from typing import List, Dict, BinaryIO, NoReturn, Union
from dataclasses import dataclass
from collections import namedtuple
from config import PICKLE_FILE
import logging
from search.tf_idf import calculate_tf, calculate_idf
from resources import lemmatizer, stop_words, translator

logger = logging.getLogger(__name__)


# named tuple to group together occurrences of a word in a document
DocOccurrences = namedtuple("DocOccurrences", "filename word num_occ tf")


# doc token represents an instance of a word in a particular document
@dataclass
class DocToken:
    word: str
    count: int
    position: int


# token represents
@dataclass
class Token:
    word: str
    count: int
    idf: float
    occurrences: List[DocOccurrences]


InvertedIndex = Dict[str, Token]


# eventually replace with nltk.probabilty.FreqDist
# add weight as well
def parse_contents(file: BinaryIO, parser="lxml") -> List[DocOccurrences]:
    soup = BeautifulSoup(file.read(), features=parser)
    # remove unneeded info
    comments = soup.find_all(string=lambda element: isinstance(element, Comment))
    [s.extract() for s in soup(["script", "style", "iframe", "a", "img"])]
    [c.extract() for c in comments]

    text = soup.get_text()
    text = text.translate(translator)

    filtered_text = [
        # stemmer.stem(word.lower())
        lemmatizer.lemmatize(word.lower())
        for word in word_tokenize(text)
        if word not in stop_words
    ]

    total_words = len(filtered_text)
    counts: FreqDist = get_count(filtered_text)

    return [
        DocOccurrences(
            filename=file.name,
            word=name,
            num_occ=count,
            tf=calculate_tf(total_words, count),
        )
        for name, count in counts.items()
    ]


def get_count(words: List[str]) -> FreqDist:
    return FreqDist(words)


def merge_word_count_dicts(
    doc_dict: Dict[str, List[DocOccurrences]], total_docs: int
) -> InvertedIndex:
    merged_dict: InvertedIndex = {}

    # Use defaultdict to store occurrences

    for name, occurrences in doc_dict.items():
        for occ in occurrences:
            if occ.word in merged_dict:
                merged_dict[occ.word].count += occ.num_occ
                merged_dict[occ.word].occurrences.append(occ)
            else:
                merged_dict[occ.word] = Token(occ.word, occ.num_occ, 0, [occ])

    for word, token in merged_dict.items():
        doc_count = len(token.occurrences)
        token.idf = calculate_idf(total_docs, doc_count)

    logger.debug("Generated inverted index")

    return merged_dict


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
