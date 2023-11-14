import pickle
from pathlib import Path
from main import Context
from search.types import QueryTerm

from bs4 import BeautifulSoup
from bs4.element import Comment
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from config import PICKLE_DIR
from typing import (
    BinaryIO,
    NoReturn,
    Sequence,
    assert_never,
    Literal,
)
import logging
from search.ranking import calculate_tf, calculate_idf, calculate_tf_idf, calculate_bm25
from resources import lemmatizer, stop_words, translator, stemmer
from enum import Enum
from collections import defaultdict
import numpy as np
from parser.types import (
    InvertedIndex,
    StripFunc,
    DocOccurrences,
    DocToken,
    Metadata,
    Token,
    FileParseSuccess,
    ParsedDirResults,
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
    return filter_tokens(strip_func, word_tokenize(clean_text(text)))


def filter_tokens(
    strip_func: StripFunc, tokens: Sequence[str | QueryTerm], query=False
) -> list[str | QueryTerm]:
    return (
        [strip_func(word) for word in tokens if word not in stop_words]
        if not query
        else [
            QueryTerm(term=strip_func(word.term), weight=word.weight)
            for word in tokens
            if word not in stop_words
        ]
    )


def get_strip_func(strip_type: Literal["lemmatize", "stem"]) -> StripFunc:
    match strip_type:
        case "lemmatize":
            return lemmatizer.lemmatize
        case "stem":
            return stemmer.stem
        case _:
            assert_never("Unreachable")


def clean_text(text: str) -> str:
    return text.replace("-", " ").translate(translator).lower()


def parse_contents(ctx: Context, file: BinaryIO, parser="lxml") -> FileParseSuccess:
    soup = BeautifulSoup(file.read(), features=parser)
    # remove unneeded info
    comments = soup.find_all(string=lambda element: isinstance(element, Comment))
    [s.extract() for s in soup(["script", "style", "iframe", "a", "img"])]
    [c.extract() for c in comments]

    occurrences: list[DocOccurrences] = []

    filtered_all_text = filter_text(get_strip_func(ctx.stripper), soup.get_text())

    tokens = defaultdict(list[DocToken])
    total_words = len(filtered_all_text)

    for i, el in enumerate(soup.find_all()):
        filtered_text = filter_text(get_strip_func(ctx.stripper), el.get_text())

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

    return {"result": occurrences, "word_count": total_words}


def get_count(words: list[str]) -> FreqDist:
    return FreqDist(words)


def set_weightings_ii(ctx: Context, ii: InvertedIndex, metadata: Metadata) -> None:
    for word, token in ii.items():
        doc_count = len(token.occurrences)

        idf = calculate_idf(metadata["total_docs"], doc_count)
        bm25_idf = calculate_idf(metadata["total_docs"], doc_count, bm25=True)

        token.idf = idf
        token.bm25_idf = bm25_idf

        for i, occ in enumerate(token.occurrences):
            doc_wc = metadata["files"][occ.filename]["word_count"]

            tf_idf = calculate_tf_idf(occ.tf, idf)
            bm25 = calculate_bm25(
                occ.tf, bm25_idf, doc_wc, metadata["average_wc"], plus=False
            )
            bm25_plus = calculate_bm25(
                occ.tf, bm25_idf, doc_wc, metadata["average_wc"], plus=True
            )
            occ.tfidf = tf_idf
            occ.bm25 = bm25
            occ.bm25_plus = bm25_plus

    if ctx.verbose:
        logger.info("Set term weights")


def merge_word_count_dicts(
    ctx: Context, doc_dict: ParsedDirResults, metadata: Metadata
) -> InvertedIndex:
    ii: InvertedIndex = {}

    for name, occurrences in doc_dict.items():
        for occ in occurrences:
            if occ.word in ii:
                ii[occ.word].count += occ.num_occ
                ii[occ.word].occurrences.append(occ)
                ii[occ.word].positions.extend([occ.positions])
            else:
                ii[occ.word] = Token(
                    occ.word, occ.num_occ, 0, 0, [occ], [occ.positions]
                )

    set_weightings_ii(ctx, ii, metadata)

    if ctx.verbose:
        logger.info("Generated inverted index")

    return ii


def generate_document_matrix(
    ctx: Context, ii: InvertedIndex, metadata: Metadata
) -> tuple[list[str], dict[str, np.ndarray]]:
    if ctx.verbose:
        logger.info("Generating document matrix")

    vector_space = list(ii.keys())

    # maybe below would be better ?
    # matrix = np.zeros([len(ii.keys()), len(metadata["files"])])

    doc_dict = {}

    for i, term in enumerate(vector_space):
        for occ in ii[term].occurrences:
            if occ.filename not in doc_dict:
                doc_dict[occ.filename] = np.zeros(len(vector_space))

            match ctx.scorer:
                case "tfidf":
                    score = occ.tfidf
                case "bm25":
                    score = occ.bm25
                case "bm25+":
                    score = occ.bm25_plus
                case _:
                    assert_never("Unreachable")

            if ctx.weighted:
                score = score * occ.weight

            doc_dict[occ.filename][i] = score

    return vector_space, doc_dict


def get_pickle_name(stripper: Literal["lemmatize", "stem"]) -> Path:
    PICKLE_DIR.mkdir(exist_ok=True)
    return PICKLE_DIR / f"{stripper}.pkl"


def pickle_obj(ctx: Context, data: InvertedIndex) -> None | NoReturn:
    if ctx.verbose:
        logger.info("Pickling index")

    try:
        with open(get_pickle_name(ctx.stripper), "wb") as f:
            pickle.dump(data, f)
            return
    except Exception as e:
        logger.critical(e)
        exit(1)


def depickle_obj(ctx: Context) -> InvertedIndex | NoReturn:
    if ctx.verbose:
        logger.info("Depickling index")
    try:
        with open(get_pickle_name(ctx.stripper), "rb") as f:
            data = pickle.load(f)
            return data
    except Exception as e:
        logger.critical(e)
        exit(1)
