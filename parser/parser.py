import pickle
from pathlib import Path

from main import Context
from search.types import QueryTerm

from bs4 import BeautifulSoup
from bs4.element import Comment
from nltk.tokenize import word_tokenize
from collections import Counter
from config import PICKLE_DIR
from typing import BinaryIO, NoReturn, Sequence, assert_never
import logging
from search.ranking import calculate_tf, calculate_idf, calculate_tf_idf, calculate_bm25
from resources import lemmatizer, stop_words, translator, stemmer
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
    Weight,
    DocEntity,
    StripperType,
)


logger = logging.getLogger(__name__)


def filter_text(strip_func: StripFunc, text: str) -> list[str]:
    """
    Tokenizes the input text, cleans it, and filters tokens based on the provided strip function.

    Args:
        strip_func (StripFunc): A function used to strip/clean tokens.
        text (str): The input text to be processed.

    Returns:
        list[str]: A list of filtered tokens after applying the strip function.
    """
    return filter_tokens(strip_func, word_tokenize(clean_text(text)), query=False)


def filter_tokens(
    strip_func: StripFunc,
    tokens: Sequence[str | QueryTerm],
    *,
    query: bool = False,
    remove_stopwords: bool = True,
) -> list[str] | list[QueryTerm]:
    """
    Filters tokens based on the provided strip function and other criteria.

    Args:
        strip_func (StripFunc): A function used to strip/clean tokens.
        tokens (Sequence[Union[str, QueryTerm]]): Sequence of tokens to filter.
        query (bool, optional): Flag indicating if the tokens represent a query. Defaults to False.
        remove_stopwords (bool, optional): Flag to remove stopwords. Defaults to True.

    Returns:
        list[QueryTerm] | list[str]: A list of filtered tokens after applying the strip function.
    """
    return (
        [
            strip_func(word)
            for word in tokens
            if not remove_stopwords or word not in stop_words
        ]
        if not query
        else [
            QueryTerm(term=strip_func(word.term), weight=word.weight)
            for word in tokens
            if not remove_stopwords or word not in stop_words
        ]
    )


def get_strip_func(strip_type: StripperType) -> StripFunc:
    """
    Returns the appropriate stripping function based on the specified type.

    Args:
        strip_type (Literal["lemmatize", "stem"]): Type of stripping function required.

    Returns:
        StripFunc: The stripping function (lemmatization or stemming).
    """
    match strip_type:
        case "lemmatize":
            return lemmatizer.lemmatize
        case "stem":
            return stemmer.stem
        case "none":
            return lambda x: x
        case _:
            assert_never("Unreachable")


def clean_text(text: str) -> str:
    """
    Cleans the input text by replacing certain characters and converting it to lowercase.

    Args:
        text (str): The input text to be cleaned.

    Returns:
        str: The cleaned text.
    """
    return text.replace("-", " ").translate(translator).casefold().strip()


def parse_contents(
    ctx: Context, file: BinaryIO, *, parser: str = "lxml"
) -> FileParseSuccess:
    """
    Parses the contents of a file using BeautifulSoup, extracts tokens, and calculates occurrences.

    Args:
        ctx (Context): The context or configuration for parsing.
        file (BinaryIO): The file object to be parsed.
        parser (str, optional): The parser to use with BeautifulSoup. Defaults to "lxml".

    Returns:
        FileParseSuccess: A dictionary containing tokens, entities, and word count information.
    """
    soup = BeautifulSoup(file.read(), features=parser)
    # remove unneeded info
    comments = soup.find_all(string=lambda element: isinstance(element, Comment))
    [s.extract() for s in soup(["script", "style", "iframe", "a", "img"])]
    [c.extract() for c in comments]
    occurrences: list[DocOccurrences] = []

    tokens = defaultdict(list[DocToken])
    entities: list[DocEntity] = []

    # save calling this function in every loop iteration
    strip_func: StripFunc = get_strip_func(ctx.stripper)

    total_words = 0
    for i, el in enumerate(soup.find_all()):
        filtered_text = filter_text(strip_func, el.get_text())
        total_words += len(filtered_text)

        weight: float = Weight.get_word_weight(el.name)
        counts: Counter = get_count(filtered_text)

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

    return {"tokens": occurrences, "entities": entities, "word_count": total_words}


def get_count(words: list[str]) -> Counter:
    """
    Counts the occurrences of words in a list and returns a Counter object.

    Args:
        words (list[str]): A list of words to count occurrences.

    Returns:
        Counter: A Counter object containing the count of each word in the input list.
    """
    return Counter(words)


def set_weightings_ii(ctx: Context, ii: InvertedIndex, metadata: Metadata) -> None:
    """
    Calculates and assigns various weightings to tokens in the inverted index.

    Args:
        ctx (Context): The context or configuration for weighting.
        ii (InvertedIndex): The inverted index containing tokens and their occurrences.
        metadata (Metadata): Metadata containing document-level information.

    Returns:
        None
    """
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
    """
    Merges word count dictionaries and generates an inverted index with weighted tokens.

    Args:
        ctx (Context): The context or configuration for merging and weighting.
        doc_dict (ParsedDirResults): Dictionary containing word occurrences in documents.
        metadata (Metadata): Metadata containing document-level information.

    Returns:
        InvertedIndex: The resulting inverted index with weighted tokens.
    """
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
    """
    Generates a document-term matrix based on the weighted tokens in the inverted index.

    Args:
        ctx (Context): The context or configuration for matrix generation.
        ii (InvertedIndex): The inverted index containing weighted tokens.
        metadata (Metadata): Metadata containing document-level information.

    Returns:
        tuple[list[str], dict[str, np.ndarray]]: A tuple containing vector space and document matrix.
    """
    if ctx.verbose:
        logger.info("Generating document matrix")

    vector_space = list(ii.keys())

    doc_dict = defaultdict(lambda: np.zeros(len(vector_space)))

    for i, term in enumerate(vector_space):
        for occ in ii[term].occurrences:
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


def get_pickle_name(stripper: StripperType) -> Path:
    """
    Generates the pickle file name based on the chosen stripping method.

    Args:
        stripper (Literal["lemmatize", "stem"]): The stripping method used.

    Returns:
        Path: The path to the pickle file.
    """
    PICKLE_DIR.mkdir(exist_ok=True)
    return PICKLE_DIR / f"{stripper}.pkl"


def pickle_obj(ctx: Context, data: InvertedIndex) -> None | NoReturn:
    """
    Pickles the inverted index data.

    Args:
        ctx (Context): The context or configuration for pickling.
        data (InvertedIndex): The inverted index data to be pickled.

    Returns:
        None | NoReturn: None if successful, NoReturn if an error occurs.
    """
    if ctx.verbose:
        logger.info("Pickling index")

    try:
        with open(get_pickle_name(ctx.stripper), "wb") as f:
            pickle.dump(data, f)
            return
    except Exception as e:
        logger.critical(e)
        exit(1)


def unpickle_obj(ctx: Context) -> InvertedIndex | NoReturn:
    """
    Unpickles the inverted index data.

    Args:
        ctx (Context): The context or configuration for unpickling.

    Returns:
        InvertedIndex | NoReturn: The unpickled inverted index if successful, NoReturn if an error occurs.
    """
    if ctx.verbose:
        logger.info("Unpickling index")
    try:
        with open(get_pickle_name(ctx.stripper), "rb") as f:
            data = pickle.load(f)
            return data
    except Exception as e:
        logger.critical(e)
        exit(1)
