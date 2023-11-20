###### IHAVE BROKEN SOMETHING
from pprint import pprint

from typing import assert_never, Sequence
from nltk.corpus import wordnet
from parser.parser import filter_tokens, get_strip_func, filter_text
from parser.types import InvertedIndex, Metadata
from search.ranking import calculate_tf, calculate_bm25, calculate_tf_idf
from collections import defaultdict
from resources import sym_spell
from symspellpy import Verbosity
import numpy as np
from main import Context
from parser.reader import get_metadata, index_documents
import logging
from search.types import SearchResults, SearchResult, QueryTerm
from config import VOCAB_PATH

logger = logging.getLogger(__name__)


# https://symspellpy.readthedocs.io
def get_suggestions(
    tokens: Sequence[QueryTerm], print_terms: bool = False
) -> list[QueryTerm]:
    new_tokens = list(tokens)
    for token in tokens:
        suggestions = sym_spell.lookup(
            token.term, Verbosity.CLOSEST, max_edit_distance=2, include_unknown=True
        )

        if suggestions[0].term == token.term or suggestions[0].distance == 0:
            new_tokens.append(QueryTerm(suggestions[0].term, token.weight))
            continue

        if print_terms:
            logger.info(f"Spell suggestions for {token.term}:")
            for sug in suggestions:
                logger.info(f"{sug.term}")

            print()

        for i, suggestion in enumerate(suggestions):
            if i == 3:
                break

            print("Did you mean: ", suggestion.term)
            if input("Y/N: ").lower() == "y":
                new_tokens.append(QueryTerm(term=suggestion.term, weight=token.weight))
                break
            else:
                new_tokens.append(token)

    return new_tokens


# need to give these less weight somehow
def expand_query(
    query_terms: Sequence[QueryTerm], print_terms: bool = False
) -> list[QueryTerm]:
    new_terms = list(query_terms)
    for query_term in set(query_terms):
        syns = wordnet.synsets(query_term.term)
        for syn in syns:
            for lemma in syn.lemmas():
                new_terms.append(QueryTerm(term=lemma.name(), weight=0.3))

    if print_terms:
        logger.info(f"Original terms: {query_terms}\nExpanded terms: {new_terms}\n")

    return new_terms


def _clean_tokenized_input(
    ctx: Context, tokens: Sequence[QueryTerm], metadata: Metadata
) -> list[QueryTerm]:
    if ctx.spellcheck:
        tokens = get_suggestions(tokens, ctx.verbose)

    if ctx.expand:
        tokens = expand_query(tokens, ctx.verbose)

    tokens = filter_tokens(get_strip_func(ctx.stripper), tokens, query=True)

    if ctx.verbose:
        logger.info(f"Final tokens: {tokens}\n")

    return tokens


def clean_input(ctx: Context, user_input: str, metadata: Metadata) -> list[QueryTerm]:
    # simple cleaner - prevent spell errors on stopwords (not in dictionary)
    tokens = [
        QueryTerm(term=term, weight=1) for term in filter_text(lambda x: x, user_input)
    ]

    return _clean_tokenized_input(ctx, tokens, metadata)


def get_input(ctx: Context, metadata: Metadata) -> Sequence[QueryTerm]:
    user_input = input("Enter search term(s): ")
    print()
    return clean_input(ctx, user_input, metadata)


def calculate_dot(vec1: np.ndarray, vec2: np.ndarray) -> int:
    total = 0
    for a, b in zip(vec1, vec2):
        total += a * b

    return total


# DO NOT USE NP.DOT -> do this manually
def cosine_similarity(query_vector: np.ndarray, doc_vector: np.ndarray) -> float:
    # dot_product = np.dot(query_vector, doc_vector.T)
    dot_product = calculate_dot(query_vector, doc_vector.T)

    query_norm = np.linalg.norm(query_vector)
    doc_norm = np.linalg.norm(doc_vector)

    return dot_product / (query_norm * doc_norm)


def vectorise_query(
    ctx: Context,
    query_terms: Sequence[QueryTerm],
    vec_space: Sequence[str],
    ii: InvertedIndex,
    metadata: Metadata,
) -> np.ndarray:
    query_vector = np.zeros(len(vec_space))

    terms = [query.term for query in query_terms]
    # this is the problem
    for i, term in enumerate(vec_space):
        if term in terms:
            term_count = terms.count(term)
            tf = calculate_tf(len(query_terms), term_count)

            match ctx.scorer:
                case "tfidf":
                    idf = ii[term].idf
                    score = calculate_tf_idf(tf, idf)
                case "bm25":
                    idf = ii[term].bm25_idf
                    score = calculate_bm25(
                        tf, idf, len(query_terms), metadata["average_wc"], plus=False
                    )
                case "bm25+":
                    idf = ii[term].bm25_idf
                    score = calculate_bm25(
                        tf, idf, len(query_terms), metadata["average_wc"], plus=True
                    )
                case _:
                    assert_never("Unreachable")

            # if ctx.weighted:
            # score = score * term.weight

            query_vector[i] = score

    return query_vector


def search_vecs(
    ctx: Context,
    query_terms: Sequence[QueryTerm],
    doc_vecs: dict[str, np.ndarray],
    vec_space: Sequence[str],
    ii: InvertedIndex,
    metadata: Metadata,
) -> SearchResults:
    query_vec = vectorise_query(ctx, query_terms, vec_space, ii, metadata)
    results = defaultdict(float)

    for doc, vec in doc_vecs.items():
        results[doc] = cosine_similarity(query_vec, vec)

    return list(sorted(results.items(), key=lambda x: x[1], reverse=True)[:10])


def search_idf(
    ctx: Context,
    query_terms: set[QueryTerm],
    inverted_index: InvertedIndex,
    metadata: Metadata,
) -> SearchResults:
    results = defaultdict(float)

    for term in query_terms:
        if term.term in inverted_index:
            token = inverted_index[term.term]

            for occurrence in token.occurrences:
                match ctx.scorer:
                    case "tfidf":
                        score = occurrence.tfidf
                    case "bm25":
                        score = occurrence.bm25
                    case "bm25+":
                        score = occurrence.bm25_plus
                    case _:
                        assert_never("Unreachable")

                if ctx.weighted:
                    results[occurrence.filename] += (
                        score * occurrence.weight * term.weight
                    )
                else:
                    results[occurrence.filename] += score

    return list(sorted(results.items(), key=lambda x: x[1], reverse=True)[:10])


def print_result(ctx: Context, result: SearchResult, metadata: Metadata) -> None:
    doc, score = result

    if doc in metadata["files"] and metadata["files"][doc]["info"] is not None:
        info: dict = metadata["files"][doc]["info"]
        for key, val in info.items():
            print(f"{key}: {val}")

    else:
        print(f"Document: {doc}")

    if ctx.verbose:
        print(f"Score: {score}")

    print()


def print_results(ctx: Context, results: SearchResults, metadata: Metadata) -> None:
    if len(results) == 0:
        print("No results found")
        return

    for i, (doc, score) in enumerate(results):
        print(f"{int(i)+1}:")
        print_result(ctx, (doc, score), metadata)


# issue with this is that the vocab will have been lemmed or stemmed
def generate_vocab(ctx: Context, vocab: InvertedIndex) -> None:
    if ctx.verbose:
        logger.info("Generating dictionary")

    with open(VOCAB_PATH, "w") as f:
        for word, token in vocab.items():
            f.write(f"{word} {token.count}\n")

    sym_spell.load_dictionary(VOCAB_PATH, term_index=0, count_index=1)


def search(ctx: Context) -> None:
    # imported to allow backspacing in input
    import readline  # ruff: noqa

    ii, doc_vectors, vec_space = index_documents(ctx)

    if ctx.spellcheck:
        generate_vocab(ctx, ii)

    metadata: Metadata = get_metadata(ctx)

    while True:
        user_input = get_input(ctx, metadata)
        match ctx.searcher:
            case "vector":
                results: SearchResults = search_vecs(
                    ctx, user_input, doc_vectors, vec_space, ii, metadata
                )
            case "score":
                results: SearchResults = search_idf(ctx, set(user_input), ii, metadata)
            case _:
                assert_never("Unreachable")

        print_results(ctx, results, metadata)
