import os
from typing import assert_never, Sequence
from nltk.corpus import wordnet
from parser.parser import filter_tokens, get_strip_func, filter_text
from parser.types import InvertedIndex, Metadata, StripFunc
from search.ranking import (
    calculate_tf,
    calculate_bm25,
    calculate_tf_idf,
    cosine_similarity,
)
from collections import defaultdict
from resources import sym_spell, console
from symspellpy import Verbosity
import numpy as np
from main import Context
from parser.reader import get_metadata, index_documents
import logging
from search.types import SearchResults, SearchResult, QueryTerm
from config import VOCAB_PATH, CACHE_SIZE
from utils import timeit
from functools import partial, cache, lru_cache
from rich.prompt import Prompt, Confirm

logger = logging.getLogger(__name__)


# https://symspellpy.readthedocs.io
def get_suggestions_external(
    tokens: Sequence[QueryTerm], *, print_terms: bool = False
) -> list[QueryTerm]:
    """
    Get suggestions for tokens using SymSpell spell checking.

    Args:
    - tokens (Sequence[QueryTerm]): List of tokens to be checked.
    - print_terms (bool, optional): Whether to print spell suggestions. Defaults to False.

    Returns:
    - list[QueryTerm]: List of tokens with corrected or unchanged terms.
    """
    new_tokens = list(tokens)
    for token in tokens:
        suggestions = sym_spell.lookup(
            token.term, Verbosity.CLOSEST, max_edit_distance=2, include_unknown=True
        )

        if len(suggestions) > 0 and (
            suggestions[0].term == token.term or suggestions[0].distance == 0
        ):
            new_tokens.append(QueryTerm(suggestions[0].term, token.weight))
            continue

        if print_terms:
            logger.info(f"Spell suggestions for {token.term}:")
            for sug in suggestions:
                logger.info(f"{sug.term}")

            print()

        for i, suggestion in enumerate(suggestions):
            if i == 3:
                new_tokens.append(token)
                break

            if Confirm.ask(f"Did you mean [bold]{suggestion.term}[/bold]?"):
                new_tokens.append(QueryTerm(term=suggestion.term, weight=token.weight))
                break
        else:
            new_tokens.append(token)

    return new_tokens


@lru_cache(maxsize=CACHE_SIZE)
def l_distance_rec(a: str, b: str) -> int:
    """
    Computes the Levenshtein distance (edit distance) between two strings using a recursive approach with cache (dynamic programming).

    Args:
    a (str): First string.
    b (str): Second string.

    Returns:
    int: The Levenshtein distance between the input strings a and b.
    """

    @cache
    def helper(i, j):
        if i == 0:
            return j
        elif j == 0:
            return i
        elif a[i - 1] == b[j - 1]:
            return helper(i - 1, j - 1)
        else:
            return 1 + min(
                helper(i, j - 1),  # insertion
                helper(i - 1, j),  # deletion
                helper(i - 1, j - 1),  # substitution
            )

    return helper(len(a), len(b))


@lru_cache(maxsize=CACHE_SIZE)
def load_words() -> set[str]:
    """
    Load words from a vocabulary file.

    Returns:
    - set[str]: Set of words loaded from the vocabulary file.
    """
    with open(VOCAB_PATH, "r") as f:
        words = set(line.split(" ")[0] for line in f.readlines())

    return words


def get_suggestions_internal(
    tokens: Sequence[QueryTerm], *, strip_func: StripFunc, print_terms: bool = False
) -> list[QueryTerm]:
    """
    Get suggestions for query tokens from an internal vocabulary.

    Args:
    - tokens (Sequence[QueryTerm]): List of query tokens.
    - print_terms (bool, optional): Whether to print suggestions. Defaults to False.

    Returns:
    - list[QueryTerm]: List of corrected or unchanged tokens.
    """
    vocab = load_words()
    new_tokens = []
    for token in tokens:
        distances = {
            word: l_distance_rec(strip_func(token.term), word) for word in vocab
        }

        top_3: list[tuple[str, int]] = sorted(distances.items(), key=lambda x: x[1])[:3]

        if len(top_3) < 1 or top_3[0][1] == 0:
            new_tokens.append(token)
            continue

        if print_terms:
            logging.info(f"Top suggestions: {top_3}")

        for suggestion in top_3:
            if Confirm.ask(f"Did you mean [bold]{suggestion[0]}[/bold]?"):
                new_tokens.append(QueryTerm(term=suggestion[0], weight=token.weight))
                break
        else:  # only executed if loop not broken
            new_tokens.append(token)

    return new_tokens


def expand_query(
    query_terms: Sequence[QueryTerm], *, print_terms: bool = False
) -> list[QueryTerm]:
    """
    Expand query terms using synonyms from WordNet.

    Args:
    - query_terms (Sequence[QueryTerm]): List of query terms to be expanded.
    - print_terms (bool, optional): Whether to print expanded terms. Defaults to False.

    Returns:
    - list[QueryTerm]: List of expanded query terms.
    """
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
    """
    Clean and process a sequence of query tokens based on the provided context.

    Args:
    - ctx (Context): Context for processing.
    - tokens (Sequence[QueryTerm]): Sequence of query tokens.
    - metadata (Metadata): Metadata related to the query.

    Returns:
    - list[QueryTerm]: Processed list of query tokens.
    """
    strip_func = get_strip_func(ctx.stripper)

    if ctx.spellcheck:
        tokens = get_suggestions_internal(
            tokens, strip_func=strip_func, print_terms=ctx.verbose
        )

    if ctx.expand:
        tokens = expand_query(tokens, print_terms=ctx.verbose)

    tokens = filter_tokens(strip_func, tokens, query=True)

    if ctx.verbose:
        logger.info(f"Final tokens: {tokens}\n")

    return tokens


def clean_input(ctx: Context, user_input: str, metadata: Metadata) -> list[QueryTerm]:
    """
    Process and clean user input for search.

    Args:
    - ctx (Context): Context for processing.
    - user_input (str): User input as a string.
    - metadata (Metadata): Metadata related to the query.

    Returns:
    - list[QueryTerm]: Processed list of query tokens.
    """
    tokens = [
        QueryTerm(term=term, weight=1) for term in filter_text(lambda x: x, user_input)
    ]

    return _clean_tokenized_input(ctx, tokens, metadata)


def get_input(ctx: Context, metadata: Metadata) -> list[QueryTerm]:
    """
    Get user input and process it for search.

    Args:
    - ctx (Context): Context for processing.
    - metadata (Metadata): Metadata related to the query.

    Returns:
    - list[QueryTerm]: Processed list of query tokens.
    """
    console.print()
    if (user_input := Prompt.ask("Enter search term(s)", default="exit")) == "exit":
        exit(0)

    console.print()
    return clean_input(ctx, user_input, metadata)


def vectorise_query(
    ctx: Context,
    query_terms: Sequence[QueryTerm],
    vec_space: Sequence[str],
    ii: InvertedIndex,
    metadata: Metadata,
) -> np.ndarray:
    """
    Convert a query into a vector in the vector space.

    Args:
    - ctx (Context): Context for search.
    - query_terms (Sequence[QueryTerm]): Query terms to be converted into a vector.
    - vec_space (Sequence[str]): Vector space vocabulary.
    - ii (InvertedIndex): Inverted index for document retrieval.
    - metadata (Metadata): Metadata related to the query and documents.

    Returns:
    - np.ndarray: Vector representation of the query in the vector space.
    """
    query_vector = np.zeros(len(vec_space))

    terms: list[str] = [query.term for query in query_terms]

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

            if ctx.weighted:
                query_term = next(token for token in query_terms if token.term == term)
                score = score * query_term.weight

            query_vector[i] = score

    return query_vector


@timeit
def search_vecs(
    ctx: Context,
    doc_vecs: dict[str, np.ndarray],
    vec_space: Sequence[str],
    ii: InvertedIndex,
    metadata: Metadata,
    query_terms: Sequence[QueryTerm],
) -> SearchResults:
    """
    Perform vector space-based search using query vectors.

    Args:
    - ctx (Context): Context for search.
    - doc_vecs (dict[str, np.ndarray]): Dictionary containing document vectors.
    - vec_space (Sequence[str]): Vector space vocabulary.
    - ii (InvertedIndex): Inverted index for document retrieval.
    - metadata (Metadata): Metadata related to the query and documents.
    - query_terms (Sequence[QueryTerm]): Query terms.

    Returns:
    - SearchResults: List of search results.
    """
    query_vec = vectorise_query(ctx, query_terms, vec_space, ii, metadata)
    results = defaultdict(float)

    for doc, vec in doc_vecs.items():
        results[doc] = cosine_similarity(query_vec, vec)

    return sorted(results.items(), key=lambda x: x[1], reverse=True)[:10]


@timeit
def search_idf(
    ctx: Context,
    inverted_index: InvertedIndex,
    metadata: Metadata,
    query_terms: Sequence[QueryTerm],
) -> SearchResults:
    """
    Perform search using Inverted Document Frequency (IDF) scoring.

    Args:
    - ctx (Context): Context for search.
    - inverted_index (InvertedIndex): Inverted index for document retrieval.
    - metadata (Metadata): Metadata related to the query and documents.
    - query_terms (Sequence[QueryTerm]): Query terms.

    Returns:
    - SearchResults: List of search results.
    """
    results: defaultdict[str, float] = defaultdict(float)

    for query in query_terms:
        if query.term in inverted_index:
            token = inverted_index[query.term]

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
                        score * occurrence.weight * query.weight
                    )
                else:
                    results[occurrence.filename] += score

    return sorted(results.items(), key=lambda x: x[1], reverse=True)[:10]


def print_result(ctx: Context, result: SearchResult, metadata: Metadata) -> None:
    """
    Print search result information for a single document.

    Args:
    - ctx (Context): Context for search.
    - result (SearchResult): Tuple containing document ID and its score.
    - metadata (Metadata): Metadata related to the documents.

    Returns:
    - None
    """
    doc, score = result

    if doc in metadata["files"] and metadata["files"][doc]["info"] is not None:
        info: dict[str, str] = metadata["files"][doc]["info"] or {}
        for key, val in info.items():
            if "url" in key:
                console.print(
                    f"[bold]{key}[/bold]: [link=file://{os.getcwd()}/{val.replace(r'/ps2.gamespy.com', '')}]{val}[/link]"
                )
            else:
                console.print(f"[bold]{key}[/bold]: {val}")

    else:
        console.print(f"[blue]Document: {doc}[/blue]")

    if ctx.verbose:
        console.print(f"[italic]Score:[/italic] {score}")

    console.print("\n")


def print_results(ctx: Context, results: SearchResults, metadata: Metadata) -> None:
    """
    Print search results for multiple documents.

    Args:
    - ctx (Context): Context for search.
    - results (List[SearchResult]): List of tuples containing document IDs and scores.
    - metadata (Metadata): Metadata related to the documents.

    Returns:
    - None
    """
    if len(results) == 0:
        console.print("[red]No results found[/red]")
        return

    for i, (doc, score) in enumerate(results):
        console.print(f"[green]Result {int(i) + 1}[/green]:")
        print_result(ctx, (doc, score), metadata)


def generate_vocab(ctx: Context, vocab: InvertedIndex) -> None:
    """
    Generate a vocabulary file from an inverted index.

    Args:
    - ctx (Context): Context for vocabulary generation.
    - vocab (InvertedIndex): Inverted index containing vocabulary information.

    Returns:
    - None
    """
    if ctx.verbose:
        logger.info("Generating dictionary")

    with open(VOCAB_PATH, "w") as f:
        for word, token in vocab.items():
            f.write(f"{word} {token.count}\n")

    sym_spell.load_dictionary(VOCAB_PATH, term_index=0, count_index=1)


def search(ctx: Context) -> None:
    """
    Perform a search operation based on the provided context.

    Args:
    - ctx (Context): Context for search.

    Returns:
    - None
    """
    ii, doc_vectors, vec_space = index_documents(ctx)

    if ctx.spellcheck:
        generate_vocab(ctx, ii)

    metadata: Metadata = get_metadata(ctx)

    match ctx.searcher:
        case "vector":
            search_func = partial(
                search_vecs,
                ctx,
                doc_vectors,
                vec_space,
                ii,
                metadata,
            )
        case "score":
            search_func = partial(search_idf, ctx, ii, metadata)
        case _:
            assert_never("Unreachable")

    while True:
        user_input = get_input(ctx, metadata)

        with console.status("Searching..."):
            results: SearchResults = search_func(user_input)

        print_results(ctx, results, metadata)
