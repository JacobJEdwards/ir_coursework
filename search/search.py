from nltk import word_tokenize
from parser.reader import timeit
from nltk.corpus import wordnet
from parser.parser import filter_tokens, get_strip_func
from parser.types import InvertedIndex, DocumentMatrix, Metadata
from collections import defaultdict
from resources import translator, sym_spell
from symspellpy import Verbosity
import numpy as np
from main import Context
from parser.reader import get_metadata, index_documents
import logging
from search.types import SearchResults
from config import VOCAB_PATH

logger = logging.getLogger(__name__)


def get_suggestions(tokens: set[str], print_terms: bool = False) -> set[str]:
    new_tokens = set(tokens)
    for token in tokens:
        suggestions = sym_spell.lookup(
            token, Verbosity.CLOSEST, max_edit_distance=2, include_unknown=True
        )

        if suggestions[0].term == token or suggestions[0].distance == 0:
            new_tokens.add(suggestions[0].term)
            continue

        if print_terms:
            logger.info(f"Spell suggestions for {token}:")
            for sug in suggestions:
                logger.info(f"{sug.term}")

            print()

        for i, suggestion in enumerate(suggestions):
            if i == 3:
                break

            print("Did you mean: ", suggestion.term)
            if input("Y/N: ").lower() == "y":
                new_tokens.add(suggestion.term)
                break
            else:
                new_tokens.add(token)

    return new_tokens


def expand_query(query_terms: set[str], print_terms: bool = False) -> set[str]:
    new_terms = set(query_terms)
    for term in query_terms:
        syns = wordnet.synsets(term)
        for syn in syns:
            for lemma in syn.lemmas():
                new_terms.add(lemma.name())

    if print_terms:
        logger.info(f"Original terms: {query_terms}\nExpanded terms: {new_terms}\n")
    return new_terms


def _clean_tokenized_input(
    ctx: Context, tokens: set[str], metadata: Metadata
) -> set[str]:
    if ctx.spellcheck:
        tokens = get_suggestions(tokens, ctx.verbose)

    if ctx.expand:
        tokens = expand_query(tokens, ctx.verbose)

    tokens = filter_tokens(get_strip_func(metadata["stripper"]), tokens)

    if ctx.verbose:
        logger.info(f"Final tokens: {tokens}\n")

    return set(tokens)


def clean_input(ctx: Context, user_input: str, metadata: Metadata) -> set[str]:
    tokens = word_tokenize(user_input.translate(translator))
    return _clean_tokenized_input(ctx, tokens, metadata)


def get_input(ctx: Context, metadata: Metadata) -> set[str]:
    user_input = input("Enter search term(s): ")
    print()
    return clean_input(ctx, user_input, metadata)


def cosine_similarity(query_vector: np.ndarray, doc_vector: np.ndarray) -> np.ndarray:
    dot_product = np.dot(query_vector, doc_vector.T)
    query_norm = np.linalg.norm(query_vector)

    doc_norms = np.linalg.norm(doc_vector, axis=1)

    return dot_product / (query_norm * doc_norms)


# need to add tf_idf -> think how many need to change implemenation (add metadata etc)
# @lru_cache(CACHE_SIZE)
def search_idf(
    ctx: Context,
    query_terms: set[str],
    inverted_index: InvertedIndex,
    doc_matrix: DocumentMatrix,
    metadata: Metadata,
) -> SearchResults:
    results = defaultdict(float)

    for term in query_terms:
        if term in inverted_index:
            token = inverted_index[term]

            for occurrence in token.occurrences:
                if ctx.weighted:
                    results[occurrence.filename] += occurrence.tfidf * occurrence.weight
                else:
                    results[occurrence.filename] += occurrence.tfidf

    return sorted(results.items(), key=lambda x: x[1], reverse=True)[:10]


def print_results(ctx: Context, results: SearchResults, metadata: Metadata) -> None:
    if len(results) == 0:
        print("No results found")
        return

    for i, (doc, score) in enumerate(results):
        print(f"{int(i)+1}:")
        if doc in metadata["files"]:
            info: dict = metadata["files"][doc]["info"]
            for key, val in info.items():
                print(f"{key}: {val}")

            print(f"Score: {score}")
        else:
            print(f"Document: {doc}, Score: {score}")

        print()


def generate_vocab(ctx: Context, vocab: InvertedIndex) -> None:
    if ctx.verbose:
        logger.info("Generating dictionary")

    with open(VOCAB_PATH, "w") as f:
        for word, token in vocab.items():
            f.write(f"{word} {token.count}\n")

    sym_spell.load_dictionary(VOCAB_PATH, term_index=0, count_index=1)


def search(ctx: Context) -> None:
    ii, doc_matrix = index_documents(ctx)

    if ctx.spellcheck:
        generate_vocab(ctx, ii)

    metadata: Metadata = get_metadata(ctx)

    while True:
        user_input = get_input(ctx, metadata)

        results: SearchResults = search_idf(ctx, user_input, ii, doc_matrix, metadata)

        print_results(ctx, results, metadata)
