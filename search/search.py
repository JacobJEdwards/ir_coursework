from typing import List, Tuple, Dict, Set
from nltk import word_tokenize
from nltk.corpus import wordnet
from parser.reader import generate_object
from parser.parser import InvertedIndex, DocumentMatrix
from search.tf_idf import calculate_tf_idf, calculate_tf
import json
from collections import defaultdict
from resources import lemmatizer, stop_words, translator, sym_spell
from symspellpy import Verbosity
import numpy as np


SearchResults = List[Tuple[str, float]]
# TYPE METADATA


def get_suggestions(tokens: Set[str]) -> Set[str]:
    new_tokens = set(tokens)
    for token in tokens:
        suggestions = sym_spell.lookup(
            token, Verbosity.CLOSEST, max_edit_distance=2, include_unknown=True
        )

        if suggestions[0].term == token:
            new_tokens.add(token)
            continue

        if suggestions[0].distance == 0:
            new_tokens.add(suggestions[0].term)
            continue

        for i, suggestion in enumerate(suggestions):
            if i == 3:
                break

            if suggestion.term == token:
                new_tokens.add(token)
                continue

            print("Did you mean: ", suggestion.term)
            if input("Y/N: ").lower() == "y":
                new_tokens.add(suggestion.term)
                break
            else:
                new_tokens.add(token)

    return new_tokens


def clean_input(user_input: str) -> Set[str]:
    tokens = word_tokenize(user_input.translate(translator))
    return set(
        [
            lemmatizer.lemmatize(word.lower())
            for word in expand_query(get_suggestions(set(tokens)))
            if word not in stop_words
        ]
    )


def expand_query(query_terms: Set[str]) -> Set[str]:
    new_terms = set(query_terms)
    for term in query_terms:
        syns = wordnet.synsets(term)
        for syn in syns:
            for lemma in syn.lemmas():
                new_terms.add(lemma.name())
    return new_terms


def get_input() -> set[str]:
    user_input = input("Enter search term(s)")
    return clean_input(user_input)


def cosine_similarity(query_vector: np.ndarray, doc_vector: np.ndarray) -> np.ndarray:
    dot_product = np.dot(query_vector, doc_vector.T)
    query_norm = np.linalg.norm(query_vector)

    doc_norms = np.linalg.norm(doc_vector, axis=1)

    return dot_product / (query_norm * doc_norms)


# need to add tf_idf -> think how many need to change implemenation (add metadata etc)
# @lru_cache(CACHE_SIZE)
def search_idf(
    query_terms: Set[str],
    inverted_index: InvertedIndex,
    doc_matrix: DocumentMatrix,
    metadata: Dict,
) -> SearchResults:
    results = defaultdict(float)

    input_tfidf = defaultdict(float)
    for term in query_terms:
        input_tfidf[term] += 1

    for term in input_tfidf:
        input_tfidf[term] = calculate_tf(len(query_terms), int(input_tfidf[term]))

    for term in query_terms:
        if term in inverted_index:
            token = inverted_index[term]

            for occurrence in token.occurrences:
                # use doc matrix and cosine similarity
                input_tfidf[term] = calculate_tf_idf(input_tfidf[term], token.idf)
                results[occurrence.filename] += occurrence.tfidf * occurrence.weight

    return sorted(results.items(), key=lambda x: x[1], reverse=True)[:10]


async def search() -> None:
    ii, doc_matrix = await generate_object()

    with open("metadata.json", "r") as f:
        metadata = json.load(f)

    while True:
        user_input = get_input()

        results: SearchResults = search_idf(user_input, ii, doc_matrix, metadata)

        if len(results) == 0:
            print("No results found")
            continue

        for doc, score in results:
            print(f"Document: {doc}, Score: {score}")
