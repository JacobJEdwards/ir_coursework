from typing import List, Tuple, Dict, Set
from nltk import word_tokenize
from nltk.corpus import wordnet
from parser.reader import generate_object
from parser.parser import InvertedIndex
from search.tf_idf import calculate_tf_idf
import json
from collections import defaultdict
from resources import lemmatizer, stop_words, translator, sym_spell
from symspellpy import Verbosity


SearchResults = List[Tuple[str, float]]
# TYPE METADATA


def get_suggestions(tokens: Set[str]) -> Set[str]:
    new_tokens = set(tokens)
    for token in tokens:
        suggestions = sym_spell.lookup(
            token, Verbosity.CLOSEST, max_edit_distance=2, include_unknown=True
        )
        for suggestion in suggestions:
            new_tokens.add(suggestion.term)
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


# need to add tf_idf -> think how many need to change implemenation (add metadata etc)
# @lru_cache(CACHE_SIZE)
def search_idf(
    query_terms: Set[str], inverted_index: InvertedIndex, metadata: Dict
) -> SearchResults:
    results = defaultdict(float)

    # clean this up its not nice
    for term in query_terms:
        if term in inverted_index:
            token = inverted_index[term]

            for occurrence in token.occurrences:
                tf_idf = calculate_tf_idf(occurrence.tf, token.idf)
                results[occurrence.filename] += tf_idf

    return sorted(results.items(), key=lambda x: x[1], reverse=True)[:10]


def search() -> None:
    ii = generate_object()

    with open("metadata.json", "r") as f:
        metadata = json.load(f)

    while True:
        user_input = get_input()
        results = search_idf(user_input, ii, metadata)

        for doc, score in results:
            print(f"Document: {doc}, Score: {score}")
