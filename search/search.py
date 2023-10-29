from typing import List, Tuple, Dict, Set
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords, wordnet
import string
from parser.reader import generate_object
from parser.parser import InvertedIndex
from search.tf_idf import calculate_tf_idf
import json
from collections import defaultdict


SearchResults = List[Tuple[str, float]]
# TYPE METADATA

lemmatizer: WordNetLemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
punctuation: str = string.punctuation + "♥•’‘€–"

translator = str.maketrans("", "", punctuation)


def clean_input(user_input: str) -> List[str]:
    tokens = word_tokenize(user_input.translate(translator))
    return [
        lemmatizer.lemmatize(word.lower())
        for word in expand_query(set(tokens))
        if word not in stop_words
    ]


def expand_query(query_terms: Set[str]) -> Set[str]:
    new_terms = set(query_terms)
    for term in query_terms:
        syns = wordnet.synsets(term)
        for syn in syns:
            for lemma in syn.lemmas():
                new_terms.add(lemma.name())
    return new_terms


def get_input() -> List[str]:
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
        results = search_idf(set(user_input), ii, metadata)

        for doc, score in results:
            print(f"Document: {doc}, Score: {score}")
