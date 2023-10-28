from typing import List, Tuple, Dict
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
from parser.reader import generate_object
from parser.parser import InvertedIndex
from search.tf_idf import calculate_idf, calculate_tf_idf, calculate_tf
import json
from collections import defaultdict


SearchResults = List[Tuple[str, float]]
# TYPE METADATA

lemmatizer: WordNetLemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
punctuation: str = string.punctuation + "♥•’‘€–"

translator = str.maketrans("", "", punctuation)


def clean_input(user_input: str) -> List[str]:
    return [
        lemmatizer.lemmatize(word.lower())
        for word in word_tokenize(user_input.translate(translator))
        if word not in stop_words
    ]


def get_input() -> List[str]:
    user_input = input("Enter search term(s)")
    return clean_input(user_input)


# need to add tf_idf -> think how many need to change implemenation (add metadata etc)
# @lru_cache(CACHE_SIZE)
def search_idf(
    query_terms: Tuple, inverted_index: InvertedIndex, metadata: Dict
) -> SearchResults:
    results = defaultdict(float)
    total_docs = metadata["total_docs"]

    # clean this up its not nice
    for term in query_terms:
        if term in inverted_index:
            token = inverted_index[term]

            docs_with_term = len(token.occurrences)
            idf = calculate_idf(total_docs, docs_with_term)

            for occurrence in token.occurrences:
                tf_idf = calculate_tf_idf(occurrence.tf, idf)
                results[occurrence.filename] += tf_idf

    return sorted(results.items(), key=lambda x: x[1], reverse=True)[:10]


def search() -> None:
    ii = generate_object()

    with open("metadata.json", "r") as f:
        metadata = json.load(f)

    while True:
        user_input = get_input()
        results = search_idf(tuple(user_input), ii, metadata)

        for doc, score in results:
            print(f"Document: {doc}, Score: {score}")
