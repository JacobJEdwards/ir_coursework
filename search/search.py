from typing import List, Tuple
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
from parser.parser import depickle_obj
from config import VIDEOGAMES_DIR
from search.tf_idf import calculate_idf, calculate_tf_idf, calculate_tf
import json
from pprint import pprint


SearchResults = List[Tuple[str, float]]


lemmatizer: WordNetLemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
punctuation: str = string.punctuation + "♥•’‘€–"

translator = str.maketrans("", "", punctuation)


def clean_input(user_input: str) -> List[str]:
    tokens = word_tokenize(user_input.translate(translator))
    return [
        lemmatizer.lemmatize(word.lower()) for word in tokens if word not in stop_words
    ]


def get_input() -> List[str]:
    user_input = input("Enter search term(s)")
    return clean_input(user_input)


# need to add tf_idf -> think how many need to change implemenation (add metadata etc)
def search_idf(query_terms: List[str]) -> SearchResults:
    inverted_index = depickle_obj()
    with open("metadata.json", "r") as f:
        metadata = json.load(f)

    results = {}
    total_docs = len(list(VIDEOGAMES_DIR.iterdir()))

    # clean this up its not nice
    for term in query_terms:
        if term in inverted_index:
            token = inverted_index[term]

            docs_with_term = len(token.occurrences)
            idf = calculate_idf(total_docs, docs_with_term)

            for occurrence in token.occurrences:
                tf = calculate_tf(metadata[occurrence[0]]["word_count"], occurrence[1])
                tf_idf = calculate_tf_idf(tf, idf)

                if occurrence[0] in results:
                    results[occurrence[0]] += tf_idf
                else:
                    results[occurrence[0]] = tf_idf

    return sorted(results.items(), key=lambda x: x[1], reverse=True)[:10]


def search() -> None:
    while True:
        user_input = get_input()
        results = search_idf(user_input)
        for doc, score in results:
            print(f"Document: {doc}, Score: {score}")
