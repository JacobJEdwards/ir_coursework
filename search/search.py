from typing import List, Tuple
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
from parser.parser import depickle_obj


lemmatizer: WordNetLemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
punctuation: str = string.punctuation + "♥•’‘€–"

translator = str.maketrans("", "", punctuation)


def clean_input(user_input: str) -> List[str]:
    tokens = word_tokenize(user_input.translate(translator))
    return [lemmatizer.lemmatize(word.lower()) for word in tokens if word not in stop_words]


def get_input() -> List[str]:
    user_input = input("Enter search term(s)")
    return clean_input(user_input)


# need to add tf_idf -> think how many need to change implemenation (add metadata etc)
def search_idf(query_terms: List[str]) -> List[Tuple[str, int]]:
    inverted_index = depickle_obj()
    results = {}
    for term in query_terms:
        if term in inverted_index:
            token = inverted_index[term]
            for occurrence in token.occurrences:
                if occurrence[0] in results:
                    results[occurrence[0]] += occurrence[1]
                else:
                    results[occurrence[0]] = occurrence[1]

    return sorted(results.items(), key=lambda x: x[1], reverse=True)


def search() -> None:
    while True:
        user_input = get_input()
        results = search_idf(user_input)
        for doc, score in results:
            print(f"Document: {doc}, Score: {score}")
