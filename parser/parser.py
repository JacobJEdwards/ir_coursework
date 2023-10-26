import pickle

from bs4 import BeautifulSoup, Comment
from typing import BinaryIO
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer, StemmerI
from nltk.corpus import stopwords
import string
from typing import List, Dict, Set
from dataclasses import dataclass
from collections import namedtuple
from config import PICKLE_FILE


lemmatizer: WordNetLemmatizer = WordNetLemmatizer()
stemmer: StemmerI = PorterStemmer()
stop_words: Set[str] = set(stopwords.words("english"))
punctuation: str = string.punctuation + "♥•’‘€–"

translator = str.maketrans("", "", punctuation)

# named tuple to group together occurrences of a word in a document
DocOccurrences = namedtuple("DocOccurrences", "filename count")


# doc token represents an instance of a word in a particular document
@dataclass
class DocToken:
    word: str
    count: int
    position: int


# token represents
@dataclass
class Token:
    word: str
    count: int
    occurrences: List[DocOccurrences]


def parse_contents(file: BinaryIO, parser="lxml") -> Dict[str, int]:
    soup = BeautifulSoup(file.read(), features=parser)
    # remove unneeded info
    comments = soup.find_all(string=lambda element: isinstance(element, Comment))
    [s.extract() for s in soup(["script", "style", "iframe", "img", "a"])]
    [c.extract() for c in comments]

    text = soup.get_text()
    text = text.translate(translator)

    filtered_text = [
        #stemmer.stem(word.lower())
        lemmatizer.lemmatize(word.lower())
        for word in word_tokenize(text)
        if word not in stop_words
    ]

    return get_count(filtered_text)


def get_count(words: List[str]) -> Dict[str, int]:
    return {
        name: words.count(name) for name in set(words)
    }


# Define a function to merge the word count dictionaries
def merge_word_count_dicts(doc_dict: Dict[str, Dict[str, int]]) -> Dict[str, Token]:
    merged_dict = {}

    for name, occurrences in doc_dict.items():
        for word, count in occurrences.items():
            if word in merged_dict:
                merged_dict[word].count += count
                merged_dict[word].occurrences.append(DocOccurrences(name, count))
            else:
                merged_dict[word] = Token(word, count, [
                    DocOccurrences(name, count)
                ])

    return merged_dict


def pickle_obj(data: Dict[str, Token]) -> None:
    with open(PICKLE_FILE, "wb") as f:
        pickle.dump(data, f)


def depickle_obj() -> Dict[str, Token]:
    with open(PICKLE_FILE, "rb") as f:
        data = pickle.load(f)
        return data
