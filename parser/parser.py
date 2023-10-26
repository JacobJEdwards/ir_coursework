import pickle

from bs4 import BeautifulSoup
from bs4.element import Comment
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer, StemmerI
from nltk.corpus import stopwords
import string
from typing import List, Dict, Set, BinaryIO, Tuple
from dataclasses import dataclass
from collections import namedtuple
from config import PICKLE_FILE
from collections import defaultdict


lemmatizer: WordNetLemmatizer = WordNetLemmatizer()
stemmer: StemmerI = PorterStemmer()
stop_words: Set[str] = set(stopwords.words("english"))
punctuation: str = string.punctuation + "♥•’‘€–"

translator = str.maketrans("", "", punctuation)

# named tuple to group together occurrences of a word in a document
DocOccurrences = namedtuple("DocOccurrences", "filename num_occ")


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


InvertedIndex = Dict[str, Token]


# eventually replace with nltk.probabilty.FreqDist
def parse_contents(file: BinaryIO, parser="lxml") -> Tuple[Dict[str, int], int]:
    soup = BeautifulSoup(file.read(), features=parser)
    # remove unneeded info
    comments = soup.find_all(string=lambda element: isinstance(element, Comment))
    # should i remove less ? maybe after adding weight
    [s.extract() for s in soup(["script", "style", "iframe", "a", "img"])]
    [c.extract() for c in comments]

    text = soup.get_text()
    text = text.translate(translator)

    filtered_text = [
        # stemmer.stem(word.lower())
        lemmatizer.lemmatize(word.lower())
        for word in word_tokenize(text)
        if word not in stop_words
    ]

    return get_count(filtered_text)


def get_count(words: List[str]) -> Tuple[Dict[str, int], int]:
    total_words = len(words)
    return (
        {
            name: words.count(name) for name in set(words)
        },
        total_words
    )


def merge_word_count_dicts(doc_dict: Dict[str, Dict[str, int]]) -> InvertedIndex:
    merged_dict: InvertedIndex = {}

    # Use defaultdict to store occurrences
    occurrences_dict = defaultdict(list)

    for name, occurrences in doc_dict.items():
        for word, count in occurrences.items():
            if word in merged_dict:
                merged_dict[word].count += count
                occurrences_dict[word].append(DocOccurrences(name, count))
            else:
                merged_dict[word] = Token(word, count, [])
                occurrences_dict[word].append(DocOccurrences(name, count))

    # Update occurrences in merged_dict
    for word, occurrences in occurrences_dict.items():
        merged_dict[word].occurrences = sorted(occurrences, key=lambda occ: occ.num_occ, reverse=True)

    return merged_dict


def pickle_obj(data: InvertedIndex) -> None:
    try:
        with open(PICKLE_FILE, "wb") as f:
            pickle.dump(data, f)
    except Exception as e:
        print(e)
        exit(1)


def depickle_obj() -> InvertedIndex:
    try:
        with open(PICKLE_FILE, "rb") as f:
            data = pickle.load(f)
            return data
    except Exception as e:
        print(e)
        exit(1)
