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
import logging
from search.tf_idf import calculate_tf, calculate_idf

logger = logging.getLogger(__name__)


lemmatizer: WordNetLemmatizer = WordNetLemmatizer()
stemmer: StemmerI = PorterStemmer()
stop_words: Set[str] = set(stopwords.words("english"))
punctuation: str = string.punctuation + "♥•’‘€–"

translator = str.maketrans("", "", punctuation)

# named tuple to group together occurrences of a word in a document
DocOccurrences = namedtuple("DocOccurrences", "filename word num_occ tf")


class TrieNode:
    def __init__(self):
        self.children: Dict[str, TrieNode] = defaultdict(TrieNode)
        self.occurrences: List[DocOccurrences] = []


class II:
    def __init__(self):
        self.documents = None
        self.root: TrieNode = TrieNode()

    def insert(self, occurrence):
        node = self.root
        for char in occurrence.word:
            node = node.children[char]
        node.occurrences.append(occurrence)

    def calculate_idf(self, total_docs: int):
        def dfs(node: TrieNode, doc_count: int):
            for occ in node.occurrences:
                token = Token(occ.filename, 0, 0, [])
                token.count += occ.num_occ
                token.idf = calculate_idf(total_docs, doc_count)
                self.documents[occ.word] = token
            for child in node.children.values():
                dfs(child, doc_count + len(node.occurrences))

        dfs(self.root, 0)

    def search(self, query):
        results = defaultdict(int)

        def dfs(node, current_word, pattern_index):
            if pattern_index == len(query):
                if node.occurrences:
                    for doc in node.occurrences:
                        results[doc.filename] += doc.tf * self.documents[doc.filename].i
                    results.update(node.occurrences)
                for char, child_node in node.children.items():
                    dfs(child_node, current_word + char, pattern_index)
            elif query[pattern_index] == '*':
                for char, child_node in node.children.values():
                    dfs(child_node, current_word + char, pattern_index)
            elif query[pattern_index] in node.children:
                next_node = node.children[query[pattern_index]]
                dfs(next_node, current_word + query[pattern_index], pattern_index + 1)

        dfs(self.root, '', 0)

        return results

    def __str__(self):
        return self._traverse(self.root, "")

    def _traverse(self, node, current_word):
        result = ""
        if node.occurrences:
            result += f"Word: {current_word}, Document Occurrences: {node.occurrences}\n"
        for char, child_node in node.children.items():
            result += self._traverse(child_node, current_word + char)
        return result


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
    idf: float
    occurrences: List[DocOccurrences]


InvertedIndex = Dict[str, Token]


# eventually replace with nltk.probabilty.FreqDist
def parse_contents(file: BinaryIO, parser="lxml") -> List[DocOccurrences]:
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

    total_words = len(filtered_text)
    counts = get_count(filtered_text)

    return [
        DocOccurrences(filename=file.name, word=name, num_occ=count, tf=calculate_tf(total_words, count))
        for name, count in counts.items()
    ]


def get_count(words: List[str]) -> Dict[str, int]:
    return {name: words.count(name) for name in set(words)}


def merge_word_count_dicts(doc_dict: Dict[str, List[DocOccurrences]], total_docs: int) -> InvertedIndex:
    merged_dict: InvertedIndex = {}
    index = II()

    # Use defaultdict to store occurrences

    for name, occurrences in doc_dict.items():
        for occ in occurrences:
            index.insert(occ)
            if occ.word in merged_dict:
                merged_dict[occ.word].count += occ.num_occ
                merged_dict[occ.word].occurrences.append(occ)
            else:
                merged_dict[occ.word] = Token(occ.word, occ.num_occ, 0, [occ])

    for word, token in merged_dict.items():
        doc_count = len(token.occurrences)
        token.idf = calculate_idf(total_docs, doc_count)

    logger.debug("Generated inverted index")

    return merged_dict, index


def pickle_obj(data: InvertedIndex) -> None:
    try:
        with open(PICKLE_FILE, "wb") as f:
            pickle.dump(data, f)
    except Exception as e:
        logger.critical(e)
        exit(1)


def depickle_obj() -> InvertedIndex:
    try:
        with open(PICKLE_FILE, "rb") as f:
            data = pickle.load(f)
            return data
    except Exception as e:
        logger.critical(e)
        exit(1)
