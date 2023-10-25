from pathlib import Path
from bs4 import BeautifulSoup, Comment
from typing import BinaryIO
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer, StemmerI
from nltk.corpus import stopwords
import string
from typing import List

lemmatizer: WordNetLemmatizer = WordNetLemmatizer()
stemmer: StemmerI = PorterStemmer()
stop_words = set(stopwords.words("english"))

translator = str.maketrans("", "", string.punctuation)


def parse_contents(file: BinaryIO, parser="lxml") -> List[str]:
    soup = BeautifulSoup(file.read(), features=parser)
    # remove unneeded info
    comments = soup.find_all(string=lambda element: isinstance(element, Comment))
    [s.extract() for s in soup(["script", "style", "iframe", "img", "a"])]
    [c.extract() for c in comments]

    text = soup.get_text()
    text = text.translate(translator)

    filtered_text = [
        lemmatizer.lemmatize(word)
        for word in word_tokenize(text)
        if word not in stop_words
    ]

    return filtered_text
