from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer, StemmerI
import string
from symspellpy import SymSpell
from config import DICTIONARY_PATH


lemmatizer: WordNetLemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
punctuation: str = string.punctuation + "♥•’‘€–"
stemmer: StemmerI = PorterStemmer()

translator = str.maketrans("", "", punctuation)

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
