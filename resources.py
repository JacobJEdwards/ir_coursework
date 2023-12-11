from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer, StemmerI
import string
from symspellpy import SymSpell
from rich.console import Console


lemmatizer: WordNetLemmatizer = WordNetLemmatizer()
stop_words: set[str] = set(stopwords.words("english"))
punctuation: str = string.punctuation + "♥•’‘€–"
stemmer: StemmerI = PorterStemmer()

translator: dict[int, int | None] = str.maketrans("", "", punctuation)

sym_spell: SymSpell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

console: Console = Console(color_system="auto")
