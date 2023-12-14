from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer, StemmerI
import string
from symspellpy import SymSpell
from rich.console import Console

# Initializing WordNetLemmatizer from NLTK for lemmatization
lemmatizer: WordNetLemmatizer = WordNetLemmatizer()

# Creating a set of English stopwords using NLTK
stop_words: set[str] = set(stopwords.words("english"))

# Defining a string containing punctuation marks to be removed from text
punctuation: str = string.punctuation + "♥•’‘€–"

# Initializing PorterStemmer from NLTK for stemming
stemmer: StemmerI = PorterStemmer()

# Creating a translation table to remove punctuation marks using str.maketrans()
translator: dict[int, int | None] = str.maketrans("", "", punctuation)

# Initializing SymSpell for spell checking and correction
sym_spell: SymSpell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

# Initializing Rich Console for enhanced console output
console: Console = Console(color_system="auto")
