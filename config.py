from pathlib import Path
from typing import Final
import logging
import pkg_resources

# Path variables for directory and files related to dataset
VIDEOGAMES_DIR: Path = Path("./videogame")
VIDEOGAME_LABELS: Path = Path("./videogame-labels.csv")

# Constants
CACHE_SIZE: Final = 100
LOG_LEVEL: Final = logging.INFO

# File paths for pickled data and directory to store pickles
PICKLE_FILE: Path = Path("./data.pkl")
PICKLE_DIR: Path = Path("./pickles")

# Path to the dictionary file used for spell checking
DICTIONARY_PATH: Final = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt"
)

# Path for storing a custom vocabulary or dictionary
VOCAB_PATH: Path = Path("./dictionary.txt")
