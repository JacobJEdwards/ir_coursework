from pathlib import Path
from typing import Final
import logging
import pkg_resources

VIDEOGAMES_DIR: Path = Path("./videogame")
VIDEOGAME_LABELS: Path = Path("./videogame-labels.csv")

CACHE_SIZE: Final = 100

LOG_LEVEL: Final = logging.INFO

PICKLE_FILE: Path = Path("./data.pkl")
PICKLE_DIR: Path = Path("./pickles")

DICTIONARY_PATH: Final = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt"
)
