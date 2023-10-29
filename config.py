from pathlib import Path
import logging
import pkg_resources

VIDEOGAMES_DIR: Path = Path("./videogames")
VIDEOGAME_LABELS: Path = Path("./videogame-labels.csv")

CACHE_SIZE: int = 100

LOG_LEVEL = logging.DEBUG

PICKLE_FILE: Path = Path("./data.pkl")

DICTIONARY_PATH = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt"
)
