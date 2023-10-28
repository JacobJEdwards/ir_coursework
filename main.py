from search.search import search
import logging
from config import LOG_LEVEL

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Starting up...")
    search()


if __name__ == "__main__":
    main()
