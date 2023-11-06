#!./.direnv/python-3.11/bin/python
from search.search import search
import logging
from config import LOG_LEVEL
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import asyncio

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

parser = ArgumentParser(
    prog="Information Retrieval",
    description="Search a collection of documents",
    formatter_class=ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--stripper",
    "-s",
    default="lemmatize",
    type=str,
    help="Choose whether to lemmatize or stem",
    nargs=1,
    choices=["lemmatize", "stem"],
    dest="stripper",
)

parser.add_argument("--regen", "-r", help="Reindex the files")

args = parser.parse_args()


async def main() -> None:
    logger.info("Starting up...")
    await search()


if __name__ == "__main__":
    asyncio.run(main())
