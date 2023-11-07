#!./.direnv/python-3.11/bin/python
import logging
from config import LOG_LEVEL
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from dataclasses import dataclass
from typing import Literal


@dataclass
class Context:
    reindex: bool = False
    stripper: Literal["lemmatize", "stem"] = "lemmatize"
    parser_type: Literal["async", "mp", "sync"] = "async"
    expand: bool = True
    spellcheck: bool = True
    weighted: bool = True
    verbose: bool = False


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
    choices=["lemmatize", "stem"],
    dest="stripper",
)

parser.add_argument("--regen", "-r", action="store_true", help="Reindex the files")

parser.add_argument(
    "--parser",
    "-p",
    help="choose how files are parsed",
    type=str,
    dest="parser_type",
    choices=["sync", "async", "mp"],
    default="async",
)

parser.add_argument(
    "--spellcheck",
    "-sc",
    action="store_true",
    dest="spellcheck",
    help="Suggest spelling corrections",
)

parser.add_argument("--expand", "-e", action="store_true", help="Expand query")

parser.add_argument(
    "--weighted",
    "-w",
    action="store_true",
    dest="weight",
    help="add weighting for document elements",
)

parser.add_argument(
    "--all", "-a", action="store_true", dest="all", help="enable most optimizations"
)

parser.add_argument(
    "--verbose", "-v", action="store_true", dest="verbose", help="increase verbosity"
)

args = parser.parse_args()


def main() -> None:
    from search.search import search

    if args.all:
        ctx = Context(stripper=args.stripper, reindex=args.regen, verbose=args.verbose)
    else:
        ctx = Context(
            parser_type=args.parser_type,
            spellcheck=args.spellcheck,
            expand=args.expand,
            weighted=args.weight,
            stripper=args.stripper,
            reindex=args.regen,
            verbose=args.verbose,
        )

    if ctx.verbose:
        logger.info("Starting up...")
        logger.info(ctx)
    search(ctx)


if __name__ == "__main__":
    main()
