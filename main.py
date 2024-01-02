#!./.direnv/python-3.11/bin/python
import logging
from config import LOG_LEVEL
from dataclasses import dataclass
from typing import Literal
from argparse import Namespace
from args_parse import parser
from parser.types import StripperType
from rich.logging import RichHandler

ParserType = Literal["async", "mp", "mt", "sync"]
ScorerType = Literal["tfidf", "bm25", "bm25+"]
SearcherType = Literal["vector", "score"]


@dataclass(
    frozen=True,
    slots=True,
)
class Context:
    """
    Dataclass to store configuration settings for the search.
    """

    reindex: bool = False
    stripper: StripperType = "lemmatize"
    parser_type: ParserType = "async"
    scorer: ScorerType = "bm25+"
    searcher: SearcherType = "vector"
    expand: bool = True
    spellcheck: bool = True
    weighted: bool = True
    entities: bool = True
    verbose: bool = False


FORMAT = "%(message)s"

# Configuring logging with RichHandler
logging.basicConfig(
    level=LOG_LEVEL,
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)


def generate_context(args: Namespace) -> Context:
    """
    Generates a Context object based on parsed command-line arguments.

    Args:
        args (Namespace): Parsed command-line arguments.

    Returns:
        Context: Context object with configured settings.
    """
    if args.all:
        return Context(
            stripper=args.stripper,
            reindex=args.regen,
            verbose=args.verbose,
            scorer=args.scorer,
            searcher=args.searcher,
        )

    return Context(
        parser_type=args.parser_type,
        spellcheck=args.spellcheck,
        expand=args.expand,
        weighted=args.weight,
        stripper=args.stripper,
        reindex=args.regen,
        verbose=args.verbose,
        scorer=args.scorer,
        searcher=args.searcher,
        entities=args.entities,
    )


def main() -> None:
    """
    Main function to initiate search based on provided arguments.
    """
    from search.search import search

    args = parser.parse_args()

    ctx = generate_context(args)

    if ctx.verbose:
        logger.info("Starting up...")
        logger.info(ctx)

    search(ctx)


if __name__ == "__main__":
    main()
