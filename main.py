#!./.direnv/python-3.11/bin/python
import logging
from config import LOG_LEVEL
from dataclasses import dataclass
from typing import Literal
from args_parse import parser


@dataclass
class Context:
    reindex: bool = False
    stripper: Literal["lemmatize", "stem"] = "lemmatize"
    parser_type: Literal["async", "mp", "sync"] = "async"
    scorer: Literal["tfidf", "bm25", "bm25+"] = "bm25+"
    expand: bool = True
    spellcheck: bool = True
    weighted: bool = True
    verbose: bool = False


logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)


def generate_context(args) -> Context:
    if args.all:
        return Context(
            stripper=args.stripper,
            reindex=args.regen,
            verbose=args.verbose,
            scorer=args.scorer,
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
    )


def main() -> None:
    from search.search import search

    args = parser.parse_args()
    ctx = generate_context(args)

    if ctx.verbose:
        logger.info("Starting up...")
        logger.info(ctx)

    search(ctx)


if __name__ == "__main__":
    main()
