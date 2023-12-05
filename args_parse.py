from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


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
    choices=["sync", "async", "mp", "mt"],
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

parser.add_argument(
    "--scorer",
    "-scr",
    help="Choose whether to use tfidf, bm25, bm25+",
    type=str,
    choices=["tfidf", "bm25", "bm25+"],
    default="bm25+",
    dest="scorer",
)

parser.add_argument(
    "--searcher",
    "-se",
    help="Choose whether to use a normal scoring algorithm or a vector space",
    type=str,
    choices=["vector", "score"],
    default="vector",
    dest="searcher",
)

parser.add_argument(
    "--stopwords",
    action="store_true",
    dest="stopwords",
    help="disable removal of stopwords",
)
