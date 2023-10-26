from parser.parser import depickle_obj, merge_word_count_dicts, pickle_obj
from parser.reader import read_dir
from search.search import search
from pprint import pprint

from config import VIDEOGAMES_DIR


# need to check if files have changed -> store metadata -> regen and pickle if
def generate_object() -> None:
    results = read_dir(VIDEOGAMES_DIR)
    success = {}

    for file_path, result in results.items():
        if not isinstance(result, str):
            success[str(file_path)] = result
        else:
            print(f"File: {file_path}")
            print(result)

    t = merge_word_count_dicts(success)
    pickle_obj(t)
    # t = depickle_obj()
    # pprint(t)


def main() -> None:
    generate_object()
    search()


if __name__ == "__main__":
    main()
