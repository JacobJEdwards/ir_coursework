from parser.parser import merge_word_count_dicts, pickle_obj
from parser.reader import read_dir
from search.search import search
import json

from config import VIDEOGAMES_DIR


# need to check if files have changed -> store metadata -> regen and pickle if
def generate_object() -> None:
    results = read_dir(VIDEOGAMES_DIR)
    success = {}
    metadata = {
        "total_docs": len(results),
    }

    for file_path, (result, count) in results.items():
        if not isinstance(result, Exception):
            success[str(file_path)] = result
            metadata[str(file_path)] = {
                "word_count": count
            }
        else:
            print(f"File: {file_path}")
            print(result)

    with open("metadata.json", "w") as f:
        json.dump(metadata, f)

    t = merge_word_count_dicts(success)
    pickle_obj(t)
    # t = depickle_obj()
    # pprint(t)


def main() -> None:
    generate_object()
    search()


if __name__ == "__main__":
    main()
