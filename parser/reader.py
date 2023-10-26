import multiprocessing
from pathlib import Path
from parser.parser import parse_contents
from typing import Union, Dict, Tuple
import json
from config import VIDEOGAMES_DIR
from parser.parser import (
    merge_word_count_dicts,
    pickle_obj,
    depickle_obj,
    InvertedIndex,
)


# need to check if files have changed -> store metadata -> regen and pickle if
def generate_object() -> InvertedIndex:
    try:
        with open("metadata.json") as f:
            metadata = json.load(f)
    except Exception as e:
        metadata = {}

    needs_regen = False
    try:
        last_pickled = Path("data.pkl").stat().st_mtime

        for file in VIDEOGAMES_DIR.iterdir():
            if file.stat().st_mtime > last_pickled:
                needs_regen = True
                break
    except Exception as e:
        needs_regen = True

    if not needs_regen:
        return depickle_obj()

    results = read_dir(VIDEOGAMES_DIR)
    success = {}
    metadata = {
        "total_docs": len(results),
    }

    for file_path, (result, count) in results.items():
        if not isinstance(result, Exception):
            success[str(file_path)] = result
            last_accessed = file_path.stat().st_mtime
            metadata[str(file_path)] = {
                "word_count": count,
                "last_modified": last_accessed,
            }

        else:
            print(f"File: {file_path}")
            print(result)

    with open("metadata.json", "w") as f:
        json.dump(metadata, f)

    t = merge_word_count_dicts(success)
    pickle_obj(t)
    return t


def parse_and_read(file_path: Path) -> Tuple[Dict[str, int], int] | Exception:
    try:
        with open(file_path, "rb") as f:
            results = parse_contents(f)
        return results
    except Exception as e:
        # change the exception handling
        e.add_note(f"Error reading/parsing {file_path}: {str(e)}")
        return e


def read_dir(directory: Path) -> Dict[Path, Union[Dict[str, int], str]]:
    try:
        with multiprocessing.Pool() as pool:
            # use a list of (file_path, result) tuples to store results
            results = []

            for file_path in directory.iterdir():
                result = pool.apply_async(parse_and_read, (file_path,))
                results.append((file_path, result))

            pool.close()
            pool.join()

            # collect results from the async tasks
            results_dict = {file_path: result.get() for file_path, result in results}

        return results_dict
    except Exception as e:
        print(e)
        exit(1)
