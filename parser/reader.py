import multiprocessing
from pathlib import Path
from parser.parser import parse_contents
from typing import Union, Dict, Tuple


def parse_and_read(file_path: Path) -> Tuple[Dict[str, int], int] | Exception:
    try:
        with open(file_path, "rb") as f:
            results = parse_contents(f)
        return results
    except Exception as e:
        # change the exception handling
        e.add_note(f"Error reading/parsing {file_path}: {str(e)}")
        return e
        #  return f"Error reading/parsing {file_path}: {str(e)}"


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

