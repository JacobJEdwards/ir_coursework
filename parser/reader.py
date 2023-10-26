import concurrent.futures
import multiprocessing
from pathlib import Path
from parser.parser import parse_contents
from typing import Union, Dict


def parse_and_read(file_path: Path) -> Union[Dict[str, int], str]:
    try:
        with open(file_path, "rb") as f:
            results = parse_contents(f)
        return results
    except Exception as e:
        return f"Error reading/parsing {file_path}: {str(e)}"


def read_dir(directory: Path) -> Dict[Path, Union[Dict[str, int], str]]:
    pool = multiprocessing.Pool()

    # Use a list of (file_path, result) tuples to store results
    results = []

    for file_path in directory.iterdir():
        result = pool.apply_async(parse_and_read, (file_path,))
        results.append((file_path, result))

    pool.close()
    pool.join()

    # Collect results from the async tasks
    results_dict = {file_path: result.get() for file_path, result in results}

    #with concurrent.futures.ThreadPoolExecutor() as executor:
        # Use a list of (future, file_path) tuples to associate results with file paths
     #   futures = {
      #      executor.submit(parse_and_read, file_path): file_path
       #     for file_path in directory.iterdir()
        #}

        #for future in concurrent.futures.as_completed(futures):
         #   file_path = futures[future]
          #  result = future.result()
           # results[file_path] = result

    return results_dict
