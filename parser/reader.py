from pathlib import Path
from parser.parser import parse_contents
from typing import Union, Dict, List, Tuple
from parser.parser import DocOccurrences
import json
from config import VIDEOGAMES_DIR, PICKLE_FILE
from parser.parser import (
    merge_word_count_dicts,
    pickle_obj,
    depickle_obj,
    InvertedIndex,
    DocumentMatrix,
)
import logging
import asyncio
from pprint import pprint


logger = logging.getLogger(__name__)


# need to check if files have changed -> store metadata -> regen and pickle if
def generate_object() -> Tuple[InvertedIndex, Union[DocumentMatrix, None]]:
    try:
        with open("metadata.json") as f:
            metadata = json.load(f)
    except Exception:
        logger.error("Error loading metadata")
        metadata = {}

    needs_regen = False
    try:
        last_pickled = PICKLE_FILE.stat().st_mtime

        for file in VIDEOGAMES_DIR.iterdir():
            if file.stat().st_mtime > last_pickled:
                logger.info("Reindexing")
                needs_regen = True
                break
    except Exception:
        logger.error("Error reading pickle file")
        needs_regen = True

    if not needs_regen:
        return depickle_obj(), None

    results = asyncio.run(read_dir(VIDEOGAMES_DIR))
    total_docs = len(results)
    success = {}
    metadata = {
        "total_docs": total_docs,
    }

    # filepath is of type Path, result of type DocOccurrences
    for file_path, result in results.items():
        if not isinstance(result, Exception):
            success[str(file_path)] = result
            last_accessed = file_path.stat().st_mtime
            metadata[str(file_path)] = {
                "last_modified": last_accessed,
            }

        else:
            logger.warning(f"File: {file_path}")
            logger.warning(result)

    try:
        with open("metadata.json", "w") as f:
            json.dump(metadata, f)
    except Exception as e:
        logger.error("Error setting metadata")
        logger.error(e)

    ii, doc_matrix = merge_word_count_dicts(success, total_docs)
    pickle_obj(ii)
    return ii, doc_matrix


async def parse_and_read(file_path: Path) -> Union[List[DocOccurrences], Exception]:
    try:
        with open(file_path, "rb") as f:
            results = parse_contents(f)
        return results
    except Exception as e:
        # change the exception handling
        e.add_note(f"Error reading/parsing {file_path}: {str(e)}")
        return e


# SWITCH TO ASYNCIO
async def read_dir(
    directory: Path,
) -> Dict[Path, Union[List[DocOccurrences], Exception]]:
    logger.debug("Beginning file parsing")
    try:
        async with asyncio.TaskGroup() as tg:
            results = []

            for file_path in directory.iterdir():
                result = await tg.create_task(parse_and_read(file_path))
                results.append((file_path, result))

        results_dict = {file_path: result for file_path, result in results}

        return results_dict
    except Exception as e:
        logger.critical(e)
        exit(1)
