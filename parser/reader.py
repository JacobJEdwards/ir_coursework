import multiprocessing
import concurrent.futures
from typing import assert_never

import numpy as np

from main import Context
from pathlib import Path
import json
from config import VIDEOGAMES_DIR, VIDEOGAME_LABELS
from parser.parser import (
    merge_word_count_dicts,
    pickle_obj,
    depickle_obj,
    parse_contents,
    get_pickle_name,
    generate_document_matrix,
)
from parser.types import (
    InvertedIndex,
    Metadata,
    ParsedDir,
    ParsedFile,
    ParsedDirResults,
    ParsedDirSuccess,
)
from utils import timeit
import logging
import csv
import asyncio


logger = logging.getLogger(__name__)


def get_metadata(ctx) -> Metadata | None:
    if ctx.verbose:
        logger.info("Loading metadata")

    try:
        with open("metadata.json") as f:
            metadata = json.load(f)
            return metadata
    except json.JSONDecodeError:
        logger.error("Error decoding metadata")
    except IOError:
        logger.error("Error loading metadata")

    return None


def needs_reindexing(ctx: Context) -> bool:
    if ctx.reindex:
        return True

    pickle_name: Path = get_pickle_name(ctx.stripper)

    if not pickle_name.exists():
        return True

    try:
        last_pickled = get_pickle_name(ctx.stripper).stat().st_mtime

        for file in VIDEOGAMES_DIR.iterdir():
            if file.stat().st_mtime > last_pickled:
                logger.info("Reindexing")
                return True

        return False

    except IOError:
        logger.error("Error reading pickle file")
        return True


@timeit
def index_documents(
    ctx: Context,
) -> tuple[InvertedIndex, dict[str, np.ndarray], list[str]]:
    needs_regen = needs_reindexing(ctx)

    if not needs_regen:
        metadata = get_metadata(ctx)
        ii = depickle_obj(ctx)
        vec_space, doc_vectors = generate_document_matrix(ctx, ii, metadata)

        return ii, doc_vectors, vec_space

    results = parse_dir(ctx, VIDEOGAMES_DIR)

    success: ParsedDirSuccess = {}

    # filepath is of type Path, result of type DocOccurrences
    for file_path, parse_result in results.items():
        if not isinstance(parse_result, Exception):
            success[file_path] = parse_result
        else:
            logger.warning(f"File: {file_path}")
            logger.warning(parse_result)

    metadata = generate_metadata(ctx, success)

    result_dict: ParsedDirResults = {
        path: val["tokens"] for path, val in success.items()
    }

    ii = merge_word_count_dicts(ctx, result_dict, metadata)
    vec_space, doc_vecs = generate_document_matrix(ctx, ii, metadata)

    pickle_obj(ctx, ii)
    return ii, doc_vecs, vec_space


def get_average_wc(metadata: Metadata) -> float:
    return (
        0
        if len(metadata["files"]) == 0
        else sum([metadata["files"][file]["word_count"] for file in metadata["files"]])
        / len(metadata["files"])
    )


def generate_metadata(ctx: Context, info: ParsedDirSuccess) -> Metadata:
    if ctx.verbose:
        logger.info("Generating metadata")

    metadata = {
        "total_docs": len(info),
        "stripper": ctx.stripper,
        "files": {},
        "average_wc": 0,
    }

    try:
        with open(VIDEOGAME_LABELS, "r") as f:
            reader = csv.DictReader(f)

            metadata["files"].update(
                {
                    str(file_path): {
                        "last_modified": file_path.stat().st_mtime,
                        "file_size": file_path.stat().st_size,
                        "info": line,
                        "word_count": info[file_path]["word_count"],
                    }
                    for line in reader
                    for file_path in info.keys()
                    if str(line["url"]).replace("ps2.gamespy.com/", "")
                    == str(file_path)
                }
            )

            metadata["files"].update(
                {
                    str(file_path): {
                        "last_modified": file_path.stat().st_mtime,
                        "file_size": file_path.stat().st_size,
                        "info": None,
                        "word_count": info[file_path]["word_count"],
                    }
                    for file_path in info.keys()
                    if str(file_path) not in metadata["files"]
                }
            )

    except IOError:
        logger.error("Error reading labels")
    except csv.Error:
        logger.error("Error parsing csv")

    metadata["average_wc"] = get_average_wc(metadata)

    try:
        with open("metadata.json", "w") as f:
            json.dump(metadata, f)
    except Exception as e:
        logger.error("Error setting metadata")
        logger.error(e)

    return metadata


def _parse_file(ctx: Context, file_path: Path) -> ParsedFile:
    try:
        with open(file_path, "rb") as f:
            results = parse_contents(ctx, f)
        return results
    except Exception as e:
        # change the exception handling
        logger.error(f"Error reading/parsing {file_path}: {str(e)}")
        e.add_note(f"Error reading/parsing {file_path}: {str(e)}")
        return e


async def _parse_file_async(ctx: Context, file_path: Path) -> ParsedFile:
    try:
        with open(file_path, "rb") as f:
            results = parse_contents(ctx, f)
        return results
    except Exception as e:
        e.add_note(f"Error reading/parsing {file_path}: {str(e)}")
        return e


@timeit
def parse_dir(
    ctx: Context,
    directory: Path,
) -> ParsedDir:
    if not directory.is_dir():
        logger.critical("Cannot find files")
        exit(1)

    match ctx.parser_type:
        case "async":
            return asyncio.run(_parse_dir_async(ctx, directory))
        case "sync":
            return _parse_dir_sync(ctx, directory)
        case "mp":
            return _parse_dir_mp(ctx, directory)
        case "mt":
            return _parse_dir_mt(ctx, directory)
        case _:
            assert_never("Unreachable")


@timeit
def _parse_dir_sync(
    ctx: Context,
    directory: Path,
) -> ParsedDir:
    try:
        results = map(lambda file: (file, _parse_file(ctx, file)), directory.iterdir())

        return {file_path: result for file_path, result in results}
    except Exception as e:
        logger.critical(e)
        exit(1)


@timeit
def _parse_dir_mp(
    ctx: Context,
    directory: Path,
) -> ParsedDir:
    logger.debug("Beginning file parsing")
    try:
        with multiprocessing.Pool() as pool:
            # use a list of (file_path, result) tuples to store results
            results = []

            for file_path in directory.iterdir():
                pool.apply_async(
                    _parse_file,
                    (
                        ctx,
                        file_path,
                    ),
                    callback=lambda res, fp=file_path: results.append((fp, res)),
                )

            pool.close()
            pool.join()

            # collect results from the async tasks
            results_dict = {file_path: result for file_path, result in results}

        return results_dict
    except Exception as e:
        logger.critical(e)
        exit(1)


async def _parse_dir_async(
    ctx: Context,
    directory: Path,
) -> ParsedDir:
    try:
        async with asyncio.TaskGroup() as tg:
            results = []

            for file_path in directory.iterdir():
                result = await tg.create_task(_parse_file_async(ctx, file_path))
                results.append((file_path, result))

        results_dict = {file_path: result for file_path, result in results}

        return results_dict
    except Exception as e:
        logger.critical(e)
        exit(1)


@timeit
def _parse_dir_mt(ctx: Context, directory: Path) -> ParsedDir:
    results = {}

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Use a list of (future, file_path) tuples to associate results with file paths
        futures = {
            executor.submit(_parse_file, ctx, file_path): file_path
            for file_path in directory.iterdir()
        }

        for future in concurrent.futures.as_completed(futures):
            file_path = futures[future]
            result = future.result()
            results[file_path] = result

    return results
