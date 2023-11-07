import multiprocessing
from typing import assert_never
from main import Context
from pathlib import Path
import json
from config import VIDEOGAMES_DIR, PICKLE_FILE, VIDEOGAME_LABELS
from parser.parser import (
    merge_word_count_dicts,
    pickle_obj,
    depickle_obj,
    parse_contents,
)
from parser.types import InvertedIndex, DocumentMatrix, Metadata, DocOccurrences
import logging
from functools import wraps
from time import time
import csv
import asyncio


def timeit(func):
    @wraps(func)
    def wrap(*args, **kw):
        if not args[0].verbose:
            return func(*args, **kw)

        ts = time()
        result = func(*args, **kw)
        te = time()
        print(
            "func:%r args:[%r, %r] took: %2.4f sec" % (func.__name__, args, kw, te - ts)
        )
        return result

    return wrap


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

    try:
        last_pickled = PICKLE_FILE.stat().st_mtime

        for file in VIDEOGAMES_DIR.iterdir():
            if file.stat().st_mtime > last_pickled:
                logger.info("Reindexing")
                return True

        return False

    except IOError:
        logger.error("Error reading pickle file")
        return False


# need to check if files have changed -> store metadata -> regen and pickle if
@timeit
def index_documents(ctx: Context) -> tuple[InvertedIndex, DocumentMatrix | None]:
    needs_regen = needs_reindexing(ctx)

    if not needs_regen:
        return depickle_obj(ctx), None

    results = parse_dir(ctx, VIDEOGAMES_DIR)

    success = {}

    # filepath is of type Path, result of type DocOccurrences
    for file_path, result in results.items():
        if not isinstance(result, Exception):
            success[file_path] = result

        else:
            logger.warning(f"File: {file_path}")
            logger.warning(result)

    metadata = generate_metadata(ctx, list(success.keys()))

    ii, doc_matrix = merge_word_count_dicts(ctx, success, metadata)
    pickle_obj(ctx, ii)
    return ii, doc_matrix


def generate_metadata(ctx: Context, files: list[Path]) -> Metadata:
    if ctx.verbose:
        logger.info("Generating metadata")

    metadata: Metadata = {
        "total_docs": len(files),
        "stripper": ctx.stripper,
        "files": {},
    }

    try:
        with open(VIDEOGAME_LABELS, "r") as f:
            reader = csv.DictReader(f)

            metadata["files"].update(
                {
                    str(file_path): {
                        "last_modified": file_path.stat().st_mtime,
                        "info": line,
                    }
                    for line in reader
                    for file_path in files
                    if str(line["url"]).replace("ps2.gamespy.com/", "")
                    == str(file_path)
                }
            )

    except IOError:
        logger.error("Error reading labels")
    except csv.Error:
        logger.error("Error parsing csv")

    try:
        with open("metadata.json", "w") as f:
            json.dump(metadata, f)
    except Exception as e:
        logger.error("Error setting metadata")
        logger.error(e)

    return metadata


def parse_file(ctx: Context, file_path: Path) -> list[DocOccurrences] | Exception:
    try:
        with open(file_path, "rb") as f:
            results = parse_contents(ctx, f)
        return results
    except Exception as e:
        # change the exception handling
        e.add_note(f"Error reading/parsing {file_path}: {str(e)}")
        return e


@timeit
def parse_dir(
    ctx: Context,
    directory: Path,
) -> dict[Path, list[DocOccurrences] | Exception]:
    match ctx.parser_type:
        case "async":
            return asyncio.run(_parse_dir_async(ctx, directory))
        case "sync":
            return _parse_dir_sync(ctx, directory)
        case "mp":
            return _parse_dir_mp(ctx, directory)
        case _:
            assert_never("Unreachable")


@timeit
def _parse_dir_sync(
    ctx: Context,
    directory: Path,
) -> dict[Path, list[DocOccurrences] | Exception]:
    try:
        results = map(lambda file: (file, parse_file(ctx, file)), directory.iterdir())

        return {file_path: result for file_path, result in results}
    except Exception as e:
        logger.critical(e)
        exit(1)


@timeit
def _parse_dir_mp(
    ctx: Context,
    directory: Path,
) -> dict[Path, list[DocOccurrences] | Exception]:
    logger.debug("Beginning file parsing")
    try:
        with multiprocessing.Pool() as pool:
            # use a list of (file_path, result) tuples to store results
            results = []

            for file_path in directory.iterdir():
                pool.apply_async(
                    parse_file,
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


async def _parse_file_async(
    ctx: Context, file_path: Path
) -> list[DocOccurrences] | Exception:
    try:
        with open(file_path, "rb") as f:
            results = parse_contents(ctx, f)
        return results
    except Exception as e:
        # change the exception handling
        e.add_note(f"Error reading/parsing {file_path}: {str(e)}")
        return e


@timeit
async def _parse_dir_async(
    ctx: Context,
    directory: Path,
) -> dict[Path, list[DocOccurrences] | Exception]:
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
