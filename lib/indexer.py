# indexer.py

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

from utils.file_scanner import scan_repository
from utils.language_router import analyze_file

logger = logging.getLogger(__name__)


def index_repository(repo_path: str, max_concurrency: int = 1) -> List:
    """Analyse files in *repo_path* and return a list of FileAnalysis objects.

    Args:
        repo_path:        Root directory to scan.
        max_concurrency:  Number of files to analyse in parallel.
                          ``1`` (default) processes files sequentially.
                          Errors in individual files are logged and skipped.

    Returns:
        List of FileAnalysis objects for successfully analysed files.
    """
    files = scan_repository(repo_path)

    if max_concurrency <= 1:
        return _analyze_sequential(files)
    return _analyze_parallel(files, max_concurrency)


def _analyze_sequential(files: List[str]) -> List:
    results = []
    for file in files:
        analysis = analyze_file(file)
        if analysis:
            results.append(analysis)
    return results


def _analyze_parallel(files: List[str], max_workers: int) -> List:
    results = []
    failures = []

    def _try_analyze(path: str) -> Optional[object]:
        try:
            return analyze_file(path)
        except Exception as exc:
            logger.error("Failed to analyse %s: %s", path, exc)
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_path = {pool.submit(_try_analyze, f): f for f in files}
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
                else:
                    failures.append(path)
            except Exception as exc:
                logger.error("Failed to analyse %s: %s", path, exc)
                failures.append(path)

    if failures:
        logger.warning(
            "File analysis: %d failures out of %d files",
            len(failures), len(files),
        )

    return results


def index_repository_with_failures(
    repo_path: str, max_concurrency: int = 1,
) -> Tuple[List, List]:
    """Like :func:`index_repository` but also returns a list of failed paths."""
    files = scan_repository(repo_path)

    if max_concurrency <= 1:
        results = []
        failures = []
        for f in files:
            try:
                analysis = analyze_file(f)
                if analysis:
                    results.append(analysis)
                else:
                    failures.append(f)
            except Exception as exc:
                logger.error("Failed to analyse %s: %s", f, exc)
                failures.append(f)
        return results, failures

    # Parallel path
    results = []
    failures = []

    def _try_analyze(path: str):
        try:
            return analyze_file(path)
        except Exception as exc:
            logger.error("Failed to analyse %s: %s", path, exc)
            return exc

    with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        future_to_path = {pool.submit(_try_analyze, f): f for f in files}
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                result = future.result()
                if isinstance(result, Exception):
                    failures.append(path)
                elif result is not None:
                    results.append(result)
                else:
                    failures.append(path)
            except Exception as exc:
                logger.error("Failed to analyse %s: %s", path, exc)
                failures.append(path)

    return results, failures