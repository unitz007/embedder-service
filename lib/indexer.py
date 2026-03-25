# indexer.py

from utils.file_scanner import scan_repository
from utils.language_router import analyze_file

def index_repository(repo_path):

    files = scan_repository(repo_path)

    results = []

    for file in files:
        analysis = analyze_file(file)
        if analysis:
            results.append(analysis)

    return results