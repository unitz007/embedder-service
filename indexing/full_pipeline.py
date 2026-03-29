# full_pipeline.py
"""
Complete pipeline implementing:
Repository -> File Discovery -> Language Detection -> AST Parsing ->
Symbol Extraction -> Import Graph -> Call Graph -> Code Chunking ->
Embedding Generation -> Chroma Vector DB -> ContextBuilder (LLM-ready)
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional

from lib.chunker import chunk_repository_analyses
from lib.embedder import (
    embed_repository_chunks,
    CodeEmbedder,
    VoyageEmbedder,
    LARGE_CODEBASE_THRESHOLD,
    EMBEDDING_DIM,
    VOYAGE_EMBEDDING_DIM,
    VOYAGE_MODEL,
    DEFAULT_MODEL,
)
from lib.graph import (
    build_import_graph, enrich_chunks_with_graph,
    build_call_graph, enrich_chunks_with_call_graph,
)
from utils.file_scanner import scan_repository
from utils.language_router import analyze_file
from store.pgvector_store import PgVectorStore


def index_repository(
    repo_path: str,
    include_dotfiles: bool = False,
) -> List[Any]:
    """
    Steps 1-5: Repository -> File Discovery -> Language Detection -> AST Parsing -> Symbol Extraction

    Args:
        repo_path:        Root directory to scan.
        include_dotfiles: Pass True when indexing dotfile repos (e.g. ~/.dotfiles)
                          so hidden files like .zshrc, .gitconfig are included.

    Returns:
        List of FileAnalysis objects.
    """
    print("Step 1-5: Scanning repository and extracting symbols...")
    files = scan_repository(repo_path, include_dotfiles=include_dotfiles)

    results = []
    for file in files:
        analysis = analyze_file(file)
        if analysis:
            results.append(analysis)

    # skipped = len(files) - len(results)

    return results




def chunk_analyses(analyses: List[Any]) -> List[Dict[str, Any]]:
    """
    Step 6: Code Chunking
    Returns: List of chunk dictionaries
    """
    print("Step 6: Chunking code...")
    chunks = chunk_repository_analyses(analyses)
    print(f"Created {len(chunks)} code chunks")
    return chunks


def generate_embeddings(
    chunks: List[Dict[str, Any]],
    model_name: str = "microsoft/codebert-base",
    embedder=None,
) -> List[Dict[str, Any]]:
    """
    Step 7: Embedding Generation

    If ``embedder`` is provided it is used directly; otherwise a local
    CodeEmbedder is instantiated from ``model_name``.

    Returns: List of chunks with embeddings added
    """
    if embedder is not None:
        return embedder.embed_chunks(chunks)
    return embed_repository_chunks(chunks, model_name)


def search_code(
    vector_store,
    query: str,
    embedder_model_name: str = "microsoft/codebert-base",
    k: int = 5,
    embedder: Optional[CodeEmbedder] = None,
) -> List[Dict[str, Any]]:
    """
    Search the vector store for code similar to the query.

    Pass a pre-loaded `embedder` to avoid reloading the model on every call.

    Args:
        vector_store: The vector store to search (ChromaStore)
        query: Natural language query about code
        embedder_model_name: Model to use if embedder is not provided
        k: Number of results to return
        embedder: Optional pre-loaded CodeEmbedder instance

    Returns:
        List of matching chunks with metadata and similarity scores
    """
    print(f"Searching for: '{query}'")

    _embedder = embedder if embedder is not None else CodeEmbedder(embedder_model_name)
    query_embedding = _embedder.encode([query])[0].tolist()
    # ChromaStore.search returns (metadata, cosine_score, internal_idx)
    results = vector_store.search(query_embedding, k=k)

    return [
        {
            "metadata": meta,
            "similarity_score": max(0.0, score),
            "rank": i + 1,
        }
        for i, (meta, score, _) in enumerate(results)
    ]


# ---------------------------------------------------------------------------
# Chroma pipeline
# ---------------------------------------------------------------------------

def full_pipeline_chroma(
    repo_path: str,
    chroma_dir: str,
    graphs_prefix: str = None,
    model_name: str = "microsoft/codebert-base",
    include_dotfiles: bool = False,
) -> tuple:
    """
    Run the complete pipeline and store vectors in Chroma instead of FAISS.

    On each call the Chroma collection is cleared first so no stale vectors
    from deleted or renamed files survive across runs.

    Args:
        repo_path:        Path to the repository to index.
        chroma_dir:       Directory where Chroma will persist its data.
        graphs_prefix:    Path prefix for call/import graph .pkl files.
                          Defaults to <chroma_dir>/../vector_store so graphs
                          land next to the Chroma dir.
        model_name:       Embedding model (sentence-transformers).
        include_dotfiles: Include hidden files.

    Returns:
        Tuple of (ChromaStore, call_graph dict, import_graph dict)
    """
    from store.chroma_store import ChromaStore

    if graphs_prefix is None:
        graphs_prefix = str(Path(chroma_dir).parent / "vector_store")

    print(f"Starting Chroma pipeline for repository: {repo_path}")
    print("=" * 50)

    # Steps 1-5: Symbol extraction
    analyses = index_repository(repo_path, include_dotfiles=include_dotfiles)

    # Step 5b: Import graph
    print("Step 5b: Building import relationship graph...")
    import_graph = build_import_graph(analyses)

    # Step 5c: Call graph
    print("Step 5c: Building call graph...")
    call_graph = build_call_graph(analyses)

    # Step 6: Chunking + graph enrichment
    chunks = chunk_analyses(analyses)
    chunks = enrich_chunks_with_graph(chunks, import_graph)
    chunks = enrich_chunks_with_call_graph(chunks, call_graph)

    # Step 7: Embeddings — pick cloud or local based on codebase size
    if len(chunks) > LARGE_CODEBASE_THRESHOLD:
        print(
            f"Step 7: {len(chunks)} chunks exceeds threshold "
            f"({LARGE_CODEBASE_THRESHOLD}). Using Voyage AI cloud embedder..."
        )
        embedder = VoyageEmbedder()
    else:
        print(
            f"Step 7: {len(chunks)} chunks. Using local CodeBERT embedder..."
        )
        embedder = CodeEmbedder(model_name)

    chunks_with_embeddings = generate_embeddings(chunks, embedder=embedder)

    # Step 8: Store in Chroma.
    # We use add_vectors(replace_all=True) instead of clear() + add_vectors().
    # clear() calls delete_collection() which triggers SQLite VACUUM INTO,
    # renaming the database file and causing SQLITE_READONLY_DBMOVED
    # (ChromaDB InternalError 1032) on every subsequent write — including
    # the write that triggered it.  replace_all upserts new chunks in-place
    # and removes only stale IDs, touching nothing at the file level.
    print("Step 8: Storing in Chroma vector database...")
    store = ChromaStore(chroma_dir)

    if chunks_with_embeddings:
        embeddings = [c["embedding"] for c in chunks_with_embeddings]
        metadata  = [c["metadata"]  for c in chunks_with_embeddings]
        store.add_vectors(embeddings, metadata, replace_all=True)
    else:
        store.add_vectors([], [], replace_all=True)

    # Persist graphs alongside the Chroma dir (same prefix as FAISS path so
    # ContextBuilder.load_from_chroma() and load_from_disk() share them).
    Path(graphs_prefix).parent.mkdir(parents=True, exist_ok=True)
    with open(f"{graphs_prefix}_call_graph.pkl", "wb") as f:
        pickle.dump(call_graph, f)
    with open(f"{graphs_prefix}_import_graph.pkl", "wb") as f:
        pickle.dump(import_graph, f)
    print(f"Graphs saved → {graphs_prefix}_call_graph.pkl / ..._import_graph.pkl")

    # Write index metadata so callers know which embedding model was used.
    # This prevents dimension mismatches at query time when the codebase size
    # crosses the local/cloud threshold between runs.
    is_cloud = isinstance(embedder, VoyageEmbedder)
    index_meta = {
        "embedder": type(embedder).__name__,
        "model": VOYAGE_MODEL if is_cloud else model_name,
        "embedding_dim": VOYAGE_EMBEDDING_DIM if is_cloud else EMBEDDING_DIM,
        "use_cloud": is_cloud,
    }
    meta_path = Path(chroma_dir) / "index_meta.json"
    with open(meta_path, "w") as f:
        json.dump(index_meta, f, indent=2)
    print(f"Index metadata saved → {meta_path}")

    print("=" * 50)
    print("Chroma pipeline completed!")
    return store, call_graph, import_graph, type(embedder).__name__


# ---------------------------------------------------------------------------
# pgvector pipeline
# ---------------------------------------------------------------------------

def full_pipeline_pgvector(
    repo_path: str,
    namespace: str,
    project_id: str,
    model_name: str = "microsoft/codebert-base",
    include_dotfiles: bool = False,
) -> tuple:
    """
    Run the complete pipeline and store vectors in Postgres/pgvector.

    Args:
        repo_path:        Path to the repository to index.
        namespace:        Namespace for multi-tenant storage.
        project_id:       Project identifier.
        model_name:       Embedding model (sentence-transformers).
        include_dotfiles: Include hidden files.

    Returns:
        Tuple of (PgVectorStore, call_graph dict, import_graph dict, index_meta dict)
    """
    print(f"Starting pgvector pipeline for repository: {repo_path}")
    print("=" * 50)

    # Steps 1-5: Symbol extraction
    analyses = index_repository(repo_path, include_dotfiles=include_dotfiles)

    # Step 5b: Import graph
    print("Step 5b: Building import relationship graph...")
    import_graph = build_import_graph(analyses)

    # Step 5c: Call graph
    print("Step 5c: Building call graph...")
    call_graph = build_call_graph(analyses)

    # Step 6: Chunking + graph enrichment
    chunks = chunk_analyses(analyses)
    chunks = enrich_chunks_with_graph(chunks, import_graph)
    chunks = enrich_chunks_with_call_graph(chunks, call_graph)

    # Step 7: Embeddings — pick cloud or local based on codebase size
    if len(chunks) > LARGE_CODEBASE_THRESHOLD:
        print(
            f"Step 7: {len(chunks)} chunks exceeds threshold "
            f"({LARGE_CODEBASE_THRESHOLD}). Using Voyage AI cloud embedder..."
        )
        embedder = VoyageEmbedder()
    else:
        print(
            f"Step 7: {len(chunks)} chunks. Using local CodeBERT embedder..."
        )
        embedder = CodeEmbedder(model_name)

    chunks_with_embeddings = generate_embeddings(chunks, embedder=embedder)

    # Step 8: Store in Postgres/pgvector (replace_all to avoid stale vectors)
    print("Step 8: Storing in pgvector...")
    store = PgVectorStore(namespace=namespace, project_id=project_id)
    if chunks_with_embeddings:
        embeddings = [c["embedding"] for c in chunks_with_embeddings]
        metadata = [c["metadata"] for c in chunks_with_embeddings]
        store.add_vectors(embeddings, metadata, replace_all=True)
    else:
        store.add_vectors([], [], replace_all=True)

    # Build index metadata so callers know which embedder was used
    is_cloud = isinstance(embedder, VoyageEmbedder)
    index_meta = {
        "embedder": type(embedder).__name__,
        "model": VOYAGE_MODEL if is_cloud else model_name,
        "embedding_dim": VOYAGE_EMBEDDING_DIM if is_cloud else EMBEDDING_DIM,
        "use_cloud": is_cloud,
    }

    print("=" * 50)
    print("pgvector pipeline completed!")
    return store, call_graph, import_graph, index_meta
