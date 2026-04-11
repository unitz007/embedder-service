# chunker.py
from typing import List, Dict, Any
from models import FileAnalysis
import os


def _read_file_lines(file_path: str) -> List[str]:
    """Read file lines once; shared across all extraction calls for the same file."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.readlines()
    except Exception:
        return []


def extract_imports_from_lines(lines: List[str], language: str) -> List[str]:
    """Extract import statements from pre-read lines."""
    imports = []
    for line in lines:
        stripped = line.strip()
        if language == "python":
            if stripped.startswith("import ") or stripped.startswith("from "):
                imports.append(stripped)
        elif language == "go":
            if stripped.startswith("import ") or (stripped.startswith("(") and "import" in stripped):
                imports.append(stripped)
        elif language in ["javascript", "typescript"]:
            if (
                stripped.startswith("import ")
                or stripped.startswith("require(")
                or ("from " in stripped and ("import" in stripped or "require" in stripped))
            ):
                imports.append(stripped)
    return imports


def extract_function_body(lines: List[str], start_line: int, language: str, context_lines: int = 10) -> str:
    """Extract function body from pre-read lines."""
    if not lines:
        return f"# Function at line {start_line}\n# [Could not extract function body]"

    start_idx = max(0, start_line - 1)
    start_context = max(0, start_idx - context_lines)
    end_idx = min(len(lines), start_idx + context_lines + 20)

    if language == "python":
        base_indent = (
            len(lines[start_idx]) - len(lines[start_idx].lstrip())
            if start_idx < len(lines) else 0
        )
        for i in range(start_idx + 1, min(len(lines), start_idx + 200)):
            line = lines[i]
            if line.strip() == "":
                continue
            indent = len(line) - len(line.lstrip())
            if indent <= base_indent and line.strip():
                end_idx = i
                break
        else:
            end_idx = min(len(lines), start_idx + 200)
    elif language in ["go", "javascript", "typescript"]:
        brace_count = 0
        found_first_brace = False
        for i in range(start_idx, min(len(lines), start_idx + 300)):
            for char in lines[i]:
                if char == "{":
                    brace_count += 1
                    found_first_brace = True
                elif char == "}":
                    brace_count -= 1
                    if found_first_brace and brace_count == 0:
                        end_idx = i + 1
                        break
            if found_first_brace and brace_count == 0:
                break

    return "".join(lines[start_context:end_idx]).rstrip() + "\n"


def extract_class_body(lines: List[str], start_line: int, language: str, context_lines: int = 10) -> str:
    """Extract class body from pre-read lines."""
    if not lines:
        return f"# Class at line {start_line}\n# [Could not extract class body]"

    start_idx = max(0, start_line - 1)
    start_context = max(0, start_idx - context_lines)
    end_idx = min(len(lines), start_idx + context_lines + 30)

    if language == "python":
        base_indent = (
            len(lines[start_idx]) - len(lines[start_idx].lstrip())
            if start_idx < len(lines) else 0
        )
        for i in range(start_idx + 1, min(len(lines), start_idx + 300)):
            line = lines[i]
            if line.strip() == "":
                continue
            indent = len(line) - len(line.lstrip())
            if indent <= base_indent and line.strip():
                end_idx = i
                break
        else:
            end_idx = min(len(lines), start_idx + 300)
    elif language in ["go", "javascript", "typescript"]:
        brace_count = 0
        found_first_brace = False
        for i in range(start_idx, min(len(lines), start_idx + 400)):
            for char in lines[i]:
                if char == "{":
                    brace_count += 1
                    found_first_brace = True
                elif char == "}":
                    brace_count -= 1
                    if found_first_brace and brace_count == 0:
                        end_idx = i + 1
                        break
            if found_first_brace and brace_count == 0:
                break

    return "".join(lines[start_context:end_idx]).rstrip() + "\n"


def chunk_file_analysis(analysis: FileAnalysis) -> List[Dict[str, Any]]:
    """
    Break a FileAnalysis into chunks based on functions, classes, and imports.

    Reads file content exactly once. Stores:
      - content: source text used for embedding
      - metadata: structured fields including signature, docstring, params,
                  return_type used for LLM context injection and BM25 scoring
    """
    chunks = []

    # Read file content ONCE
    file_lines = _read_file_lines(analysis.file_path)
    file_content = "".join(file_lines)

    # --- Imports chunk ---
    if analysis.imports:
        actual_imports = extract_imports_from_lines(file_lines, analysis.language)
        imports_content = "\n".join(actual_imports) if actual_imports else "\n".join(analysis.imports)
        chunks.append({
            "content": imports_content,
            "type": "imports",
            "metadata": {
                "file_path": analysis.file_path,
                "language": analysis.language,
                "chunk_type": "imports",
                "import_count": len(actual_imports) if actual_imports else len(analysis.imports),
                "imports": actual_imports if actual_imports else analysis.imports,
                "package": analysis.package,
                "content": imports_content,
            },
        })

    # --- Function chunks ---
    for func in analysis.functions:
        func_body = extract_function_body(file_lines, func.line, analysis.language)
        chunks.append({
            "content": func_body,
            "type": "function",
            "metadata": {
                "file_path": analysis.file_path,
                "language": analysis.language,
                "function_name": func.name,
                "line_number": func.line,
                "chunk_type": "function",
                "symbol_type": "function",
                "symbol_name": func.name,
                "signature": func.signature,
                "docstring": func.docstring,
                "params": func.params,
                "return_type": func.return_type,
                "package": analysis.package,
                "content": func_body,
            },
        })

    # --- Class / struct / interface chunks ---
    for cls in analysis.classes:
        class_body = extract_class_body(file_lines, cls.line, analysis.language)
        chunks.append({
            "content": class_body,
            "type": "class",
            "metadata": {
                "file_path": analysis.file_path,
                "language": analysis.language,
                "class_name": cls.name,
                "line_number": cls.line,
                "chunk_type": "class",
                "symbol_type": cls.kind or "class",
                "symbol_name": cls.name,
                "docstring": cls.docstring,
                "package": analysis.package,
                "fields": [{"name": f.name, "type_str": f.type_str, "tag": f.tag} for f in cls.fields],
                "content": class_body,
            },
        })

    # --- File-level summary chunk ---
    # Always add a compact file-level chunk so the LLM can find a file by its
    # overall responsibility even when individual symbol searches miss.
    summary_parts = [f"File: {analysis.file_path}"]
    if analysis.package:
        summary_parts.append(f"Package: {analysis.package}")
    if analysis.functions:
        summary_parts.append("Functions: " + ", ".join(f.name for f in analysis.functions[:20]))
    if analysis.classes:
        summary_parts.append("Types: " + ", ".join(c.name for c in analysis.classes[:20]))
    if analysis.imports:
        summary_parts.append("Imports: " + ", ".join(analysis.imports[:15]))
    summary_content = "\n".join(summary_parts)

    chunks.append({
        "content": summary_content,
        "type": "file_summary",
        "metadata": {
            "file_path": analysis.file_path,
            "language": analysis.language,
            "chunk_type": "file_summary",
            "symbol_type": "file",
            "symbol_name": os.path.basename(analysis.file_path),
            "package": analysis.package,
            "function_count": len(analysis.functions),
            "class_count": len(analysis.classes),
            "content": summary_content,
        },
    })

    # --- Full-file fallback (no symbols found) ---
    if len(chunks) == 1:  # only summary was added
        if not file_content.strip():
            file_content = f"# File: {analysis.file_path}\n# Language: {analysis.language}\n# [Empty file]"
        chunks.append({
            "content": file_content,
            "type": "file",
            "metadata": {
                "file_path": analysis.file_path,
                "language": analysis.language,
                "chunk_type": "file",
                "symbol_type": "file",
                "symbol_name": os.path.basename(analysis.file_path),
                "package": analysis.package,
                "content": file_content,
            },
        })

    return chunks


def chunk_repository_analyses(analyses: List[FileAnalysis]) -> List[Dict[str, Any]]:
    """Chunk all FileAnalysis objects from a repository."""
    all_chunks = []
    for analysis in analyses:
        all_chunks.extend(chunk_file_analysis(analysis))
    return all_chunks
