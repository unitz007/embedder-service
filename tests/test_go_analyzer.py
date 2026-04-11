import os
import sys
import unittest

# Ensure project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analyzers.go_analyzer import analyze_go
from lib.chunker import chunk_file_analysis

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures", "go")


class TestGoAnalyzerBasic(unittest.TestCase):
    """Tests using fixtures/go/basic.go"""

    def setUp(self):
        self.analysis = analyze_go(os.path.join(FIXTURES_DIR, "basic.go"))

    def test_package_name(self):
        self.assertEqual(self.analysis.package, "basic")

    def test_import_count(self):
        self.assertEqual(len(self.analysis.imports), 3)
        self.assertIn("fmt", self.analysis.imports)
        self.assertIn("math", self.analysis.imports)
        self.assertIn("time", self.analysis.imports)

    def test_function_count(self):
        self.assertEqual(len(self.analysis.functions), 3)

    def test_function_names(self):
        names = [f.name for f in self.analysis.functions]
        self.assertIn("add", names)
        self.assertIn("greet", names)
        self.assertIn("distance", names)

    def test_class_count(self):
        # User, Config, Reader
        self.assertEqual(len(self.analysis.classes), 3)

    def test_struct_kinds(self):
        kinds = {c.name: c.kind for c in self.analysis.classes}
        self.assertEqual(kinds["User"], "struct")
        self.assertEqual(kinds["Config"], "struct")
        self.assertEqual(kinds["Reader"], "interface")


class TestGoAnalyzerEmpty(unittest.TestCase):
    """Tests using fixtures/go/empty.go"""

    def setUp(self):
        self.analysis = analyze_go(os.path.join(FIXTURES_DIR, "empty.go"))

    def test_package_name(self):
        self.assertEqual(self.analysis.package, "empty")

    def test_no_symbols(self):
        self.assertEqual(len(self.analysis.functions), 0)
        self.assertEqual(len(self.analysis.classes), 0)
        self.assertEqual(len(self.analysis.imports), 0)


class TestGoAnalyzerMethods(unittest.TestCase):
    """Tests using fixtures/go/methods.go"""

    def setUp(self):
        self.analysis = analyze_go(os.path.join(FIXTURES_DIR, "methods.go"))

    def test_package_name(self):
        self.assertEqual(self.analysis.package, "methods")

    def test_method_count(self):
        self.assertEqual(len(self.analysis.functions), 3)

    def test_method_names(self):
        names = [f.name for f in self.analysis.functions]
        self.assertIn("Repo.Find", names)
        self.assertIn("Repo.Save", names)
        self.assertIn("Repo.Delete", names)

    def test_struct_count(self):
        # Repo struct and DB interface
        self.assertEqual(len(self.analysis.classes), 2)

    def test_repo_struct_kind(self):
        repo = next(c for c in self.analysis.classes if c.name == "Repo")
        self.assertEqual(repo.kind, "struct")


class TestGoStructFields(unittest.TestCase):
    """Tests for struct field extraction — the main feature of this PR."""

    def setUp(self):
        self.analysis = analyze_go(os.path.join(FIXTURES_DIR, "basic.go"))
        self.user = next(c for c in self.analysis.classes if c.name == "User")
        self.config = next(c for c in self.analysis.classes if c.name == "Config")

    def test_user_field_count(self):
        """User struct should have 4 fields: ID, Name, Email, CreatedAt."""
        self.assertEqual(len(self.user.fields), 4)

    def test_user_field_names(self):
        names = [f.name for f in self.user.fields]
        self.assertEqual(names, ["ID", "Name", "Email", "CreatedAt"])

    def test_user_field_types(self):
        types = [f.type_str for f in self.user.fields]
        self.assertEqual(types, ["int", "string", "string", "time.Time"])

    def test_user_field_tags(self):
        """Each User field should have json, db, and yaml tags."""
        tags = [f.tag for f in self.user.fields]
        self.assertEqual(len(tags), 4)
        # ID tags
        self.assertIn('json:"id"', tags[0])
        self.assertIn('db:"id"', tags[0])
        self.assertIn('yaml:"id"', tags[0])
        # Name tags (with omitempty)
        self.assertIn('json:"name,omitempty"', tags[1])
        self.assertIn('yaml:"name,omitempty"', tags[1])

    def test_config_field_count(self):
        """Config struct should have 2 fields: Debug, Port."""
        self.assertEqual(len(self.config.fields), 2)

    def test_config_field_names(self):
        names = [f.name for f in self.config.fields]
        self.assertEqual(names, ["Debug", "Port"])

    def test_config_field_types(self):
        types = [f.type_str for f in self.config.fields]
        self.assertEqual(types, ["bool", "int"])

    def test_interface_has_no_fields(self):
        """Reader interface should have no fields."""
        reader = next(c for c in self.analysis.classes if c.name == "Reader")
        self.assertEqual(len(reader.fields), 0)


class TestGoChunkerFields(unittest.TestCase):
    """Tests that struct fields propagate into chunk metadata."""

    def setUp(self):
        self.analysis = analyze_go(os.path.join(FIXTURES_DIR, "basic.go"))
        self.chunks = chunk_file_analysis(self.analysis)

    def test_struct_chunk_has_fields_key(self):
        """Struct chunks should have a 'fields' key in metadata."""
        struct_chunks = [
            c for c in self.chunks
            if c["metadata"].get("symbol_type") == "struct"
        ]
        self.assertTrue(len(struct_chunks) > 0)
        for chunk in struct_chunks:
            self.assertIn("fields", chunk["metadata"])

    def test_user_chunk_fields_content(self):
        """User struct chunk fields should contain ID, Name, Email, CreatedAt."""
        user_chunk = next(
            c for c in self.chunks
            if c["metadata"].get("class_name") == "User"
        )
        fields = user_chunk["metadata"]["fields"]
        self.assertEqual(len(fields), 4)
        self.assertEqual(fields[0]["name"], "ID")
        self.assertEqual(fields[0]["type_str"], "int")
        self.assertIn('json:"id"', fields[0]["tag"])

    def test_interface_chunk_has_empty_fields(self):
        """Interface chunks should have an empty fields list."""
        reader_chunk = next(
            c for c in self.chunks
            if c["metadata"].get("class_name") == "Reader"
        )
        self.assertEqual(reader_chunk["metadata"]["fields"], [])


class TestGoChunkerEmpty(unittest.TestCase):
    """Tests that empty files produce a full-file fallback chunk."""

    def setUp(self):
        self.analysis = analyze_go(os.path.join(FIXTURES_DIR, "empty.go"))
        self.chunks = chunk_file_analysis(self.analysis)

    def test_empty_file_has_fallback_chunk(self):
        """Empty file should have a summary + a fallback file chunk."""
        types = [c["type"] for c in self.chunks]
        self.assertIn("file_summary", types)
        self.assertIn("file", types)


if __name__ == "__main__":
    unittest.main()
