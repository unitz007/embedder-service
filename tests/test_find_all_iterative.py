"""Tests for the iterative _find_all in generic_ts_analyzer."""

from analyzers.generic_ts_analyzer import _find_all


class FakeNode:
    """Minimal tree-sitter node stand-in for testing."""

    def __init__(self, node_type: str, children=None):
        self.type = node_type
        self.children = children or []

    def __repr__(self):
        return f"FakeNode({self.type!r})"


def _build_tree(*desc):
    """Build a tree from (type, [child, ...]) tuples.

    Leaf: just a string like "identifier"
    Branch: ("call_expression", [child, ...])
    """
    if isinstance(desc, str):
        return FakeNode(desc)
    node_type, children = desc
    return FakeNode(node_type, [_build_tree(c) for c in children])


class TestFindAllIterative:
    """Verify the iterative _find_all matches expected behavior."""

    def test_no_matches(self):
        root = _build_tree("module", ["function_item", "variable"])
        results = _find_all(root, ["call_expression"])
        assert results == []

    def test_single_match(self):
        root = _build_tree("module", [
            ("call_expression", ["identifier"]),
            "function_item",
        ])
        results = _find_all(root, ["call_expression"])
        assert len(results) == 1
        assert results[0].type == "call_expression"

    def test_multiple_matches(self):
        root = _build_tree("module", [
            ("call_expression", ["a"]),
            "function_item",
            ("call_expression", ["b"]),
        ])
        results = _find_all(root, ["call_expression"])
        assert len(results) == 2

    def test_nested_matches(self):
        """Deeply nested call_expression nodes are still found."""
        root = _build_tree("module", [
            ("block", [
                ("if_statement", [
                    ("call_expression", ["f"]),
                ]),
                ("call_expression", ["g"]),
            ]),
        ])
        results = _find_all(root, ["call_expression"])
        assert len(results) == 2

    def test_multiple_types(self):
        """Searching for multiple node types returns both."""
        root = _build_tree("module", [
            ("function_item", []),
            ("struct_item", []),
            "variable",
            ("function_item", []),
        ])
        results = _find_all(root, ["function_item", "struct_item"])
        assert len(results) == 3

    def test_deeply_nested_no_stack_overflow(self):
        """Very deep nesting should not hit recursion limit."""
        # Build a chain 5000 levels deep
        node = FakeNode("leaf")
        for _ in range(5000):
            node = FakeNode("inner", [node])
        results = _find_all(node, ["leaf"])
        assert len(results) == 1

    def test_empty_tree(self):
        root = FakeNode("module", [])
        results = _find_all(root, ["anything"])
        assert results == []

    def test_preserves_left_to_right_order(self):
        """Results should appear in document order (left-to-right)."""
        root = _build_tree("module", [
            ("call_expression", ["a"]),
            ("call_expression", ["b"]),
            ("call_expression", ["c"]),
        ])
        results = _find_all(root, ["call_expression"])
        types = [r.children[0].type for r in results]
        assert types == ["a", "b", "c"]
