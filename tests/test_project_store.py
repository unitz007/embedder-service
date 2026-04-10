# tests/test_project_store.py
"""Unit tests for store/project_store.py — JSON-file-backed Project store."""

import os
import tempfile
import pytest

from store.project_store import ProjectStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    return ProjectStore(str(tmp_path))


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------

class TestCreate:
    def test_basic_create(self, store):
        project = store.create(name="my-proj", namespace="ns1")
        assert project["name"] == "my-proj"
        assert project["namespace"] == "ns1"
        assert "id" in project
        assert project["created_at"] is not None

    def test_create_with_all_fields(self, store):
        project = store.create(
            name="full",
            namespace="ns2",
            description="A full project",
            github_owner="org",
            github_repo="repo",
            github_ref="main",
        )
        assert project["description"] == "A full project"
        assert project["github_owner"] == "org"
        assert project["github_repo"] == "repo"
        assert project["github_ref"] == "main"

    def test_create_generates_unique_ids(self, store):
        a = store.create(name="a", namespace="ns")
        b = store.create(name="b", namespace="ns")
        assert a["id"] != b["id"]


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------

class TestGet:
    def test_get_existing(self, store):
        created = store.create(name="x", namespace="ns")
        fetched = store.get(created["id"])
        assert fetched["name"] == "x"
        assert fetched["id"] == created["id"]

    def test_get_missing(self, store):
        assert store.get("nonexistent") is None

    def test_persists_across_instances(self, tmp_path):
        s1 = ProjectStore(str(tmp_path))
        created = s1.create(name="persist", namespace="ns")
        s2 = ProjectStore(str(tmp_path))
        fetched = s2.get(created["id"])
        assert fetched is not None
        assert fetched["name"] == "persist"


# ---------------------------------------------------------------------------
# list_all
# ---------------------------------------------------------------------------

class TestListAll:
    def test_empty(self, store):
        assert store.list_all() == []

    def test_returns_all(self, store):
        store.create(name="a", namespace="ns1")
        store.create(name="b", namespace="ns2")
        result = store.list_all()
        assert len(result) == 2

    def test_filter_by_namespace(self, store):
        store.create(name="a", namespace="ns1")
        store.create(name="b", namespace="ns2")
        store.create(name="c", namespace="ns1")
        result = store.list_all(namespace="ns1")
        assert len(result) == 2
        assert all(p["namespace"] == "ns1" for p in result)

    def test_sorted_by_created_at_desc(self, store):
        first = store.create(name="first", namespace="ns")
        second = store.create(name="second", namespace="ns")
        result = store.list_all()
        assert result[0]["id"] == second["id"]
        assert result[1]["id"] == first["id"]


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------

class TestUpdate:
    def test_update_name(self, store):
        created = store.create(name="old", namespace="ns")
        updated = store.update(created["id"], name="new")
        assert updated["name"] == "new"
        assert updated["updated_at"] != created["updated_at"]

    def test_update_missing(self, store):
        assert store.update("nonexistent", name="x") is None

    def test_update_ignores_disallowed_fields(self, store):
        created = store.create(name="proj", namespace="ns")
        updated = store.update(created["id"], name="updated", id="hacked")
        assert updated["name"] == "updated"
        assert updated["id"] == created["id"]  # id should not change

    def test_partial_update(self, store):
        created = store.create(name="proj", namespace="ns", description="old desc")
        updated = store.update(created["id"], description="new desc")
        assert updated["description"] == "new desc"
        assert updated["name"] == "proj"


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------

class TestDelete:
    def test_delete_existing(self, store):
        created = store.create(name="del-me", namespace="ns")
        assert store.delete(created["id"]) is True
        assert store.get(created["id"]) is None

    def test_delete_missing(self, store):
        assert store.delete("nonexistent") is False

    def test_delete_removes_from_list(self, store):
        store.create(name="a", namespace="ns")
        created = store.create(name="b", namespace="ns")
        store.delete(created["id"])
        assert len(store.list_all()) == 1
