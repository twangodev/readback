from __future__ import annotations

from readback import store


def test_write_then_read_roundtrips(tmp_path):
    path = tmp_path / "nested" / "shard.jsonl"
    written = store.write_jsonl(path, [{"a": 1}, {"a": 2}])
    assert written == 2
    assert list(store.read_jsonl(path)) == [{"a": 1}, {"a": 2}]


def test_write_leaves_no_staging_file(tmp_path):
    path = tmp_path / "shard.jsonl"
    store.write_jsonl(path, [{"a": 1}])
    assert not path.with_name(path.name + ".tmp").exists()


def test_read_skips_blank_lines(tmp_path):
    path = tmp_path / "shard.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n')
    assert list(store.read_jsonl(path)) == [{"a": 1}, {"a": 2}]


def test_index_by_keys_on_field(tmp_path):
    path = tmp_path / "shard.jsonl"
    store.write_jsonl(path, [{"id": "u0", "v": 1}, {"id": "u1", "v": 2}])
    assert store.index_by(path, "id")["u1"]["v"] == 2


def test_append_jsonl_appends_and_creates_parents(tmp_path):
    path = tmp_path / "nested" / "wal.jsonl"
    store.append_jsonl(path, {"a": 1})
    store.append_jsonl(path, {"a": 2})
    assert list(store.read_jsonl(path)) == [{"a": 1}, {"a": 2}]
