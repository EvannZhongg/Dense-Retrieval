import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from dense_retrieval.embeddings.cache import find_cache_dir, find_query_cache_path


def _complete_cache(path):
    path.mkdir(parents=True)
    (path / "queries.npy").touch()
    (path / "documents.npy").touch()


def _temporary_root():
    root = Path(__file__).parents[1] / "tmp"
    root.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(dir=root)


def test_cache_discovery_prefers_unique_explicit_migration_target():
    with _temporary_root() as directory:
        tmp_path = Path(directory)
        root = tmp_path / "cache" / "dataset" / "model"
        source = root / "source"
        target = root / "target"
        _complete_cache(source)
        _complete_cache(target)
        (source / "queries_train.npy").touch()
        (target / "empty_text_migration.json").write_text(
            json.dumps({"source_cache_key": "source"}), encoding="utf-8"
        )

        selected = find_cache_dir(tmp_path / "cache", "dataset", "model")

        assert selected == target
        assert find_query_cache_path(selected, "queries_train.npy") == (
            source / "queries_train.npy"
        )


def test_query_cache_prefers_file_on_migration_target():
    with _temporary_root() as directory:
        target = Path(directory) / "target"
        target.mkdir()
        direct = target / "queries_dev.npy"
        direct.touch()
        (target / "empty_text_migration.json").write_text(
            json.dumps({"source_cache_key": "source"}), encoding="utf-8"
        )

        assert find_query_cache_path(target, "queries_dev.npy") == direct


def test_cache_discovery_rejects_ambiguous_unmigrated_candidates():
    with _temporary_root() as directory:
        tmp_path = Path(directory)
        root = tmp_path / "cache" / "dataset" / "model"
        _complete_cache(root / "first")
        _complete_cache(root / "second")

        try:
            find_cache_dir(tmp_path / "cache", "dataset", "model")
        except RuntimeError as error:
            assert "found 2" in str(error)
        else:
            raise AssertionError("ambiguous caches must be rejected")
