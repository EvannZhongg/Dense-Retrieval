import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import conftest


def test_basetemp_defaults_inside_the_repository():
    """An unreadable platform temp directory must not be used for tmp_path."""
    config = SimpleNamespace(option=SimpleNamespace(basetemp=None))

    conftest.pytest_configure(config)

    assert Path(config.option.basetemp) == conftest.ROOT / "tmp" / "pytest"


def test_basetemp_keeps_a_user_supplied_directory():
    given = Path("D:/elsewhere")

    config = SimpleNamespace(option=SimpleNamespace(basetemp=given))
    conftest.pytest_configure(config)

    assert Path(config.option.basetemp) == given
