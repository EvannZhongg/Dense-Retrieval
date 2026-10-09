import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

BASETEMP = ROOT / "tmp" / "pytest"


def pytest_configure(config) -> None:
    """Pin pytest's base temporary directory inside the repository.

    ``TEMP``/``TMP`` can point at a directory the test process is not allowed
    to read (for example the Windows ``Temp`` directory), which fails every
    ``tmp_path`` test during setup. A base directory under ``tmp/`` avoids
    that, and an explicit ``--basetemp`` still wins.
    """
    if config.option.basetemp is None:
        # pytest deletes and recreates the base directory itself, but it does
        # not create missing parents, so ``tmp/`` has to exist first.
        BASETEMP.parent.mkdir(parents=True, exist_ok=True)
        config.option.basetemp = BASETEMP
