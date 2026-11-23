"""imbed ships no pickle and never unpickles a file from its own package (addaix/imbed#23)."""

from pathlib import Path

import imbed
from imbed.components.components_util import get_standard_components

PACKAGE = Path(imbed.__file__).parent


def test_no_pickle_files_ship_in_the_package():
    assert not list(PACKAGE.rglob("*.pickle")) and not list(PACKAGE.rglob("*.pkl"))


def test_standard_components_come_from_the_modules():
    components = get_standard_components()
    assert set(components) >= {"segmenters", "embedders", "clusterers", "planarizers"}
    assert get_standard_components(refresh=True) == components


def test_components_util_does_not_unpickle():
    source = (PACKAGE / "components" / "components_util.py").read_text()
    assert "pickle" not in source
