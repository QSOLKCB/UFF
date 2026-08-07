from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).parents[1]
README = ROOT / "README.md"
PYPROJECT = ROOT / "pyproject.toml"
CITATION = ROOT / "CITATION.cff"
CHANGELOG = ROOT / "CHANGELOG.md"
PACKAGE_INIT = ROOT / "uff" / "__init__.py"
RELEASE_NOTES = ROOT / "RELEASE_NOTES_v5.0.0.md"
OLD_DOI = "10.5281/zenodo.17669627"


def test_v5_release_metadata_is_consistent_and_old_doi_is_removed() -> None:
    readme = README.read_text(encoding="utf-8")
    pyproject = PYPROJECT.read_text(encoding="utf-8")
    citation = CITATION.read_text(encoding="utf-8")
    changelog = CHANGELOG.read_text(encoding="utf-8")
    package_init = PACKAGE_INIT.read_text(encoding="utf-8")
    release_notes = RELEASE_NOTES.read_text(encoding="utf-8")

    assert "QSOL UFF v5.0.0" in readme
    assert "# QSOL UFF v5.0.0" in release_notes
    assert "## [5.0.0] - 2026-08-07" in changelog
    assert re.search(r'^version = "5\.0\.0"$', pyproject, re.MULTILINE)
    assert re.search(r'^version: 5\.0\.0$', citation, re.MULTILINE)
    assert '__version__ = "5.0.0"' in package_init
    assert re.search(
        r'^Changelog = "https://github\.com/QSOLKCB/UFF/blob/main/CHANGELOG\.md"$',
        pyproject,
        re.MULTILINE,
    )

    for text in (readme, pyproject, citation, changelog, release_notes):
        assert OLD_DOI not in text

    assert re.search(
        r"^\| `uff\.sheridan-crucible\.v2` \| [^|\n]+ \| Planned, not implemented \|$",
        readme,
        re.MULTILINE,
    )
