from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).parents[1]
README = ROOT / "README.md"
PYPROJECT = ROOT / "pyproject.toml"
CITATION = ROOT / "CITATION.cff"
CHANGELOG = ROOT / "CHANGELOG.md"
PACKAGE_INIT = ROOT / "uff" / "__init__.py"
RELEASE_NOTES = ROOT / "RELEASE_NOTES_v5.1.0.md"
ZENODO = ROOT / ".zenodo.json"
LICENSE = ROOT / "LICENSE"
ZENODO_MANIFEST = ROOT / "zenodo" / "v5.1.0" / "MANIFEST.json"
ZENODO_SUMS = ROOT / "zenodo" / "v5.1.0" / "SHA256SUMS.txt"
OBSOLETE_DOI = "10.5281/zenodo.17669627"
PREVIOUS_VERSION_DOI = "10.5281/zenodo.21830630"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v5_1_release_metadata_is_consistent_and_pending_doi_is_not_fabricated() -> None:
    readme = README.read_text(encoding="utf-8")
    pyproject = PYPROJECT.read_text(encoding="utf-8")
    citation = CITATION.read_text(encoding="utf-8")
    changelog = CHANGELOG.read_text(encoding="utf-8")
    package_init = PACKAGE_INIT.read_text(encoding="utf-8")
    release_notes = RELEASE_NOTES.read_text(encoding="utf-8")
    zenodo = json.loads(ZENODO.read_text(encoding="utf-8"))
    manifest = json.loads(ZENODO_MANIFEST.read_text(encoding="utf-8"))
    sums = ZENODO_SUMS.read_text(encoding="utf-8")

    assert "QSOL UFF v5.1.0" in readme
    assert "# QSOL UFF v5.1.0" in release_notes
    assert "## [5.1.0] - 2026-08-07" in changelog
    assert re.search(r'^version = "5\.1\.0"$', pyproject, re.MULTILINE)
    assert re.search(r'^version: 5\.1\.0$', citation, re.MULTILINE)
    assert '__version__ = "5.1.0"' in package_init
    assert re.search(
        r'^Changelog = "https://github\.com/QSOLKCB/UFF/blob/main/CHANGELOG\.md"$',
        pyproject,
        re.MULTILINE,
    )
    assert re.search(
        r'^Release = "https://github\.com/QSOLKCB/UFF/releases/tag/v5\.1\.0"$',
        pyproject,
        re.MULTILINE,
    )
    assert re.search(
        r'^"Previous Zenodo Archive" = "https://doi\.org/10\.5281/zenodo\.21830630"$',
        pyproject,
        re.MULTILINE,
    )

    for text in (readme, pyproject, citation, changelog, release_notes):
        assert OBSOLETE_DOI not in text

    # The immutable v5.0.0 version DOI remains historical context only.
    for text in (readme, pyproject, changelog, release_notes):
        assert PREVIOUS_VERSION_DOI in text

    # Do not fabricate a v5.1.0 version DOI before Zenodo publishes the new version.
    assert not re.search(r"^doi:", citation, re.MULTILINE)
    assert re.search(
        r'^url: "https://github\.com/QSOLKCB/UFF/releases/tag/v5\.1\.0"$',
        citation,
        re.MULTILINE,
    )

    canonical_title = (
        "QSOL UFF v5.1.0: Defense-in-Depth Trust, Witness, Calibration, "
        "and Telemetry for Reproducible Astrophysics"
    )
    assert zenodo["title"] == canonical_title
    assert f'title: "{canonical_title}"' in citation
    assert zenodo["upload_type"] == "software"
    assert zenodo["version"] == "5.1.0"
    assert zenodo["creators"] == [
        {
            "name": "Slade, Trent",
            "orcid": "0009-0002-4515-9237",
            "affiliation": "QSOL-IMC",
        }
    ]
    assert any(
        contributor["name"] == "OpenAI ChatGPT"
        for contributor in zenodo["contributors"]
    )
    assert any(
        item["identifier"] == "https://github.com/QSOLKCB/UFF/releases/tag/v5.1.0"
        and item["relation"] == "isIdenticalTo"
        for item in zenodo["related_identifiers"]
    )

    manifest_entries = {item["path"]: item for item in manifest["files"]}
    for path in (CITATION, LICENSE):
        relative = path.relative_to(ROOT).as_posix()
        entry = manifest_entries[relative]
        assert entry["bytes"] == path.stat().st_size
        assert entry["sha256"] == _sha256(path)
        assert f"{_sha256(path)}  {relative}\n" in sums

    assert re.search(
        r"^\| `uff\.sheridan-crucible\.v2` \| [^|\n]+ \| Planned, not implemented \|$",
        readme,
        re.MULTILINE,
    )
    assert re.search(
        r"^\| `ENSEMBLE_CALIBRATED` \| [^|\n]+ \| Planned, not implemented \|$",
        readme,
        re.MULTILINE,
    )
