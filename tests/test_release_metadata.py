from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).parents[1]
README = ROOT / "README.md"
PYPROJECT = ROOT / "pyproject.toml"
CITATION = ROOT / "CITATION.cff"
CHANGELOG = ROOT / "CHANGELOG.md"
PACKAGE_INIT = ROOT / "uff" / "__init__.py"
RELEASE_NOTES = ROOT / "RELEASE_NOTES_v5.3.0.md"
ZENODO = ROOT / ".zenodo.json"
ZENODO_V5_3 = ROOT / "zenodo" / "v5.3.0" / "metadata.json"
ZENODO_V5_3_GUIDE = ROOT / "zenodo" / "v5.3.0" / "ZENODO_UPLOAD_README.md"
ZENODO_V5_2 = ROOT / "zenodo" / "v5.2.0" / "metadata.json"
ZENODO_V5_2_GUIDE = ROOT / "zenodo" / "v5.2.0" / "ZENODO_UPLOAD_README.md"
ZENODO_V5_1 = ROOT / "zenodo" / "v5.1.0" / "metadata.json"
ZENODO_V5_1_MANIFEST = ROOT / "zenodo" / "v5.1.0" / "MANIFEST.json"
OBSOLETE_DOI = "10.5281/zenodo.17669627"
PUBLISHED_V5_0_DOI = "10.5281/zenodo.21830630"
PUBLISHED_V5_2_DOI = "10.5281/zenodo.21911644"
PUBLISHED_V5_3_DOI = "10.5281/zenodo.22026554"
RELEASE_COMMIT = "1e310b257ec51c92cccf12271900cec5aa972c50"


def test_v5_3_release_metadata_is_consistent_and_doi_is_bound() -> None:
    readme = README.read_text(encoding="utf-8")
    pyproject = PYPROJECT.read_text(encoding="utf-8")
    citation = CITATION.read_text(encoding="utf-8")
    changelog = CHANGELOG.read_text(encoding="utf-8")
    package_init = PACKAGE_INIT.read_text(encoding="utf-8")
    release_notes = RELEASE_NOTES.read_text(encoding="utf-8")
    zenodo_guide = ZENODO_V5_3_GUIDE.read_text(encoding="utf-8")
    zenodo = json.loads(ZENODO.read_text(encoding="utf-8"))
    zenodo_v5_3 = json.loads(ZENODO_V5_3.read_text(encoding="utf-8"))

    assert "QSOL UFF v5.3.0" in readme
    assert "# QSOL UFF v5.3.0" in release_notes
    assert "## [5.3.0] - 2026-08-20" in changelog
    assert re.search(r'^version = "5\.3\.0"$', pyproject, re.MULTILINE)
    assert re.search(r'^version: 5\.3\.0$', citation, re.MULTILINE)
    assert '__version__ = "5.3.0"' in package_init
    assert re.search(
        r'^Release = "https://github\.com/QSOLKCB/UFF/releases/tag/v5\.3\.0"$',
        pyproject,
        re.MULTILINE,
    )
    assert re.search(
        r'^DOI = "https://doi\.org/10\.5281/zenodo\.22026554"$',
        pyproject,
        re.MULTILINE,
    )
    assert re.search(
        r'^"Previous Zenodo v5\.2\.0" = "https://doi\.org/10\.5281/zenodo\.21911644"$',
        pyproject,
        re.MULTILINE,
    )
    assert re.search(
        r'^"Zenodo v5\.0\.0 Archive" = "https://doi\.org/10\.5281/zenodo\.21830630"$',
        pyproject,
        re.MULTILINE,
    )

    current_texts = (
        readme,
        pyproject,
        citation,
        changelog,
        release_notes,
        zenodo_guide,
    )
    for text in current_texts:
        assert OBSOLETE_DOI not in text
        assert PUBLISHED_V5_3_DOI in text

    for text in (readme, pyproject, changelog):
        assert PUBLISHED_V5_0_DOI in text
    for text in current_texts:
        assert PUBLISHED_V5_2_DOI in text

    assert "pending assignment" not in readme
    assert "pending assignment" not in citation
    assert re.search(
        r'^doi: "10\.5281/zenodo\.22026554"$',
        citation,
        re.MULTILINE,
    )
    assert re.search(
        r'^url: "https://doi\.org/10\.5281/zenodo\.22026554"$',
        citation,
        re.MULTILINE,
    )

    canonical_title = (
        "QSOL UFF v5.3.0: Nonclaim Calibration and Evidence-Scope Discipline "
        "for Reproducible Astrophysics"
    )
    assert zenodo["title"] == canonical_title
    assert zenodo_v5_3 == zenodo
    assert f'title: "{canonical_title}"' in citation
    assert zenodo["upload_type"] == "software"
    assert zenodo["publication_date"] == "2026-08-20"
    assert zenodo["version"] == "5.3.0"
    assert PUBLISHED_V5_3_DOI in zenodo["description"]
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
        item["identifier"] == "https://github.com/QSOLKCB/UFF/releases/tag/v5.3.0"
        and item["relation"] == "isIdenticalTo"
        for item in zenodo["related_identifiers"]
    )
    assert any(
        item["identifier"] == "https://arxiv.org/abs/2509.15302"
        and item["relation"] == "references"
        for item in zenodo["related_identifiers"]
    )
    assert any(
        item["identifier"]
        == "https://github.com/QSOLKCB/QSOL-NEXUS/tree/main/archives/v1.0.0"
        and item["relation"] == "references"
        for item in zenodo["related_identifiers"]
    )

    assert RELEASE_COMMIT in release_notes
    assert RELEASE_COMMIT in zenodo_guide
    assert "Do **not** move, recreate, or retag `v5.3.0`" in zenodo_guide

    assert re.search(
        r"^\| `uff\.nonclaim-reference\.v1` \| [^|\n]+ \| No; provenance/calibration record \|$",
        readme,
        re.MULTILINE,
    )
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


def test_v5_2_zenodo_snapshot_remains_historical_and_immutable() -> None:
    metadata = json.loads(ZENODO_V5_2.read_text(encoding="utf-8"))
    guide = ZENODO_V5_2_GUIDE.read_text(encoding="utf-8")

    assert metadata["version"] == "5.2.0"
    assert metadata["title"].startswith("QSOL UFF v5.2.0:")
    assert PUBLISHED_V5_2_DOI in guide
    assert "v5.2.0` tag" in guide


def test_v5_1_zenodo_support_package_remains_historical() -> None:
    metadata = json.loads(ZENODO_V5_1.read_text(encoding="utf-8"))
    manifest = json.loads(ZENODO_V5_1_MANIFEST.read_text(encoding="utf-8"))

    assert metadata["version"] == "5.1.0"
    assert metadata["title"].startswith("QSOL UFF v5.1.0:")
    assert manifest["release"] == "5.1.0"
    assert manifest["previous_zenodo_doi"] == PUBLISHED_V5_0_DOI
    assert any(item["path"] == "CITATION.cff" for item in manifest["files"])
    assert any(item["path"] == "README.md" for item in manifest["files"])
