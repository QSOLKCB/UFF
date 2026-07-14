from __future__ import annotations

import json
from pathlib import Path

from uff.cli import main


def test_fit_cli_writes_reproducible_outputs(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    result = main(
        [
            "fit",
            "--csv",
            str(repository / "DEMO_GALAXY.csv"),
            "--gal",
            "CLI_TEST",
            "--out",
            str(tmp_path),
            "--models",
            "nfw,mond-rar",
            "--fixed-stellar-ml",
            "--restarts",
            "2",
            "--max-nfev",
            "3000",
        ]
    )
    assert result == 0
    summary_path = tmp_path / "CLI_TEST_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert summary["schema"] == "uff.rotation-curve-summary.v4"
    assert summary["best_model"] in {"nfw", "mond-rar"}
    assert (tmp_path / "CLI_TEST_models.png").exists()


def test_compact_cli_writes_json(tmp_path):
    path = tmp_path / "compact.json"
    result = main(
        [
            "compact-object",
            "--mass-msun",
            "4300000",
            "--spin",
            "0.5",
            "--out",
            str(path),
        ]
    )
    assert result == 0
    payload = json.loads(path.read_text())
    assert payload["kerr"]["mass_msun"] == 4_300_000.0
    assert "area_gap_over_radius_squared" in payload["lqg"]


def test_models_command_lists_canonical_names(capsys):
    assert main(["models"]) == 0
    output = capsys.readouterr().out
    assert "nfw" in output
    assert "mond-efe" in output
    assert "uff-empirical" in output
