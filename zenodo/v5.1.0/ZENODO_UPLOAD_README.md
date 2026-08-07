# QSOL UFF v5.1.0 - Zenodo upload package

This directory contains the supporting archival material for the Zenodo **new version** that supersedes QSOL UFF v5.0.0 (DOI `10.5281/zenodo.21830630`).

## Recommended record type

**Software**

## Recommended version

`5.1.0`

## Recommended title

**QSOL UFF v5.1.0: Defense-in-Depth Trust, Witness, Calibration, and Telemetry for Reproducible Astrophysics**

## Recommended files

Upload/preserve:

1. the source archive generated from the GitHub tag `v5.1.0`;
2. `UFF_v5.1.0_DEFENSE_IN_DEPTH_TECHNICAL_REPORT.pdf`;
3. `UFF_v5.1.0_DEFENSE_IN_DEPTH_TECHNICAL_REPORT.md`;
4. `RELEASE_NOTES_v5.1.0.md`;
5. `CITATION.cff`;
6. `.zenodo.json` (or use its values in the Zenodo metadata form);
7. `LICENSE`;
8. `README.md` snapshot; and
9. `SHA256SUMS.txt` plus `MANIFEST.json` for the supporting package.

The GitHub source archive is the canonical software payload. The PDF report is the formal methods/assurance document accompanying that software release.

## Versioning workflow

Use **New version** from record 21830630 rather than editing the files of the already-published v5.0.0 record. Zenodo will preserve the previous version and issue a new version DOI when v5.1.0 is published.

## Post-publication DOI patch

After Zenodo publishes v5.1.0, update the repository with the newly assigned version DOI in:

- `README.md` DOI badge/citation;
- `CITATION.cff` (`doi` and `url`);
- `pyproject.toml` project DOI URL;
- `CHANGELOG.md` v5.1.0 release link; and
- any release description where the final DOI is desired.

Do not replace the historical v5.0.0 DOI where it is explicitly identified as the previous archived version.

## Scientific boundary

The new record should describe QEC/SPECTRAL/SONIFICATION as narrow engineering mechanisms and the statistical-mechanics material as an interpretation guardrail. Do not describe these layers as proving scientific truth, physical ontology, historical blindness, catalogue independence, or null-model adequacy.
