# UFF v5.3.0 — Human-Readable Related Identifiers

This note explains the Zenodo **Related identifiers** for QSOL UFF v5.3.0 in ordinary language.

Zenodo's relationship labels are precise but can be easy to misread when filling out the form. The entries below say both **what to enter** and **what it means**.

## 1. GitHub release for v5.3.0

**Identifier**

`https://github.com/QSOLKCB/UFF/releases/tag/v5.3.0`

**Zenodo relation**

`Is identical to`

**Scheme**

`URL`

**Plain-English meaning**

This is the GitHub release page for the exact UFF v5.3.0 software release archived by this Zenodo record.

The important identity is the frozen `v5.3.0` tag, which resolves to release commit:

`1e310b257ec51c92cccf12271900cec5aa972c50`

The phrase **“Is identical to”** means that the Zenodo deposit and this GitHub release represent the same released software version. It does **not** mean that every byte of the GitHub web page is identical to the Zenodo record.

Use this relation when Zenodo asks how the GitHub v5.3.0 release relates to the deposited software.

---

## 2. Main UFF GitHub repository

**Identifier**

`https://github.com/QSOLKCB/UFF`

**Zenodo relation**

`Is a version of`

**Scheme**

`URL`

**Plain-English meaning**

UFF v5.3.0 is one specific released version of the continuing QSOLKCB/UFF project.

The repository can continue to change after v5.3.0. Therefore the repository as a whole is **not identical** to the frozen Zenodo deposit.

In ordinary language:

> This Zenodo record is the v5.3.0 version of the broader UFF software project hosted at this repository.

---

## 3. Previous UFF Zenodo version

**Identifier**

`10.5281/zenodo.21911644`

**Zenodo relation**

`Is a new version of`

**Scheme**

`DOI`

**Plain-English meaning**

UFF v5.3.0 succeeds the previously published UFF v5.2.0 Zenodo record.

In ordinary language:

> This v5.3.0 deposit is the next published version after UFF v5.2.0.

The v5.2.0 DOI remains immutable historical provenance. It is not replaced or reused by v5.3.0.

---

## 4. Jampolski & Rezzolla gravastar paper

**Identifier**

`https://arxiv.org/abs/2509.15302`

**Zenodo relation**

`References`

**Scheme**

`URL`

**Plain-English meaning**

UFF v5.3.0 cites this paper as the worked scientific example for its **nonclaim calibration** framework.

The paper is used to illustrate distinctions such as:

`can occur != likely to occur`

`fine-tuned success != robust success`

`constructed trajectory != generic outcome`

`theoretical possibility != empirical occurrence`

The relation **“References”** does not mean that the paper proves UFF, endorses UFF, or supplies empirical evidence for gravastars. It simply means that UFF v5.3.0 refers to this source as part of its documented calibration example.

---

## 5. QSOL-NEXUS v1.0.0 archive

**Identifier**

`https://github.com/QSOLKCB/QSOL-NEXUS/tree/main/archives/v1.0.0`

**Zenodo relation**

`References`

**Scheme**

`URL`

**Plain-English meaning**

UFF references this archived QSOL-NEXUS material as architectural and provenance context for parts of its assurance design lineage.

The NEXUS archive is **not** the UFF release, is **not** a dependency that defines the UFF v5.3.0 source identity, and is **not** proof authority for UFF.

In ordinary language:

> UFF v5.3.0 points to this NEXUS archive as documented design/provenance context.

---

# Quick copy guide

When filling in Zenodo's **Related identifiers** section, use:

| Identifier | Relation to select | Scheme | What it means |
|---|---|---|---|
| `https://github.com/QSOLKCB/UFF/releases/tag/v5.3.0` | `Is identical to` | `URL` | Same released software version as this Zenodo deposit |
| `https://github.com/QSOLKCB/UFF` | `Is a version of` | `URL` | v5.3.0 is one version of the continuing UFF project |
| `10.5281/zenodo.21911644` | `Is a new version of` | `DOI` | v5.3.0 succeeds the published v5.2.0 record |
| `https://arxiv.org/abs/2509.15302` | `References` | `URL` | Scientific source used for the nonclaim-calibration example |
| `https://github.com/QSOLKCB/QSOL-NEXUS/tree/main/archives/v1.0.0` | `References` | `URL` | Architectural/provenance context referenced by UFF |

# Release identity reminder

The published UFF v5.3.0 identity is:

- **GitHub tag:** `v5.3.0`
- **Exact release commit:** `1e310b257ec51c92cccf12271900cec5aa972c50`
- **Zenodo DOI:** `10.5281/zenodo.22026554`

Later metadata-only commits on `main` do not change that frozen release identity and must not cause the `v5.3.0` tag to be moved or recreated.
