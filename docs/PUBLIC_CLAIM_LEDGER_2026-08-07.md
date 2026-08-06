# Public Claim Ledger — 7 August 2026

**Status:** provenance record for adversarial preregistration; not an executable confirmatory contract  
**Subject:** publicly stated celestial-node, catalogue-anomaly, and Planet Nine claims attributed to Dr. Logvinovich  
**Purpose:** freeze the public empirical assertions before testing them with UFF-SLFA and the Sheridan Crucible

## Epistemic boundary

This ledger records publicly stated claims and query text. It does **not** endorse the claims, certify the reported query results, or infer missing parameters. The source snapshot must be archived separately with its acquisition time and checksum before publication-grade use.

A Sheridan run must not silently choose among contradictory coordinate sets, radii, thresholds, or anomaly definitions. Unresolved fields below are explicit blockers, not analyst discretion.

## Public claim families

### A. Ideal cuboctahedral twelve-node frame

One public statement defines exactly twelve nodes:

- declination `+45°`, right ascension `[0°, 90°, 180°, 270°]`;
- declination `-45°`, right ascension `[0°, 90°, 180°, 270°]`;
- declination `0°`, right ascension `[45°, 135°, 225°, 315°]`.

The stated morphology is a circular caustic with a strict radius of `5.0°` at each node.

### B. Rotated E-node list

A later public statement supplies a different coordinate list:

| Node | RA (deg) | Dec (deg) |
|---|---:|---:|
| E-1 | 45.7 | 67.8 |
| E-2 | 116.8 | 43.3 |
| E-3 | 135.9 | -31.5 |
| E-4 | 90.6 | -12.8 |
| E-5 | 60.5 | 43.2 |
| E-6 | 225.4 | 12.6 |
| E-8 | 300.2 | -23.1 |
| E-9 | 180.3 | -47.9 |
| E-10 | 315.1 | 55.2 |
| E-11 | 270.8 | 11.4 |

E-7 and E-12 are absent from the captured statement. This ten-coordinate list is not interchangeable with the ideal frame and cannot instantiate a twelve-node contract without clarification.

### C. Query-specific hotspot coordinates

The published extraction text additionally targets:

- `(RA, Dec) = (270.0°, 45.0°)` with radii `1.5°` and `5.0°`;
- `(RA, Dec) = (357.135896°, -47.874018°)` with radius `3.0°`.

The second hotspot is not one of the twelve coordinates in the ideal frame and does not match the partial rotated list. Its relation to the declared node architecture is unresolved.

### D. Thirty-six-node ontology

A separate public summary describes a `36-node` lattice comprising `12` primary E-nodes and `24` secondary overtone nodes. No complete sky-coordinate table for all thirty-six nodes is present in the captured material.

## Published catalogue predicates

### NOIRLab NSC DR2 — three-degree extraction

```sql
SELECT ra, dec, pmra, pmdec, pmraerr, pmdecerr, ndet, deltamjd
FROM nsc_dr2.object
WHERE q3c_radial_query(ra, dec, 357.135896, -47.874018, 3.0)
  AND (
    pmraerr > 90.0
    OR pmdecerr > 90.0
    OR (pmra*pmra + pmdec*pmdec) > 2500.0
  )
ORDER BY deltamjd DESC
```

This is a **node-conditioned extraction**. It can describe records inside the selected region but cannot by itself estimate a full-sky concentration, outside-node rate, or percentage of all anomalies located at nodes.

### NOIRLab NSC DR2 — vector extraction

```sql
SELECT
  ra, dec, pmra, pmdec, pmraerr, pmdecerr, ndet, deltamjd,
  SQRT(pmra*pmra + pmdec*pmdec) AS total_pm,
  (ATAN2(pmra, pmdec) * 180.0 / PI() + 360.0)
    - FLOOR((ATAN2(pmra, pmdec) * 180.0 / PI() + 360.0) / 360.0) * 360.0
      AS pos_angle_deg
FROM nsc_dr2.object
WHERE q3c_radial_query(ra, dec, 270.0, 45.0, 1.5)
  AND ndet >= 3
  AND deltamjd > 365.0
  AND pmra IS NOT NULL
  AND pmdec IS NOT NULL
  AND (
    pmraerr > 50.0
    OR pmdecerr > 50.0
    OR (pmra*pmra + pmdec*pmdec) > 2500.0
  )
ORDER BY SQRT(pmra*pmra + pmdec*pmdec) DESC
```

The reported count is `42,030` persistent records inside `1.5°`. A later statement reports `373,440` records after expanding to `5.0°`, with `ndet >= 3` and `deltamjd > 365`.

The public discriminant states that a single coherent vector direction would support systematic coordinate drift and falsify the proposed interpretation; an isotropic angle distribution is claimed instead. This discriminant can be frozen and tested, but isotropy alone does not identify a unique physical cause.

### AllWISE — rectangular region extraction

```sql
SELECT TOP 500000
  designation, ra, dec, w1mpro, w2mpro, w3mpro, w4mpro,
  (w3mpro - w4mpro) AS w3_w4_color
FROM allwise.source
WHERE ra BETWEEN 267.0 AND 273.0
  AND dec BETWEEN 42.0 AND 45.0
ORDER BY w3_w4_color DESC
```

This query has no colour threshold and samples a rectangle rather than a `5°` spherical cap. Ordering by colour does not establish a spatial excess or a catalogue-wide anomaly rate.

### AllWISE–NSC cross-match

```sql
SELECT
  wise.ra, wise.dec, wise.designation, wise.ph_qual, wise.ext_flg,
  wise.w3mpro, wise.w4mpro,
  (wise.w3mpro - wise.w4mpro) AS w3_w4_color,
  nsc.pmra, nsc.pmdec, nsc.pmraerr, nsc.pmdecerr, nsc.ndet, nsc.deltamjd
FROM allwise.source AS wise
JOIN nsc_dr2.object AS nsc
  ON q3c_join(wise.ra, wise.dec, nsc.ra, nsc.dec, 0.000833)
WHERE q3c_radial_query(wise.ra, wise.dec, 357.135896, -47.874018, 3.0)
  AND wise.w4snr > 3.0
  AND (wise.w3mpro - wise.w4mpro) > 1.5
  AND nsc.deltamjd > 100
ORDER BY w3_w4_color DESC
```

This is a spatially conditioned cross-match. It is useful for inspecting a preselected field, but it is not an independent full-sky replication of node concentration.

### Gaia DR3

The source claims concentration in `astrometric_excess_noise` at the same locations and says exact Gaia TAP queries were supplied. No Gaia ADQL query or frozen Gaia anomaly threshold is present in the captured text. The Gaia component is therefore not reproducible from this snapshot.

## Reported quantitative assertions

The captured public statements include:

- twelve non-overlapping `5°` caps cover `2.283%` of the sphere;
- `100%` of approximately half a million filtered anomalies fall inside those caps;
- a `43.8×` amplification and `p < 10^-300`;
- a target-field failure rate of `51.5%` versus `13.9%` in a control at `+15°` declination;
- a separate Monte Carlo with `100,000` iterations, an ecliptic-bias width of `15°`, and `p < 10^-5`;
- an exact radial fall to zero at `5.0°` across NOIRLab, Gaia, and AllWISE;
- `42,030` NSC records inside `1.5°` and `373,440` inside `5.0°`;
- processing of `1,553,229` MPCORB objects and claimed ETNO-node clustering;
- a residual acceleration of `7.06 × 10^-16 m s^-2` at Saturn and a stated Cassini comparison threshold of `1.0 × 10^-14 m s^-2`;
- a leading non-spherical multipole at `l = 4` and a claimed residual acceleration scaling proportional to `1/r^6`.

These values are recorded as assertions, not verified measurements.

## Material incompatibilities that must be frozen before testing

1. **Node coordinates:** ideal twelve-node frame versus incomplete rotated E-node list versus query hotspots.
2. **Node count:** twelve empirical regions versus a thirty-six-node physical architecture.
3. **Radius:** `1.5°`, `3.0°`, and `5.0°` are all used.
4. **NSC anomaly threshold:** proper-motion error thresholds of `50` and `90` appear in different queries.
5. **Temporal baseline:** `deltamjd > 100` and `deltamjd > 365` both appear.
6. **Claim scope:** an early statement says `100%` of filtered anomalies lie in the windows; a later clarification says the claim is a density spike rather than confinement of every generic high-RUWE source.
7. **Catalogue equivalence:** NSC proper-motion errors, Gaia excess noise, and AllWISE colour/extension are different observables and require separate anomaly models.
8. **Gaia reproducibility:** no Gaia query or frozen threshold appears in the captured text.
9. **Control construction:** the reported `+15°` control lacks a complete coordinate, mask, depth, extinction, and selection-function contract.
10. **Null construction:** the `100,000`-iteration Monte Carlo is described but its code, sampling measure, trial family, and look-elsewhere accounting are not frozen here.
11. **Radial boundary:** an exact drop to zero at `5°` requires bin edges, exposure normalization, completeness masks, uncertainty intervals, and treatment of empty bins.
12. **Independence:** conditioning every catalogue on the same preselected sky windows does not make the resulting tests statistically independent.

## Sheridan deployment requirements

Before a confirmatory run, require the claimant to provide:

- one complete node table with frame, epoch, coordinate system, and all declared primary/secondary nodes;
- one radius or a preregistered radius family with multiplicity correction;
- separate frozen anomaly predicates for NSC, Gaia, and AllWISE;
- complete full-sky or survey-footprint catalogues selected independently of the nodes;
- survey support grids, masks, completeness, and exposure covariates;
- the exact Gaia TAP query;
- target and control definitions fixed before outcome inspection;
- all Monte Carlo code, seeds, null measures, and tested alternatives;
- a holdout or genuinely independent catalogue;
- explicit positive, negative, and forbidden outcomes for each physical interpretation.

Until those fields are supplied, the public material is sufficient for a **claim audit** but not for a valid confirmatory calculation.

## Strategic posture

The ledger is deliberately neutral. It prevents both sides from choosing the most convenient version after seeing results. Every future test must name the exact public claim revision it evaluates, preserve null outcomes, and distinguish catalogue association from causal ontology.
