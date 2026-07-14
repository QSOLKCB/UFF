# Rotation-curve data contract

UFF v4 accepts comma-separated files with one row per radius. Rows may appear
in any order; the loader sorts them by radius after validation.

## Canonical columns

| Meaning | Canonical name | Accepted aliases | Unit | Required |
|---|---|---|---|---|
| Galactocentric radius | `R_kpc` | `Rad`, `radius_kpc`, `radius`, `R` | kpc | Yes |
| Observed circular speed | `V_obs_kms` | `Vobs`, `velocity_obs_kms`, `V_obs` | km/s | Yes |
| 1σ speed uncertainty | `e_V_kms` | `errV`, `velocity_err_kms`, `eV` | km/s | Yes |
| Gas reference contribution | `V_gas_kms` | `Vgas`, `velocity_gas_kms`, `V_gas` | km/s | No |
| Disk reference contribution | `V_disk_kms` | `Vdisk`, `velocity_disk_kms`, `V_disk` | km/s | No |
| Bulge reference contribution | `V_bul_kms` | `Vbul`, `velocity_bulge_kms`, `V_bul` | km/s | No |

Missing optional components are filled with zeros and recorded in the JSON
warning list. MOND/RAR and baryon-sensitive fits are generally not meaningful
without the relevant mass components.

## Validation

The loader rejects:

- fewer than three rows;
- missing required columns;
- non-numeric, NaN, or infinite values;
- non-positive or duplicate radii;
- negative observed circular speeds; and
- non-positive uncertainties.

## Signed gas

Do not pre-square the gas column. Some SPARC inner radii have negative `Vgas`.
UFF preserves it through

```text
Vgas_squared_contribution = Vgas * abs(Vgas)
```

The total baryonic `V²` is floored at zero only after the signed gas and stellar
terms are combined.

## Scalar metadata

Any extra column containing one repeated scalar value is copied into the JSON
metadata object. Common examples are:

| Column | Meaning |
|---|---|
| `GALNAME` | Galaxy identifier |
| `DIST_Mpc` | Reference distance |
| `INC_deg` | Reference inclination |
| `ML_disk`, `ML_bulge` | Catalog/reference mass-to-light values |

`--fit-inclination` reads `INC_deg` automatically. If it is absent, provide
`--inclination-deg`.

## Example

```csv
R_kpc,V_obs_kms,e_V_kms,V_gas_kms,V_disk_kms,V_bul_kms,GALNAME,DIST_Mpc,INC_deg
0.5,40.0,5.0,5.0,20.0,10.0,DEMO,3.2,62.0
1.0,65.0,5.0,10.0,40.0,12.0,DEMO,3.2,62.0
2.0,90.0,4.0,15.0,60.0,14.0,DEMO,3.2,62.0
```

## Provenance

Each run records the exact input path and SHA-256 digest. Preserve the original
source citation and quality flags separately; a file hash proves byte identity,
not observational validity.

Authoritative original data: [SPARC](https://astroweb.case.edu/SPARC/).
