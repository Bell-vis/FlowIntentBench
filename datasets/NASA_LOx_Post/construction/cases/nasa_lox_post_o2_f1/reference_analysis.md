# NASA_LOx_Post nasa_lox_post_o2_f1 Reference Analysis

This is a dataset-specific construction artifact, not model-facing input.
All candidates were executed on the supplied reader-visible flow field; no physical units are inferred.

## Authored question

Which high-speed flow region is strongest in the LOx-post flow field, and where is it located? Identify contiguous high-speed regions using an appropriate speed-based criterion and use an appropriate measure to compare their strength. Treat locations connected through shared grid faces as one contiguous region, represent the selected region by the mean spatial location of its high-speed locations, and report its location and strength.

## Candidate-O exploration and adjudication

### `o2_q90_peak`

- Dimensions differing from baseline candidate: `['none']`
- Feature: Face-connected locations form one contiguous high-speed flow region.
- Criterion: Retain locations at or above the 90th percentile of non-zero recorded speed values.
- Measure: Use peak speed as the strength of each retained region.
- Representation: Represent the selected region by the arithmetic mean spatial location of its retained high-speed locations.
- Adjudication: **PROVISIONALLY_ACCEPTED** — Provisional: fixed distribution-informed criterion with peak strength.
- Candidate: `o2_q90_peak` (provisionally_accepted)
- Criterion: `>= 90th percentile (1.153866514531)`; retained points: `9994`
- Connected regions: `2`; sizes: `[5064, 4930]`
- Selected region size: `4930`
- Reported location (mean_spatial_location): `[1.9714835118861307, -2.6243898427147894, 3.586038704950959]`
- Measure kind: `peak`
- Strength: `1.391444101423406`

### `o2_q90_mean`

- Dimensions differing from baseline candidate: `['property_measure']`
- Feature: Face-connected locations form one contiguous high-speed flow region.
- Criterion: Retain locations at or above the 90th percentile of non-zero recorded speed values.
- Measure: Use mean speed across retained locations as the strength of each retained region.
- Representation: Represent the selected region by the arithmetic mean spatial location of its retained high-speed locations.
- Adjudication: **PROVISIONALLY_ACCEPTED** — Provisional: same criterion with sustained regional strength.
- Candidate: `o2_q90_mean` (provisionally_accepted)
- Criterion: `>= 90th percentile (1.153866514531)`; retained points: `9994`
- Connected regions: `2`; sizes: `[5064, 4930]`
- Selected region size: `4930`
- Reported location (mean_spatial_location): `[1.9714835118861307, -2.6243898427147894, 3.586038704950959]`
- Measure kind: `mean`
- Strength: `1.2483379152602692`

### `o2_q75_mean`

- Dimensions differing from baseline candidate: `['criterion', 'property_measure']`
- Feature: Face-connected locations form one contiguous high-speed flow region.
- Criterion: Retain locations at or above the 75th percentile of non-zero recorded speed values.
- Measure: Use mean speed across retained locations as the strength of each retained region.
- Representation: Represent the selected region by the arithmetic mean spatial location of its retained high-speed locations.
- Adjudication: **PROVISIONALLY_ACCEPTED** — Provisional: a distinct distribution-informed criterion/measure bundle; grounding remains pending.
- Candidate: `o2_q75_mean` (provisionally_accepted)
- Criterion: `>= 75th percentile (0.821895835782)`; retained points: `24985`
- Connected regions: `1`; sizes: `[24985]`
- Selected region size: `24985`
- Reported location (mean_spatial_location): `[-0.2586535748643758, 0.004317644593928942, 3.18852633498706]`
- Measure kind: `mean`
- Strength: `1.1169056975303706`

### `o2_q90_integrated_excess`

- Dimensions differing from baseline candidate: `['property_measure']`
- Feature: Face-connected locations form one contiguous high-speed flow region.
- Criterion: Retain locations at or above the 90th percentile of non-zero recorded speed values.
- Measure: Use the point-summed speed excess above the selected criterion as region strength.
- Representation: Represent the selected region by the arithmetic mean spatial location of its retained high-speed locations.
- Adjudication: **REJECTED** — Rejected: point-summed strength is sampling-density dependent without a cell-volume contract.
- Candidate: `o2_q90_integrated_excess` (rejected)
- Criterion: `>= 90th percentile (1.153866514531)`; retained points: `9994`
- Connected regions: `2`; sizes: `[5064, 4930]`
- Selected region size: `5064`
- Reported location (mean_spatial_location): `[1.9208217011032682, 2.7226240939768194, 3.5903704281290185]`
- Measure kind: `integrated_excess`
- Strength: `474.00870544623376`

## O2 unresolved-dimension review

Both criterion and property measure were explored using complete candidate bundles; no Cartesian product was generated. The q90 and q75 criteria remain provisional under the same grounding standard.

Scientific Grounding for this dataset-specific operationalization remains PENDING; Formal Release is HOLD / CONDITIONAL.
