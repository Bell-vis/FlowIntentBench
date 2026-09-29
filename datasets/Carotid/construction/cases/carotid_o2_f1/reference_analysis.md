# Carotid carotid_o2_f1 Reference Analysis

This is a dataset-specific construction artifact, not model-facing input.
All candidates were executed on the supplied reader-visible flow field; no physical units are inferred.

## Authored question

Which high-speed flow region is strongest in the carotid artery flow field, and where is it located? Identify contiguous high-speed regions using an appropriate speed-based criterion and use an appropriate measure to compare their strength. Treat locations connected through shared grid faces as one contiguous region, represent the selected region by the mean spatial location of its high-speed locations, and report its location and strength.

## Candidate-O exploration and adjudication

### `o2_q90_peak`

- Dimensions differing from baseline candidate: `['none']`
- Feature: Face-connected locations form one contiguous high-speed flow region.
- Criterion: Retain locations at or above the 90th percentile of non-zero recorded speed values.
- Measure: Use peak speed as the strength of each retained region.
- Representation: Represent the selected region by the arithmetic mean spatial location of its retained high-speed locations.
- Adjudication: **PROVISIONALLY_ACCEPTED** — Provisional: fixed distribution-informed criterion with peak strength.
- Candidate: `o2_q90_peak` (provisionally_accepted)
- Criterion: `>= 90th percentile (0.039422780215)`; retained points: `16757`
- Connected regions: `642`; sizes: `[1, 1, 16, 1, 3, 2, 1, 10, 1, 1, 4, 1, 1, 13735, 3, 2, 1, 1, 3, 8, 1, 1, 1, 1, 2, 4, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 6, 2, 1, 1, 2, 1, 4, 2, 1, 9, 26, 2, 1, 2, 2, 1, 1, 1, 2, 1, 1, 1, 11, 3, 1, 2, 1, 1, 3, 3, 1, 1, 2, 1, 1, 1, 1, 7, 1, 1, 4, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 1, 4, 1, 1, 3, 2, 2, 2, 1, 1, 2, 1, 1029, 3, 2, 1, 1, 2, 2, 2, 1, 1, 1, 2, 1, 5, 1, 105, 1, 7, 2, 11, 2, 3, 1, 1, 1, 5, 3, 1, 1, 10, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 1, 1, 1, 1, 1, 3, 307, 1, 2, 47, 1, 1, 1, 1, 2, 1, 1, 1, 3, 16, 6, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 9, 1, 2, 5, 4, 1, 2, 1, 3, 2, 3, 1, 2, 1, 1, 1, 1, 2, 1, 3, 1, 1, 1, 5, 1, 2, 1, 1, 2, 2, 7, 1, 1, 4, 1, 2, 1, 7, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 3, 1, 4, 10, 1, 1, 1, 2, 5, 1, 1, 2, 1, 2, 1, 1, 2, 2, 9, 3, 1, 2, 6, 2, 2, 1, 1, 1, 2, 23, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 1, 3, 12, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 2, 3, 1, 3, 1, 1, 1, 5, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 2, 1, 3, 7, 2, 2, 1, 1, 2, 2, 14, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 4, 2, 2, 3, 6, 1, 1, 1, 2, 1, 4, 7, 1, 3, 4, 1, 6, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 2, 1, 1, 1, 1, 1, 5, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 1, 2, 1, 1, 1, 1, 2, 6, 1, 1, 1, 1, 3, 3, 1, 1, 1, 8, 34, 2, 2, 2, 5, 1, 2, 2, 1, 2, 1, 1, 1, 1, 1, 2, 5, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 2, 3, 2, 1, 2, 1, 1, 6, 1, 1, 6, 1, 2, 1, 2, 1, 1, 1, 3, 1, 1, 1, 1, 1, 2, 1, 2, 3, 8, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 13, 1, 4, 1, 1, 3, 1, 3, 1, 1, 1, 1, 1, 1, 14, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 1, 4, 2, 3, 2, 2, 1, 2, 1, 1, 1, 9, 1, 2, 1, 2, 3, 39, 1, 1, 86, 2, 3, 1, 2, 4, 11, 1, 2, 1, 6, 7, 1, 3, 2, 1, 1, 1, 1, 1, 4, 1, 3, 4, 5, 1, 2, 1, 4, 6, 1, 3, 3, 1, 1, 1, 4, 1, 1, 8, 1, 1, 4, 1, 1, 1, 2, 2, 1, 1, 1, 3, 3, 1, 1, 2, 1, 1, 1, 1, 3, 6, 1, 5, 2, 7, 1, 1, 2, 1, 10, 1, 1, 2, 4, 1, 2, 2, 11, 1, 24, 2, 3, 1, 2, 1, 1, 1, 2, 1, 4, 1, 1, 1, 14, 1, 1, 1, 1]`
- Selected region size: `13735`
- Reported location (mean_spatial_location): `[137.80844557699308, 94.49144521295959, 15.773061521659992]`
- Measure kind: `peak`
- Strength: `22.694927991153424`

### `o2_q90_mean`

- Dimensions differing from baseline candidate: `['property_measure']`
- Feature: Face-connected locations form one contiguous high-speed flow region.
- Criterion: Retain locations at or above the 90th percentile of non-zero recorded speed values.
- Measure: Use mean speed across retained locations as the strength of each retained region.
- Representation: Represent the selected region by the arithmetic mean spatial location of its retained high-speed locations.
- Adjudication: **PROVISIONALLY_ACCEPTED** — Provisional: same criterion with sustained regional strength.
- Candidate: `o2_q90_mean` (provisionally_accepted)
- Criterion: `>= 90th percentile (0.039422780215)`; retained points: `16757`
- Connected regions: `642`; sizes: `[1, 1, 16, 1, 3, 2, 1, 10, 1, 1, 4, 1, 1, 13735, 3, 2, 1, 1, 3, 8, 1, 1, 1, 1, 2, 4, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 6, 2, 1, 1, 2, 1, 4, 2, 1, 9, 26, 2, 1, 2, 2, 1, 1, 1, 2, 1, 1, 1, 11, 3, 1, 2, 1, 1, 3, 3, 1, 1, 2, 1, 1, 1, 1, 7, 1, 1, 4, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 1, 4, 1, 1, 3, 2, 2, 2, 1, 1, 2, 1, 1029, 3, 2, 1, 1, 2, 2, 2, 1, 1, 1, 2, 1, 5, 1, 105, 1, 7, 2, 11, 2, 3, 1, 1, 1, 5, 3, 1, 1, 10, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 1, 1, 1, 1, 1, 3, 307, 1, 2, 47, 1, 1, 1, 1, 2, 1, 1, 1, 3, 16, 6, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 9, 1, 2, 5, 4, 1, 2, 1, 3, 2, 3, 1, 2, 1, 1, 1, 1, 2, 1, 3, 1, 1, 1, 5, 1, 2, 1, 1, 2, 2, 7, 1, 1, 4, 1, 2, 1, 7, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 3, 1, 4, 10, 1, 1, 1, 2, 5, 1, 1, 2, 1, 2, 1, 1, 2, 2, 9, 3, 1, 2, 6, 2, 2, 1, 1, 1, 2, 23, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 1, 3, 12, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 2, 3, 1, 3, 1, 1, 1, 5, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 2, 1, 3, 7, 2, 2, 1, 1, 2, 2, 14, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 4, 2, 2, 3, 6, 1, 1, 1, 2, 1, 4, 7, 1, 3, 4, 1, 6, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 2, 1, 1, 1, 1, 1, 5, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 1, 2, 1, 1, 1, 1, 2, 6, 1, 1, 1, 1, 3, 3, 1, 1, 1, 8, 34, 2, 2, 2, 5, 1, 2, 2, 1, 2, 1, 1, 1, 1, 1, 2, 5, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 2, 3, 2, 1, 2, 1, 1, 6, 1, 1, 6, 1, 2, 1, 2, 1, 1, 1, 3, 1, 1, 1, 1, 1, 2, 1, 2, 3, 8, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 13, 1, 4, 1, 1, 3, 1, 3, 1, 1, 1, 1, 1, 1, 14, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 1, 4, 2, 3, 2, 2, 1, 2, 1, 1, 1, 9, 1, 2, 1, 2, 3, 39, 1, 1, 86, 2, 3, 1, 2, 4, 11, 1, 2, 1, 6, 7, 1, 3, 2, 1, 1, 1, 1, 1, 4, 1, 3, 4, 5, 1, 2, 1, 4, 6, 1, 3, 3, 1, 1, 1, 4, 1, 1, 8, 1, 1, 4, 1, 1, 1, 2, 2, 1, 1, 1, 3, 3, 1, 1, 2, 1, 1, 1, 1, 3, 6, 1, 5, 2, 7, 1, 1, 2, 1, 10, 1, 1, 2, 4, 1, 2, 2, 11, 1, 24, 2, 3, 1, 2, 1, 1, 1, 2, 1, 4, 1, 1, 1, 14, 1, 1, 1, 1]`
- Selected region size: `3`
- Reported location (mean_spatial_location): `[144.66666666666666, 115.33333333333333, 23.0]`
- Measure kind: `mean`
- Strength: `4.273159224483293`

### `o2_q75_mean`

- Dimensions differing from baseline candidate: `['criterion', 'property_measure']`
- Feature: Face-connected locations form one contiguous high-speed flow region.
- Criterion: Retain locations at or above the 75th percentile of non-zero recorded speed values.
- Measure: Use mean speed across retained locations as the strength of each retained region.
- Representation: Represent the selected region by the arithmetic mean spatial location of its retained high-speed locations.
- Adjudication: **PROVISIONALLY_ACCEPTED** — Provisional: a distinct distribution-informed criterion/measure bundle; grounding remains pending.
- Candidate: `o2_q75_mean` (provisionally_accepted)
- Criterion: `>= 75th percentile (0.013181109908)`; retained points: `41891`
- Connected regions: `869`; sizes: `[1, 40396, 1, 3, 1, 3, 1, 2, 1, 1, 7, 1, 1, 1, 3, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 8, 1, 3, 1, 1, 1, 1, 2, 1, 5, 1, 2, 1, 1, 2, 1, 1, 2, 1, 4, 2, 3, 3, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 2, 1, 1, 2, 1, 2, 4, 1, 2, 1, 1, 4, 2, 1, 1, 1, 1, 10, 2, 1, 2, 1, 2, 2, 1, 1, 1, 1, 3, 1, 1, 1, 2, 1, 4, 1, 1, 3, 6, 1, 2, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 3, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 1, 1, 2, 1, 1, 1, 2, 3, 1, 1, 1, 1, 1, 1, 7, 7, 1, 1, 1, 1, 2, 2, 1, 3, 2, 1, 8, 4, 1, 1, 1, 2, 1, 1, 2, 4, 1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 3, 2, 1, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 2, 2, 1, 1, 1, 1, 1, 2, 3, 1, 1, 1, 1, 1, 1, 1, 2, 1, 5, 1, 1, 1, 6, 1, 1, 1, 1, 6, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 3, 1, 2, 1, 3, 2, 1, 7, 6, 1, 1, 1, 1, 1, 1, 1, 6, 1, 3, 1, 1, 6, 1, 1, 3, 2, 1, 1, 6, 1, 15, 1, 7, 1, 1, 1, 2, 1, 3, 1, 2, 4, 1, 1, 1, 3, 4, 3, 1, 3, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 2, 1, 2, 1, 4, 1, 1, 4, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 9, 1, 1, 4, 2, 1, 2, 1, 1, 2, 3, 1, 1, 2, 2, 1, 1, 1, 2, 1, 1, 2, 3, 1, 1, 3, 1, 6, 4, 1, 2, 5, 4, 1, 1, 1, 1, 2, 1, 2, 2, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 9, 1, 1, 1, 1, 5, 4, 1, 1, 1, 1, 1, 2, 1, 4, 3, 3, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 2, 3, 1, 9, 2, 1, 2, 2, 1, 2, 1, 2, 5, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 5, 1, 3, 1, 1, 1, 2, 2, 1, 1, 3, 1, 2, 1, 1, 1, 2, 1, 3, 1, 1, 1, 1, 1, 1, 1, 7, 1, 1, 2, 2, 1, 3, 1, 1, 1, 1, 1, 9, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 2, 3, 1, 4, 2, 1, 1, 1, 1, 4, 3, 1, 1, 7, 1, 2, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 3, 2, 2, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 3, 2, 1, 1, 7, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 2, 1, 3, 1, 1, 3, 1, 1, 1, 2, 1, 1, 2, 1, 1, 3, 1, 1, 1, 2, 1, 4, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 4, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 3, 2, 1, 3, 6, 1, 4, 1, 2, 1, 3, 1, 1, 16, 5, 1, 1, 3, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 4, 1, 1, 2, 1, 6, 1, 1, 1, 12, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 2, 2, 1, 3, 1, 2, 7, 2, 1, 1, 1, 2, 4, 1, 2, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 3, 1, 1, 1, 1, 1, 4, 3, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 4, 1, 1, 2, 1, 1, 1, 1, 2, 4, 1, 3, 1, 5, 2, 2, 2, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 2, 2, 1, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 7, 4, 3, 2, 3, 4, 3, 1, 1, 1, 3, 1, 1, 2, 1, 1, 1, 2, 1, 3, 2, 2, 1, 1, 1, 1, 1, 2, 1, 3, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 2, 1, 1, 2, 1, 2, 3, 6, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1]`
- Selected region size: `40396`
- Reported location (mean_spatial_location): `[138.10699079116745, 97.91917516585801, 14.491929894048916]`
- Measure kind: `mean`
- Strength: `0.6786588359876302`

### `o2_q90_integrated_excess`

- Dimensions differing from baseline candidate: `['property_measure']`
- Feature: Face-connected locations form one contiguous high-speed flow region.
- Criterion: Retain locations at or above the 90th percentile of non-zero recorded speed values.
- Measure: Use the point-summed speed excess above the selected criterion as region strength.
- Representation: Represent the selected region by the arithmetic mean spatial location of its retained high-speed locations.
- Adjudication: **REJECTED** — Rejected: point-summed strength is sampling-density dependent without a cell-volume contract.
- Candidate: `o2_q90_integrated_excess` (rejected)
- Criterion: `>= 90th percentile (0.039422780215)`; retained points: `16757`
- Connected regions: `642`; sizes: `[1, 1, 16, 1, 3, 2, 1, 10, 1, 1, 4, 1, 1, 13735, 3, 2, 1, 1, 3, 8, 1, 1, 1, 1, 2, 4, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 6, 2, 1, 1, 2, 1, 4, 2, 1, 9, 26, 2, 1, 2, 2, 1, 1, 1, 2, 1, 1, 1, 11, 3, 1, 2, 1, 1, 3, 3, 1, 1, 2, 1, 1, 1, 1, 7, 1, 1, 4, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 1, 4, 1, 1, 3, 2, 2, 2, 1, 1, 2, 1, 1029, 3, 2, 1, 1, 2, 2, 2, 1, 1, 1, 2, 1, 5, 1, 105, 1, 7, 2, 11, 2, 3, 1, 1, 1, 5, 3, 1, 1, 10, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 1, 1, 1, 1, 1, 3, 307, 1, 2, 47, 1, 1, 1, 1, 2, 1, 1, 1, 3, 16, 6, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 9, 1, 2, 5, 4, 1, 2, 1, 3, 2, 3, 1, 2, 1, 1, 1, 1, 2, 1, 3, 1, 1, 1, 5, 1, 2, 1, 1, 2, 2, 7, 1, 1, 4, 1, 2, 1, 7, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 3, 1, 4, 10, 1, 1, 1, 2, 5, 1, 1, 2, 1, 2, 1, 1, 2, 2, 9, 3, 1, 2, 6, 2, 2, 1, 1, 1, 2, 23, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 1, 3, 12, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 2, 3, 1, 3, 1, 1, 1, 5, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 2, 1, 3, 7, 2, 2, 1, 1, 2, 2, 14, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 4, 2, 2, 3, 6, 1, 1, 1, 2, 1, 4, 7, 1, 3, 4, 1, 6, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 2, 1, 1, 1, 1, 1, 5, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 1, 2, 1, 1, 1, 1, 2, 6, 1, 1, 1, 1, 3, 3, 1, 1, 1, 8, 34, 2, 2, 2, 5, 1, 2, 2, 1, 2, 1, 1, 1, 1, 1, 2, 5, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 2, 3, 2, 1, 2, 1, 1, 6, 1, 1, 6, 1, 2, 1, 2, 1, 1, 1, 3, 1, 1, 1, 1, 1, 2, 1, 2, 3, 8, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 13, 1, 4, 1, 1, 3, 1, 3, 1, 1, 1, 1, 1, 1, 14, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 1, 4, 2, 3, 2, 2, 1, 2, 1, 1, 1, 9, 1, 2, 1, 2, 3, 39, 1, 1, 86, 2, 3, 1, 2, 4, 11, 1, 2, 1, 6, 7, 1, 3, 2, 1, 1, 1, 1, 1, 4, 1, 3, 4, 5, 1, 2, 1, 4, 6, 1, 3, 3, 1, 1, 1, 4, 1, 1, 8, 1, 1, 4, 1, 1, 1, 2, 2, 1, 1, 1, 3, 3, 1, 1, 2, 1, 1, 1, 1, 3, 6, 1, 5, 2, 7, 1, 1, 2, 1, 10, 1, 1, 2, 4, 1, 2, 2, 11, 1, 24, 2, 3, 1, 2, 1, 1, 1, 2, 1, 4, 1, 1, 1, 14, 1, 1, 1, 1]`
- Selected region size: `13735`
- Reported location (mean_spatial_location): `[137.80844557699308, 94.49144521295959, 15.773061521659992]`
- Measure kind: `integrated_excess`
- Strength: `25936.465010726788`

## O2 unresolved-dimension review

Both criterion and property measure were explored using complete candidate bundles; no Cartesian product was generated. The q90 and q75 criteria remain provisional under the same grounding standard.

Scientific Grounding for this dataset-specific operationalization remains PENDING; Formal Release is HOLD / CONDITIONAL.
