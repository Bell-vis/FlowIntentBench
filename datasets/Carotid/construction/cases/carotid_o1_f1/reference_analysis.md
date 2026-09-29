# Carotid carotid_o1_f1 Reference Analysis

This is a dataset-specific construction artifact, not model-facing input.
All candidates were executed on the supplied reader-visible flow field; no physical units are inferred.

## Authored question

Which high-speed flow region is strongest in the carotid artery flow field, and where is it located? For this analysis, compute speed as the Euclidean magnitude of the stored point-data vector array named `vectors`, then define high-speed locations as those at or above the 90th percentile of its non-zero values, treat locations connected through shared grid faces as one contiguous region, use peak speed as the measure of strength, and represent the strongest region by the mean spatial location of its retained high-speed locations. Identify the strongest region and report its location and strength.

## Candidate-O exploration and adjudication

### `o1_high_speed_regions`

- Dimensions differing from baseline candidate: `['none']`
- Feature: Face-connected locations form one contiguous high-speed flow region.
- Criterion: Retain locations at or above the 90th percentile of non-zero speed values computed as the Euclidean magnitude of the stored point-data vector array named `vectors`.
- Measure: Use peak speed as the strength of each retained region.
- Representation: Represent the selected region by the arithmetic mean spatial location of its retained high-speed locations.
- Adjudication: **PROVISIONALLY_ACCEPTED** — The authored O1 specification fixes this complete operationalization.
- Candidate: `o1_high_speed_regions` (provisionally_accepted)
- Criterion: `>= 90th percentile (0.039422780215)`; retained points: `16757`
- Connected regions: `642`; sizes: `[1, 1, 16, 1, 3, 2, 1, 10, 1, 1, 4, 1, 1, 13735, 3, 2, 1, 1, 3, 8, 1, 1, 1, 1, 2, 4, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 6, 2, 1, 1, 2, 1, 4, 2, 1, 9, 26, 2, 1, 2, 2, 1, 1, 1, 2, 1, 1, 1, 11, 3, 1, 2, 1, 1, 3, 3, 1, 1, 2, 1, 1, 1, 1, 7, 1, 1, 4, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 1, 4, 1, 1, 3, 2, 2, 2, 1, 1, 2, 1, 1029, 3, 2, 1, 1, 2, 2, 2, 1, 1, 1, 2, 1, 5, 1, 105, 1, 7, 2, 11, 2, 3, 1, 1, 1, 5, 3, 1, 1, 10, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 1, 1, 1, 1, 1, 3, 307, 1, 2, 47, 1, 1, 1, 1, 2, 1, 1, 1, 3, 16, 6, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 9, 1, 2, 5, 4, 1, 2, 1, 3, 2, 3, 1, 2, 1, 1, 1, 1, 2, 1, 3, 1, 1, 1, 5, 1, 2, 1, 1, 2, 2, 7, 1, 1, 4, 1, 2, 1, 7, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 3, 1, 4, 10, 1, 1, 1, 2, 5, 1, 1, 2, 1, 2, 1, 1, 2, 2, 9, 3, 1, 2, 6, 2, 2, 1, 1, 1, 2, 23, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 1, 3, 12, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 2, 3, 1, 3, 1, 1, 1, 5, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 2, 1, 3, 7, 2, 2, 1, 1, 2, 2, 14, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 4, 2, 2, 3, 6, 1, 1, 1, 2, 1, 4, 7, 1, 3, 4, 1, 6, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 2, 1, 1, 1, 1, 1, 5, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 1, 2, 1, 1, 1, 1, 2, 6, 1, 1, 1, 1, 3, 3, 1, 1, 1, 8, 34, 2, 2, 2, 5, 1, 2, 2, 1, 2, 1, 1, 1, 1, 1, 2, 5, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 2, 3, 2, 1, 2, 1, 1, 6, 1, 1, 6, 1, 2, 1, 2, 1, 1, 1, 3, 1, 1, 1, 1, 1, 2, 1, 2, 3, 8, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 13, 1, 4, 1, 1, 3, 1, 3, 1, 1, 1, 1, 1, 1, 14, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 1, 4, 2, 3, 2, 2, 1, 2, 1, 1, 1, 9, 1, 2, 1, 2, 3, 39, 1, 1, 86, 2, 3, 1, 2, 4, 11, 1, 2, 1, 6, 7, 1, 3, 2, 1, 1, 1, 1, 1, 4, 1, 3, 4, 5, 1, 2, 1, 4, 6, 1, 3, 3, 1, 1, 1, 4, 1, 1, 8, 1, 1, 4, 1, 1, 1, 2, 2, 1, 1, 1, 3, 3, 1, 1, 2, 1, 1, 1, 1, 3, 6, 1, 5, 2, 7, 1, 1, 2, 1, 10, 1, 1, 2, 4, 1, 2, 2, 11, 1, 24, 2, 3, 1, 2, 1, 1, 1, 2, 1, 4, 1, 1, 1, 14, 1, 1, 1, 1]`
- Selected region size: `13735`
- Reported location (mean_spatial_location): `[137.80844557699308, 94.49144521295959, 15.773061521659992]`
- Measure kind: `peak`
- Strength: `22.694927991153424`

Scientific Grounding for this dataset-specific operationalization remains PENDING; Formal Release is HOLD / CONDITIONAL.
