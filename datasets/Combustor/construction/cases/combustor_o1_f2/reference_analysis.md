# Combustor combustor_o1_f2 Reference Analysis

This is a dataset-specific construction artifact, not model-facing input.
All candidates were executed on the supplied reader-visible flow field; no physical units are inferred.

## Authored question

Which high-speed flow region is strongest in the annular-combustor flow field? For this analysis, define high-speed locations as those at or above the 90th percentile of non-zero recorded speed in the stored velocity scale, treat locations connected through shared grid faces as one contiguous region, use peak speed as the measure of strength, and represent the strongest region by the mean spatial location of its retained high-speed locations. Characterize the strongest high-speed flow region.

## Candidate-O exploration and adjudication

### `o1_high_speed_regions_open`

- Dimensions differing from baseline candidate: `['none']`
- Feature: Face-connected locations form one contiguous high-speed flow region.
- Criterion: Retain locations at or above the 90th percentile of non-zero recorded speed values.
- Measure: Use peak speed as the strength of each retained region.
- Representation: Represent the selected region by the arithmetic mean spatial location of its retained high-speed locations.
- Adjudication: **PROVISIONALLY_ACCEPTED** — The authored O1 specification fixes this complete operationalization.
- Candidate: `o1_high_speed_regions_open` (provisionally_accepted)
- Criterion: `>= 90th percentile (1135.982844645889)`; retained points: `4379`
- Connected regions: `4`; sizes: `[54, 116, 2725, 1484]`
- Selected region size: `2725`
- Reported location (mean_spatial_location): `[13.673052853925512, 2.427423816624038, 32.3577929498515]`
- Measure kind: `peak`
- Strength: `1528.9350173010403`

Scientific Grounding for this dataset-specific operationalization remains PENDING; Formal Release is HOLD / CONDITIONAL.
