# NASA_LOx_Post nasa_lox_post_o1_f2 Reference Analysis

This is a dataset-specific construction artifact, not model-facing input.
All candidates were executed on the supplied reader-visible flow field; no physical units are inferred.

## Authored question

Which high-speed flow region is strongest in the LOx-post flow field? For this analysis, define high-speed locations as those at or above the 90th percentile of non-zero recorded speed in the stored velocity scale, treat locations connected through shared grid faces as one contiguous region, use peak speed as the measure of strength, and represent the strongest region by the mean spatial location of its retained high-speed locations. Characterize the strongest high-speed flow region.

## Candidate-O exploration and adjudication

### `o1_high_speed_regions_open`

- Dimensions differing from baseline candidate: `['none']`
- Feature: Face-connected locations form one contiguous high-speed flow region.
- Criterion: Retain locations at or above the 90th percentile of non-zero recorded speed values.
- Measure: Use peak speed as the strength of each retained region.
- Representation: Represent the selected region by the arithmetic mean spatial location of its retained high-speed locations.
- Adjudication: **PROVISIONALLY_ACCEPTED** — The authored O1 specification fixes this complete operationalization.
- Candidate: `o1_high_speed_regions_open` (provisionally_accepted)
- Criterion: `>= 90th percentile (1.153866514531)`; retained points: `9994`
- Connected regions: `2`; sizes: `[5064, 4930]`
- Selected region size: `4930`
- Reported location (mean_spatial_location): `[1.9714835118861307, -2.6243898427147894, 3.586038704950959]`
- Measure kind: `peak`
- Strength: `1.391444101423406`

Scientific Grounding for this dataset-specific operationalization remains PENDING; Formal Release is HOLD / CONDITIONAL.
