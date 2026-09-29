# Blunt_Fin blunt_fin_o1_f1 Reference Analysis

This is a dataset-specific construction artifact, not model-facing input.
All candidates were executed on the supplied reader-visible flow field; no physical units are inferred.

## Authored question

Which high-speed flow region is strongest in the blunt-fin flow field, and where is it located? For this analysis, define high-speed locations as those at or above the 90th percentile of non-zero recorded speed in the stored velocity scale, treat locations connected through shared grid faces as one contiguous region, use peak speed as the measure of strength, and represent the strongest region by the mean spatial location of its retained high-speed locations. Identify the strongest region and report its location and strength.

## Candidate-O exploration and adjudication

### `o1_high_speed_regions`

- Dimensions differing from baseline candidate: `['none']`
- Feature: Face-connected locations form one contiguous high-speed flow region.
- Criterion: Retain locations at or above the 90th percentile of non-zero recorded speed values.
- Measure: Use peak speed as the strength of each retained region.
- Representation: Represent the selected region by the arithmetic mean spatial location of its retained high-speed locations.
- Adjudication: **PROVISIONALLY_ACCEPTED** — The authored O1 specification fixes this complete operationalization.
- Candidate: `o1_high_speed_regions` (provisionally_accepted)
- Criterion: `>= 90th percentile (2.776219898197)`; retained points: `3844`
- Connected regions: `6`; sizes: `[3831, 1, 2, 1, 7, 2]`
- Selected region size: `3831`
- Reported location (mean_spatial_location): `[-0.43620374257338634, 3.0521042773374156, 2.5893470661131714]`
- Measure kind: `peak`
- Strength: `3.1229319383189305`

Scientific Grounding for this dataset-specific operationalization remains PENDING; Formal Release is HOLD / CONDITIONAL.
