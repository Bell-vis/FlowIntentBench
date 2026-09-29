# FireFlow fireflow_o1_f2 Reference Analysis

This is a dataset-specific construction artifact, not model-facing input.
All candidates were executed on the supplied reader-visible flow field; no physical units are inferred.

## Authored question

Which high-speed flow region is strongest in the FireFlow room flow field? For this analysis, define high-speed locations as those at or above the 90th percentile of non-zero recorded speed in the stored velocity scale, treat locations belonging to the same connected mesh neighborhood as one region, use peak speed as the measure of strength, and represent the strongest region by the mean spatial location of its retained high-speed locations. Characterize the strongest high-speed flow region.

## Candidate-O exploration and adjudication

### `o1_high_speed_regions_open`

- Dimensions differing from baseline candidate: `['none']`
- Feature: Mesh-connected locations form one contiguous high-speed flow region.
- Criterion: Retain locations at or above the 90th percentile of non-zero recorded speed values.
- Measure: Use peak speed as the strength of each retained region.
- Representation: Represent the selected region by the arithmetic mean spatial location of its retained high-speed locations.
- Adjudication: **PROVISIONALLY_ACCEPTED** — The authored O1 specification fixes this complete operationalization.
- Candidate: `o1_high_speed_regions_open` (provisionally_accepted)
- Criterion: `>= 90th percentile (0.973593823082)`; retained points: `1233`
- Connected regions: `4`; sizes: `[517, 710, 3, 3]`
- Selected region size: `517`
- Reported location (mean_spatial_location): `[0.5113998996219967, 1.975870378012814, 0.48199224898616855]`
- Measure kind: `peak`
- Strength: `3.822313175748469`

Scientific Grounding for this dataset-specific operationalization remains PENDING; Formal Release is HOLD / CONDITIONAL.
