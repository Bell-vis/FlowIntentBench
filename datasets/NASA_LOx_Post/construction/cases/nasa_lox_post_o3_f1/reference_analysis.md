# NASA_LOx_Post nasa_lox_post_o3_f1 Reference Analysis

This is a dataset-specific construction artifact, not model-facing input.
All candidates were executed on the supplied reader-visible flow field; no physical units are inferred.

## Authored question

Which flow region in the LOx-post flow field is strongest, and where is it located? Identify and characterize the strongest region using an appropriate scientific analysis, and report its location and strength.

## Candidate-O exploration and adjudication

### `o3_q90_peak`

- Dimensions differing from baseline candidate: `['none']`
- Feature: Face-connected locations form one contiguous high-speed flow region.
- Criterion: Retain locations at or above the 90th percentile of non-zero recorded speed values.
- Measure: Use peak speed as the strength of each retained region.
- Representation: Represent the selected region by the arithmetic mean spatial location of its retained high-speed locations.
- Adjudication: **PROVISIONALLY_ACCEPTED** — Provisional complete O with distribution-informed criterion and peak strength.
- Candidate: `o3_q90_peak` (provisionally_accepted)
- Criterion: `>= 90th percentile (1.153866514531)`; retained points: `9994`
- Connected regions: `2`; sizes: `[5064, 4930]`
- Selected region size: `4930`
- Reported location (mean_spatial_location): `[1.9714835118861307, -2.6243898427147894, 3.586038704950959]`
- Measure kind: `peak`
- Strength: `1.391444101423406`

### `o3_q90_mean`

- Dimensions differing from baseline candidate: `['property_measure']`
- Feature: Face-connected locations form one contiguous high-speed flow region.
- Criterion: Retain locations at or above the 90th percentile of non-zero recorded speed values.
- Measure: Use mean speed across retained locations as the strength of each retained region.
- Representation: Represent the selected region by the arithmetic mean spatial location of its retained high-speed locations.
- Adjudication: **PROVISIONALLY_ACCEPTED** — Provisional complete O with sustained regional strength.
- Candidate: `o3_q90_mean` (provisionally_accepted)
- Criterion: `>= 90th percentile (1.153866514531)`; retained points: `9994`
- Connected regions: `2`; sizes: `[5064, 4930]`
- Selected region size: `4930`
- Reported location (mean_spatial_location): `[1.9714835118861307, -2.6243898427147894, 3.586038704950959]`
- Measure kind: `mean`
- Strength: `1.2483379152602692`

### `o3_q75_mean`

- Dimensions differing from baseline candidate: `['criterion', 'property_measure']`
- Feature: Face-connected locations form one contiguous high-speed flow region.
- Criterion: Retain locations at or above the 75th percentile of non-zero recorded speed values.
- Measure: Use mean speed across retained locations as the strength of each retained region.
- Representation: Represent the selected region by the arithmetic mean spatial location of its retained high-speed locations.
- Adjudication: **PROVISIONALLY_ACCEPTED** — Provisional complete O with a distinct criterion and measure.
- Candidate: `o3_q75_mean` (provisionally_accepted)
- Criterion: `>= 75th percentile (0.821895835782)`; retained points: `24985`
- Connected regions: `1`; sizes: `[24985]`
- Selected region size: `24985`
- Reported location (mean_spatial_location): `[-0.2586535748643758, 0.004317644593928942, 3.18852633498706]`
- Measure kind: `mean`
- Strength: `1.1169056975303706`

### `o3_q90_peak_point`

- Dimensions differing from baseline candidate: `['aggregation_or_representation']`
- Feature: Face-connected locations form one contiguous high-speed flow region.
- Criterion: Retain locations at or above the 90th percentile of non-zero recorded speed values.
- Measure: Use peak speed as the strength of each retained region.
- Representation: Represent the selected region by the spatial location of its peak-speed location.
- Adjudication: **PROVISIONALLY_ACCEPTED** — Provisional complete O with peak-point representation.
- Candidate: `o3_q90_peak_point` (provisionally_accepted)
- Criterion: `>= 90th percentile (1.153866514531)`; retained points: `9994`
- Connected regions: `2`; sizes: `[5064, 4930]`
- Selected region size: `4930`
- Reported location (peak_speed_location): `[-0.016844864934682846, -0.8041654229164124, 1.7766473293304443]`
- Measure kind: `peak`
- Strength: `1.391444101423406`

### `o3_local_peak_features`

- Dimensions differing from baseline candidate: `['feature_definition', 'criterion', 'aggregation_or_representation']`
- Feature: Treat isolated local speed maxima above the selected cutoff as point features rather than connected flow regions.
- Criterion: Retain speed locations above the selected cutoff that are strictly greater than their mesh neighbors.
- Measure: Use peak speed as the strength of each retained region.
- Representation: Represent each feature by the spatial location of its local maximum.
- Adjudication: **REJECTED** — Rejected: isolated point features do not satisfy the question's target entity of a flow region.
- Candidate: `o3_local_peak_features` (rejected)
- Criterion: `>= selected cutoff (0.000000000000)`; retained points: `36`
- Connected regions: `36`; sizes: `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]`
- Selected region size: `1`
- Reported location (mean_spatial_location): `[-0.016844864934682846, -0.8041654229164124, 1.7766473293304443]`
- Measure kind: `peak`
- Strength: `1.391444101423406`

## O3 breadth review

Criterion, measure, representation and feature-definition candidates were considered. The isolated local-peak candidate was rejected because it represents point features rather than the flow region requested by the question.

Scientific Grounding for this dataset-specific operationalization remains PENDING; Formal Release is HOLD / CONDITIONAL.
