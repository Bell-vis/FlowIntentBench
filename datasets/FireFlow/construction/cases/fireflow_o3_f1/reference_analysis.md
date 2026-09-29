# FireFlow fireflow_o3_f1 Reference Analysis

This is a dataset-specific construction artifact, not model-facing input.
All candidates were executed on the supplied reader-visible flow field; no physical units are inferred.

## Authored question

Which flow region in the FireFlow room flow field is strongest, and where is it located? Identify and characterize the strongest region using an appropriate scientific analysis, and report its location and strength.

## Candidate-O exploration and adjudication

### `o3_q90_peak`

- Dimensions differing from baseline candidate: `['none']`
- Feature: Mesh-connected locations form one contiguous high-speed flow region.
- Criterion: Retain locations at or above the 90th percentile of non-zero recorded speed values.
- Measure: Use peak speed as the strength of each retained region.
- Representation: Represent the selected region by the arithmetic mean spatial location of its retained high-speed locations.
- Adjudication: **PROVISIONALLY_ACCEPTED** — Provisional complete O with distribution-informed criterion and peak strength.
- Candidate: `o3_q90_peak` (provisionally_accepted)
- Criterion: `>= 90th percentile (0.973593823082)`; retained points: `1233`
- Connected regions: `4`; sizes: `[517, 710, 3, 3]`
- Selected region size: `517`
- Reported location (mean_spatial_location): `[0.5113998996219967, 1.975870378012814, 0.48199224898616855]`
- Measure kind: `peak`
- Strength: `3.822313175748469`

### `o3_q90_mean`

- Dimensions differing from baseline candidate: `['property_measure']`
- Feature: Mesh-connected locations form one contiguous high-speed flow region.
- Criterion: Retain locations at or above the 90th percentile of non-zero recorded speed values.
- Measure: Use mean speed across retained locations as the strength of each retained region.
- Representation: Represent the selected region by the arithmetic mean spatial location of its retained high-speed locations.
- Adjudication: **PROVISIONALLY_ACCEPTED** — Provisional complete O with sustained regional strength.
- Candidate: `o3_q90_mean` (provisionally_accepted)
- Criterion: `>= 90th percentile (0.973593823082)`; retained points: `1233`
- Connected regions: `4`; sizes: `[517, 710, 3, 3]`
- Selected region size: `517`
- Reported location (mean_spatial_location): `[0.5113998996219967, 1.975870378012814, 0.48199224898616855]`
- Measure kind: `mean`
- Strength: `1.6380479260626386`

### `o3_q75_mean`

- Dimensions differing from baseline candidate: `['criterion', 'property_measure']`
- Feature: Mesh-connected locations form one contiguous high-speed flow region.
- Criterion: Retain locations at or above the 75th percentile of non-zero recorded speed values.
- Measure: Use mean speed across retained locations as the strength of each retained region.
- Representation: Represent the selected region by the arithmetic mean spatial location of its retained high-speed locations.
- Adjudication: **PROVISIONALLY_ACCEPTED** — Provisional complete O with a distinct criterion and measure.
- Candidate: `o3_q75_mean` (provisionally_accepted)
- Criterion: `>= 75th percentile (0.486353495383)`; retained points: `3081`
- Connected regions: `1`; sizes: `[3081]`
- Selected region size: `3081`
- Reported location (mean_spatial_location): `[2.417761593795914, 1.6560045202195897, 1.0163907527207787]`
- Measure kind: `mean`
- Strength: `1.0174865547746001`

### `o3_q90_peak_point`

- Dimensions differing from baseline candidate: `['aggregation_or_representation']`
- Feature: Mesh-connected locations form one contiguous high-speed flow region.
- Criterion: Retain locations at or above the 90th percentile of non-zero recorded speed values.
- Measure: Use peak speed as the strength of each retained region.
- Representation: Represent the selected region by the spatial location of its peak-speed location.
- Adjudication: **PROVISIONALLY_ACCEPTED** — Provisional complete O with peak-point representation.
- Candidate: `o3_q90_peak_point` (provisionally_accepted)
- Criterion: `>= 90th percentile (0.973593823082)`; retained points: `1233`
- Connected regions: `4`; sizes: `[517, 710, 3, 3]`
- Selected region size: `517`
- Reported location (peak_speed_location): `[0.19999998807907104, 1.428571343421936, 0.19999998807907104]`
- Measure kind: `peak`
- Strength: `3.822313175748469`

### `o3_local_peak_features`

- Dimensions differing from baseline candidate: `['feature_definition', 'criterion', 'aggregation_or_representation']`
- Feature: Treat isolated local speed maxima above the selected cutoff as point features rather than connected flow regions.
- Criterion: Retain speed locations above the selected cutoff that are strictly greater than their mesh neighbors.
- Measure: Use peak speed as the strength of each retained region.
- Representation: Represent each feature by the spatial location of its local maximum.
- Adjudication: **REJECTED** — Rejected: isolated point features do not satisfy the question's target entity of a flow region.
- Candidate: `o3_local_peak_features` (rejected)
- Criterion: `>= selected cutoff (0.000000000000)`; retained points: `26`
- Connected regions: `26`; sizes: `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]`
- Selected region size: `1`
- Reported location (mean_spatial_location): `[0.19999998807907104, 1.428571343421936, 0.19999998807907104]`
- Measure kind: `peak`
- Strength: `3.822313175748469`

## O3 breadth review

Criterion, measure, representation and feature-definition candidates were considered. The isolated local-peak candidate was rejected because it represents point features rather than the flow region requested by the question.

Scientific Grounding for this dataset-specific operationalization remains PENDING; Formal Release is HOLD / CONDITIONAL.
