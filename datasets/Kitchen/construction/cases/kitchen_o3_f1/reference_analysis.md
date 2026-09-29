# Kitchen kitchen_o3_f1 Reference Analysis

This is a dataset-specific construction artifact, not model-facing input.
All candidates were executed on the supplied reader-visible flow field; no physical units are inferred.

## Authored question

Which flow region in the kitchen airflow field is strongest, and where is it located? Identify and characterize the strongest region using an appropriate scientific analysis, and report its location and strength.

## Candidate-O exploration and adjudication

### `o3_q90_peak`

- Dimensions differing from baseline candidate: `['none']`
- Feature: Face-connected locations form one contiguous high-speed flow region.
- Criterion: Retain locations at or above the 90th percentile of non-zero recorded speed values.
- Measure: Use peak speed as the strength of each retained region.
- Representation: Represent the selected region by the arithmetic mean spatial location of its retained high-speed locations.
- Adjudication: **PROVISIONALLY_ACCEPTED** — Provisional complete O with distribution-informed criterion and peak strength.
- Candidate: `o3_q90_peak` (provisionally_accepted)
- Criterion: `>= 90th percentile (0.109679118156)`; retained points: `1130`
- Connected regions: `9`; sizes: `[214, 278, 93, 466, 39, 1, 5, 31, 3]`
- Selected region size: `278`
- Reported location (mean_spatial_location): `[0.4474820164132783, 4.69920858719366, 0.6970863330626874]`
- Measure kind: `peak`
- Strength: `0.45226635527610837`

### `o3_q90_mean`

- Dimensions differing from baseline candidate: `['property_measure']`
- Feature: Face-connected locations form one contiguous high-speed flow region.
- Criterion: Retain locations at or above the 90th percentile of non-zero recorded speed values.
- Measure: Use mean speed across retained locations as the strength of each retained region.
- Representation: Represent the selected region by the arithmetic mean spatial location of its retained high-speed locations.
- Adjudication: **PROVISIONALLY_ACCEPTED** — Provisional complete O with sustained regional strength.
- Candidate: `o3_q90_mean` (provisionally_accepted)
- Criterion: `>= 90th percentile (0.109679118156)`; retained points: `1130`
- Connected regions: `9`; sizes: `[214, 278, 93, 466, 39, 1, 5, 31, 3]`
- Selected region size: `93`
- Reported location (mean_spatial_location): `[6.447096727227652, 4.213978562303769, 0.2433333361060709]`
- Measure kind: `mean`
- Strength: `0.23879296346952839`

### `o3_q75_mean`

- Dimensions differing from baseline candidate: `['criterion', 'property_measure']`
- Feature: Face-connected locations form one contiguous high-speed flow region.
- Criterion: Retain locations at or above the 75th percentile of non-zero recorded speed values.
- Measure: Use mean speed across retained locations as the strength of each retained region.
- Representation: Represent the selected region by the arithmetic mean spatial location of its retained high-speed locations.
- Adjudication: **PROVISIONALLY_ACCEPTED** — Provisional complete O with a distinct criterion and measure.
- Candidate: `o3_q75_mean` (provisionally_accepted)
- Criterion: `>= 75th percentile (0.067245612765)`; retained points: `2824`
- Connected regions: `5`; sizes: `[2814, 2, 1, 1, 6]`
- Selected region size: `2814`
- Reported location (mean_spatial_location): `[3.632793141895668, 3.2369260641599937, 1.425760498289919]`
- Measure kind: `mean`
- Strength: `0.1204053141440439`

### `o3_q90_peak_point`

- Dimensions differing from baseline candidate: `['aggregation_or_representation']`
- Feature: Face-connected locations form one contiguous high-speed flow region.
- Criterion: Retain locations at or above the 90th percentile of non-zero recorded speed values.
- Measure: Use peak speed as the strength of each retained region.
- Representation: Represent the selected region by the spatial location of its peak-speed location.
- Adjudication: **PROVISIONALLY_ACCEPTED** — Provisional complete O with peak-point representation.
- Candidate: `o3_q90_peak_point` (provisionally_accepted)
- Criterion: `>= 90th percentile (0.109679118156)`; retained points: `1130`
- Connected regions: `9`; sizes: `[214, 278, 93, 466, 39, 1, 5, 31, 3]`
- Selected region size: `278`
- Reported location (peak_speed_location): `[0.10000000149011612, 4.5, 0.30000001192092896]`
- Measure kind: `peak`
- Strength: `0.45226635527610837`

### `o3_local_peak_features`

- Dimensions differing from baseline candidate: `['feature_definition', 'criterion', 'aggregation_or_representation']`
- Feature: Treat isolated local speed maxima above the selected cutoff as point features rather than connected flow regions.
- Criterion: Retain speed locations above the selected cutoff that are strictly greater than their mesh neighbors.
- Measure: Use peak speed as the strength of each retained region.
- Representation: Represent each feature by the spatial location of its local maximum.
- Adjudication: **REJECTED** — Rejected: isolated point features do not satisfy the question's target entity of a flow region.
- Candidate: `o3_local_peak_features` (rejected)
- Criterion: `>= selected cutoff (0.000000000000)`; retained points: `118`
- Connected regions: `118`; sizes: `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]`
- Selected region size: `1`
- Reported location (mean_spatial_location): `[0.10000000149011612, 4.5, 0.30000001192092896]`
- Measure kind: `peak`
- Strength: `0.45226635527610837`

## O3 breadth review

Criterion, measure, representation and feature-definition candidates were considered. The isolated local-peak candidate was rejected because it represents point features rather than the flow region requested by the question.

Scientific Grounding for this dataset-specific operationalization remains PENDING; Formal Release is HOLD / CONDITIONAL.
