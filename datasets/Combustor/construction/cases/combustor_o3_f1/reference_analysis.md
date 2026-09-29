# Combustor combustor_o3_f1 Reference Analysis

This is a dataset-specific construction artifact, not model-facing input.
All candidates were executed on the supplied reader-visible flow field; no physical units are inferred.

## Authored question

Which flow region in the annular-combustor flow field is strongest, and where is it located? Identify and characterize the strongest region using an appropriate scientific analysis, and report its location and strength.

## Candidate-O exploration and adjudication

### `o3_q90_peak`

- Dimensions differing from baseline candidate: `['none']`
- Feature: Face-connected locations form one contiguous high-speed flow region.
- Criterion: Retain locations at or above the 90th percentile of non-zero recorded speed values.
- Measure: Use peak speed as the strength of each retained region.
- Representation: Represent the selected region by the arithmetic mean spatial location of its retained high-speed locations.
- Adjudication: **PROVISIONALLY_ACCEPTED** — Provisional complete O with distribution-informed criterion and peak strength.
- Candidate: `o3_q90_peak` (provisionally_accepted)
- Criterion: `>= 90th percentile (1135.982844645889)`; retained points: `4379`
- Connected regions: `4`; sizes: `[54, 116, 2725, 1484]`
- Selected region size: `2725`
- Reported location (mean_spatial_location): `[13.673052853925512, 2.427423816624038, 32.3577929498515]`
- Measure kind: `peak`
- Strength: `1528.9350173010403`

### `o3_q90_mean`

- Dimensions differing from baseline candidate: `['property_measure']`
- Feature: Face-connected locations form one contiguous high-speed flow region.
- Criterion: Retain locations at or above the 90th percentile of non-zero recorded speed values.
- Measure: Use mean speed across retained locations as the strength of each retained region.
- Representation: Represent the selected region by the arithmetic mean spatial location of its retained high-speed locations.
- Adjudication: **PROVISIONALLY_ACCEPTED** — Provisional complete O with sustained regional strength.
- Candidate: `o3_q90_mean` (provisionally_accepted)
- Criterion: `>= 90th percentile (1135.982844645889)`; retained points: `4379`
- Connected regions: `4`; sizes: `[54, 116, 2725, 1484]`
- Selected region size: `2725`
- Reported location (mean_spatial_location): `[13.673052853925512, 2.427423816624038, 32.3577929498515]`
- Measure kind: `mean`
- Strength: `1245.4429821372214`

### `o3_q75_mean`

- Dimensions differing from baseline candidate: `['criterion', 'property_measure']`
- Feature: Face-connected locations form one contiguous high-speed flow region.
- Criterion: Retain locations at or above the 75th percentile of non-zero recorded speed values.
- Measure: Use mean speed across retained locations as the strength of each retained region.
- Representation: Represent the selected region by the arithmetic mean spatial location of its retained high-speed locations.
- Adjudication: **PROVISIONALLY_ACCEPTED** — Provisional complete O with a distinct criterion and measure.
- Candidate: `o3_q75_mean` (provisionally_accepted)
- Criterion: `>= 75th percentile (934.449576504776)`; retained points: `10946`
- Connected regions: `6`; sizes: `[10840, 18, 52, 20, 9, 7]`
- Selected region size: `10840`
- Reported location (mean_spatial_location): `[13.448981669732126, 0.24563814047242882, 32.80086336329414]`
- Measure kind: `mean`
- Strength: `1118.6772287108902`

### `o3_q90_peak_point`

- Dimensions differing from baseline candidate: `['aggregation_or_representation']`
- Feature: Face-connected locations form one contiguous high-speed flow region.
- Criterion: Retain locations at or above the 90th percentile of non-zero recorded speed values.
- Measure: Use peak speed as the strength of each retained region.
- Representation: Represent the selected region by the spatial location of its peak-speed location.
- Adjudication: **PROVISIONALLY_ACCEPTED** — Provisional complete O with peak-point representation.
- Candidate: `o3_q90_peak_point` (provisionally_accepted)
- Criterion: `>= 90th percentile (1135.982844645889)`; retained points: `4379`
- Connected regions: `4`; sizes: `[54, 116, 2725, 1484]`
- Selected region size: `2725`
- Reported location (peak_speed_location): `[13.265520095825195, 1.8527919054031372, 31.562437057495117]`
- Measure kind: `peak`
- Strength: `1528.9350173010403`

### `o3_local_peak_features`

- Dimensions differing from baseline candidate: `['feature_definition', 'criterion', 'aggregation_or_representation']`
- Feature: Treat isolated local speed maxima above the selected cutoff as point features rather than connected flow regions.
- Criterion: Retain speed locations above the selected cutoff that are strictly greater than their mesh neighbors.
- Measure: Use peak speed as the strength of each retained region.
- Representation: Represent each feature by the spatial location of its local maximum.
- Adjudication: **REJECTED** — Rejected: isolated point features do not satisfy the question's target entity of a flow region.
- Candidate: `o3_local_peak_features` (rejected)
- Criterion: `>= selected cutoff (0.000000000000)`; retained points: `127`
- Connected regions: `127`; sizes: `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]`
- Selected region size: `1`
- Reported location (mean_spatial_location): `[13.265520095825195, 1.8527919054031372, 31.562437057495117]`
- Measure kind: `peak`
- Strength: `1528.9350173010403`

## O3 breadth review

Criterion, measure, representation and feature-definition candidates were considered. The isolated local-peak candidate was rejected because it represents point features rather than the flow region requested by the question.

Scientific Grounding for this dataset-specific operationalization remains PENDING; Formal Release is HOLD / CONDITIONAL.
