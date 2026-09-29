# Kitchen kitchen_o2_f1 Reference Analysis

This is a dataset-specific construction artifact, not model-facing input.
All candidates were executed on the supplied reader-visible flow field; no physical units are inferred.

## Authored question

Which high-speed flow region is strongest in the kitchen airflow field, and where is it located? Identify contiguous high-speed regions using an appropriate speed-based criterion and use an appropriate measure to compare their strength. Treat locations connected through shared grid faces as one contiguous region, represent the selected region by the mean spatial location of its high-speed locations, and report its location and strength.

## Candidate-O exploration and adjudication

### `o2_q90_peak`

- Dimensions differing from baseline candidate: `['none']`
- Feature: Face-connected locations form one contiguous high-speed flow region.
- Criterion: Retain locations at or above the 90th percentile of non-zero recorded speed values.
- Measure: Use peak speed as the strength of each retained region.
- Representation: Represent the selected region by the arithmetic mean spatial location of its retained high-speed locations.
- Adjudication: **PROVISIONALLY_ACCEPTED** — Provisional: fixed distribution-informed criterion with peak strength.
- Candidate: `o2_q90_peak` (provisionally_accepted)
- Criterion: `>= 90th percentile (0.109679118156)`; retained points: `1130`
- Connected regions: `9`; sizes: `[214, 278, 93, 466, 39, 1, 5, 31, 3]`
- Selected region size: `278`
- Reported location (mean_spatial_location): `[0.4474820164132783, 4.69920858719366, 0.6970863330626874]`
- Measure kind: `peak`
- Strength: `0.45226635527610837`

### `o2_q90_mean`

- Dimensions differing from baseline candidate: `['property_measure']`
- Feature: Face-connected locations form one contiguous high-speed flow region.
- Criterion: Retain locations at or above the 90th percentile of non-zero recorded speed values.
- Measure: Use mean speed across retained locations as the strength of each retained region.
- Representation: Represent the selected region by the arithmetic mean spatial location of its retained high-speed locations.
- Adjudication: **PROVISIONALLY_ACCEPTED** — Provisional: same criterion with sustained regional strength.
- Candidate: `o2_q90_mean` (provisionally_accepted)
- Criterion: `>= 90th percentile (0.109679118156)`; retained points: `1130`
- Connected regions: `9`; sizes: `[214, 278, 93, 466, 39, 1, 5, 31, 3]`
- Selected region size: `93`
- Reported location (mean_spatial_location): `[6.447096727227652, 4.213978562303769, 0.2433333361060709]`
- Measure kind: `mean`
- Strength: `0.23879296346952839`

### `o2_q75_mean`

- Dimensions differing from baseline candidate: `['criterion', 'property_measure']`
- Feature: Face-connected locations form one contiguous high-speed flow region.
- Criterion: Retain locations at or above the 75th percentile of non-zero recorded speed values.
- Measure: Use mean speed across retained locations as the strength of each retained region.
- Representation: Represent the selected region by the arithmetic mean spatial location of its retained high-speed locations.
- Adjudication: **PROVISIONALLY_ACCEPTED** — Provisional: a distinct distribution-informed criterion/measure bundle; grounding remains pending.
- Candidate: `o2_q75_mean` (provisionally_accepted)
- Criterion: `>= 75th percentile (0.067245612765)`; retained points: `2824`
- Connected regions: `5`; sizes: `[2814, 2, 1, 1, 6]`
- Selected region size: `2814`
- Reported location (mean_spatial_location): `[3.632793141895668, 3.2369260641599937, 1.425760498289919]`
- Measure kind: `mean`
- Strength: `0.1204053141440439`

### `o2_q90_integrated_excess`

- Dimensions differing from baseline candidate: `['property_measure']`
- Feature: Face-connected locations form one contiguous high-speed flow region.
- Criterion: Retain locations at or above the 90th percentile of non-zero recorded speed values.
- Measure: Use the point-summed speed excess above the selected criterion as region strength.
- Representation: Represent the selected region by the arithmetic mean spatial location of its retained high-speed locations.
- Adjudication: **REJECTED** — Rejected: point-summed strength is sampling-density dependent without a cell-volume contract.
- Candidate: `o2_q90_integrated_excess` (rejected)
- Criterion: `>= 90th percentile (0.109679118156)`; retained points: `1130`
- Connected regions: `9`; sizes: `[214, 278, 93, 466, 39, 1, 5, 31, 3]`
- Selected region size: `466`
- Reported location (mean_spatial_location): `[5.621244591704766, 3.9145922396360007, 1.1744206037368354]`
- Measure kind: `integrated_excess`
- Strength: `25.267098290797325`

## O2 unresolved-dimension review

Both criterion and property measure were explored using complete candidate bundles; no Cartesian product was generated. The q90 and q75 criteria remain provisional under the same grounding standard.

Scientific Grounding for this dataset-specific operationalization remains PENDING; Formal Release is HOLD / CONDITIONAL.
