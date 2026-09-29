# FireFlow fireflow_o2_f1 Reference Analysis

This is a dataset-specific construction artifact, not model-facing input.
All candidates were executed on the supplied reader-visible flow field; no physical units are inferred.

## Authored question

Which high-speed flow region is strongest in the FireFlow room flow field, and where is it located? Identify contiguous high-speed regions using an appropriate speed-based criterion and use an appropriate measure to compare their strength. Treat locations belonging to the same connected mesh neighborhood as one region, represent the selected region by the mean spatial location of its high-speed locations, and report its location and strength.

## Candidate-O exploration and adjudication

### `o2_q90_peak`

- Dimensions differing from baseline candidate: `['none']`
- Feature: Mesh-connected locations form one contiguous high-speed flow region.
- Criterion: Retain locations at or above the 90th percentile of non-zero recorded speed values.
- Measure: Use peak speed as the strength of each retained region.
- Representation: Represent the selected region by the arithmetic mean spatial location of its retained high-speed locations.
- Adjudication: **PROVISIONALLY_ACCEPTED** — Provisional: fixed distribution-informed criterion with peak strength.
- Candidate: `o2_q90_peak` (provisionally_accepted)
- Criterion: `>= 90th percentile (0.973593823082)`; retained points: `1233`
- Connected regions: `4`; sizes: `[517, 710, 3, 3]`
- Selected region size: `517`
- Reported location (mean_spatial_location): `[0.5113998996219967, 1.975870378012814, 0.48199224898616855]`
- Measure kind: `peak`
- Strength: `3.822313175748469`

### `o2_q90_mean`

- Dimensions differing from baseline candidate: `['property_measure']`
- Feature: Mesh-connected locations form one contiguous high-speed flow region.
- Criterion: Retain locations at or above the 90th percentile of non-zero recorded speed values.
- Measure: Use mean speed across retained locations as the strength of each retained region.
- Representation: Represent the selected region by the arithmetic mean spatial location of its retained high-speed locations.
- Adjudication: **PROVISIONALLY_ACCEPTED** — Provisional: same criterion with sustained regional strength.
- Candidate: `o2_q90_mean` (provisionally_accepted)
- Criterion: `>= 90th percentile (0.973593823082)`; retained points: `1233`
- Connected regions: `4`; sizes: `[517, 710, 3, 3]`
- Selected region size: `517`
- Reported location (mean_spatial_location): `[0.5113998996219967, 1.975870378012814, 0.48199224898616855]`
- Measure kind: `mean`
- Strength: `1.6380479260626386`

### `o2_q75_mean`

- Dimensions differing from baseline candidate: `['criterion', 'property_measure']`
- Feature: Mesh-connected locations form one contiguous high-speed flow region.
- Criterion: Retain locations at or above the 75th percentile of non-zero recorded speed values.
- Measure: Use mean speed across retained locations as the strength of each retained region.
- Representation: Represent the selected region by the arithmetic mean spatial location of its retained high-speed locations.
- Adjudication: **PROVISIONALLY_ACCEPTED** — Provisional: a distinct distribution-informed criterion/measure bundle; grounding remains pending.
- Candidate: `o2_q75_mean` (provisionally_accepted)
- Criterion: `>= 75th percentile (0.486353495383)`; retained points: `3081`
- Connected regions: `1`; sizes: `[3081]`
- Selected region size: `3081`
- Reported location (mean_spatial_location): `[2.417761593795914, 1.6560045202195897, 1.0163907527207787]`
- Measure kind: `mean`
- Strength: `1.0174865547746001`

### `o2_q90_integrated_excess`

- Dimensions differing from baseline candidate: `['property_measure']`
- Feature: Mesh-connected locations form one contiguous high-speed flow region.
- Criterion: Retain locations at or above the 90th percentile of non-zero recorded speed values.
- Measure: Use the point-summed speed excess above the selected criterion as region strength.
- Representation: Represent the selected region by the arithmetic mean spatial location of its retained high-speed locations.
- Adjudication: **REJECTED** — Rejected: point-summed strength is sampling-density dependent without a cell-volume contract.
- Candidate: `o2_q90_integrated_excess` (rejected)
- Criterion: `>= 90th percentile (0.973593823082)`; retained points: `1233`
- Connected regions: `4`; sizes: `[517, 710, 3, 3]`
- Selected region size: `517`
- Reported location (mean_spatial_location): `[0.5113998996219967, 1.975870378012814, 0.48199224898616855]`
- Measure kind: `integrated_excess`
- Strength: `343.5227712408218`

## O2 unresolved-dimension review

Both criterion and property measure were explored using complete candidate bundles; no Cartesian product was generated. The q90 and q75 criteria remain provisional under the same grounding standard.

Scientific Grounding for this dataset-specific operationalization remains PENDING; Formal Release is HOLD / CONDITIONAL.
