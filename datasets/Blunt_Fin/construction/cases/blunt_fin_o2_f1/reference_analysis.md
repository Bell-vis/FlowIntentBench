# Blunt_Fin blunt_fin_o2_f1 Reference Analysis

This is a dataset-specific construction artifact, not model-facing input.
All candidates were executed on the supplied reader-visible flow field; no physical units are inferred.

## Authored question

Which high-speed flow region is strongest in the blunt-fin flow field, and where is it located? Identify contiguous high-speed regions using an appropriate speed-based criterion and use an appropriate measure to compare their strength. Treat locations connected through shared grid faces as one contiguous region, represent the selected region by the mean spatial location of its high-speed locations, and report its location and strength.

## Candidate-O exploration and adjudication

### `o2_q90_peak`

- Dimensions differing from baseline candidate: `['none']`
- Feature: Face-connected locations form one contiguous high-speed flow region.
- Criterion: Retain locations at or above the 90th percentile of non-zero recorded speed values.
- Measure: Use peak speed as the strength of each retained region.
- Representation: Represent the selected region by the arithmetic mean spatial location of its retained high-speed locations.
- Adjudication: **PROVISIONALLY_ACCEPTED** — Provisional: fixed distribution-informed criterion with peak strength.
- Candidate: `o2_q90_peak` (provisionally_accepted)
- Criterion: `>= 90th percentile (2.776219898197)`; retained points: `3844`
- Connected regions: `6`; sizes: `[3831, 1, 2, 1, 7, 2]`
- Selected region size: `3831`
- Reported location (mean_spatial_location): `[-0.43620374257338634, 3.0521042773374156, 2.5893470661131714]`
- Measure kind: `peak`
- Strength: `3.1229319383189305`

### `o2_q90_mean`

- Dimensions differing from baseline candidate: `['property_measure']`
- Feature: Face-connected locations form one contiguous high-speed flow region.
- Criterion: Retain locations at or above the 90th percentile of non-zero recorded speed values.
- Measure: Use mean speed across retained locations as the strength of each retained region.
- Representation: Represent the selected region by the arithmetic mean spatial location of its retained high-speed locations.
- Adjudication: **PROVISIONALLY_ACCEPTED** — Provisional: same criterion with sustained regional strength.
- Candidate: `o2_q90_mean` (provisionally_accepted)
- Criterion: `>= 90th percentile (2.776219898197)`; retained points: `3844`
- Connected regions: `6`; sizes: `[3831, 1, 2, 1, 7, 2]`
- Selected region size: `3831`
- Reported location (mean_spatial_location): `[-0.43620374257338634, 3.0521042773374156, 2.5893470661131714]`
- Measure kind: `mean`
- Strength: `2.922646208400968`

### `o2_q75_mean`

- Dimensions differing from baseline candidate: `['criterion', 'property_measure']`
- Feature: Face-connected locations form one contiguous high-speed flow region.
- Criterion: Retain locations at or above the 75th percentile of non-zero recorded speed values.
- Measure: Use mean speed across retained locations as the strength of each retained region.
- Representation: Represent the selected region by the arithmetic mean spatial location of its retained high-speed locations.
- Adjudication: **PROVISIONALLY_ACCEPTED** — Provisional: a distinct distribution-informed criterion/measure bundle; grounding remains pending.
- Candidate: `o2_q75_mean` (provisionally_accepted)
- Criterion: `>= 75th percentile (2.356383458626)`; retained points: `9610`
- Connected regions: `1`; sizes: `[9610]`
- Selected region size: `9610`
- Reported location (mean_spatial_location): `[1.2812338045775953, 2.2513677567553696, 1.922935472026421]`
- Measure kind: `mean`
- Strength: `2.6986647780551447`

### `o2_q90_integrated_excess`

- Dimensions differing from baseline candidate: `['property_measure']`
- Feature: Face-connected locations form one contiguous high-speed flow region.
- Criterion: Retain locations at or above the 90th percentile of non-zero recorded speed values.
- Measure: Use the point-summed speed excess above the selected criterion as region strength.
- Representation: Represent the selected region by the arithmetic mean spatial location of its retained high-speed locations.
- Adjudication: **REJECTED** — Rejected: point-summed strength is sampling-density dependent without a cell-volume contract.
- Candidate: `o2_q90_integrated_excess` (rejected)
- Criterion: `>= 90th percentile (2.776219898197)`; retained points: `3844`
- Connected regions: `6`; sizes: `[3831, 1, 2, 1, 7, 2]`
- Selected region size: `3831`
- Reported location (mean_spatial_location): `[-0.43620374257338634, 3.0521042773374156, 2.5893470661131714]`
- Measure kind: `integrated_excess`
- Strength: `560.959194391829`

## O2 unresolved-dimension review

Both criterion and property measure were explored using complete candidate bundles; no Cartesian product was generated. The q90 and q75 criteria remain provisional under the same grounding standard.

Scientific Grounding for this dataset-specific operationalization remains PENDING; Formal Release is HOLD / CONDITIONAL.
