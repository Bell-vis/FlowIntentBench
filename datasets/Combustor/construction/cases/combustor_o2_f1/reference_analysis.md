# Combustor combustor_o2_f1 Reference Analysis

This is a dataset-specific construction artifact, not model-facing input.
All candidates were executed on the supplied reader-visible flow field; no physical units are inferred.

## Authored question

Which high-speed flow region is strongest in the annular-combustor flow field, and where is it located? Identify contiguous high-speed regions using an appropriate speed-based criterion and use an appropriate measure to compare their strength. Treat locations connected through shared grid faces as one contiguous region, represent the selected region by the mean spatial location of its high-speed locations, and report its location and strength.

## Candidate-O exploration and adjudication

### `o2_q90_peak`

- Dimensions differing from baseline candidate: `['none']`
- Feature: Face-connected locations form one contiguous high-speed flow region.
- Criterion: Retain locations at or above the 90th percentile of non-zero recorded speed values.
- Measure: Use peak speed as the strength of each retained region.
- Representation: Represent the selected region by the arithmetic mean spatial location of its retained high-speed locations.
- Adjudication: **PROVISIONALLY_ACCEPTED** — Provisional: fixed distribution-informed criterion with peak strength.
- Candidate: `o2_q90_peak` (provisionally_accepted)
- Criterion: `>= 90th percentile (1135.982844645889)`; retained points: `4379`
- Connected regions: `4`; sizes: `[54, 116, 2725, 1484]`
- Selected region size: `2725`
- Reported location (mean_spatial_location): `[13.673052853925512, 2.427423816624038, 32.3577929498515]`
- Measure kind: `peak`
- Strength: `1528.9350173010403`

### `o2_q90_mean`

- Dimensions differing from baseline candidate: `['property_measure']`
- Feature: Face-connected locations form one contiguous high-speed flow region.
- Criterion: Retain locations at or above the 90th percentile of non-zero recorded speed values.
- Measure: Use mean speed across retained locations as the strength of each retained region.
- Representation: Represent the selected region by the arithmetic mean spatial location of its retained high-speed locations.
- Adjudication: **PROVISIONALLY_ACCEPTED** — Provisional: same criterion with sustained regional strength.
- Candidate: `o2_q90_mean` (provisionally_accepted)
- Criterion: `>= 90th percentile (1135.982844645889)`; retained points: `4379`
- Connected regions: `4`; sizes: `[54, 116, 2725, 1484]`
- Selected region size: `2725`
- Reported location (mean_spatial_location): `[13.673052853925512, 2.427423816624038, 32.3577929498515]`
- Measure kind: `mean`
- Strength: `1245.4429821372214`

### `o2_q75_mean`

- Dimensions differing from baseline candidate: `['criterion', 'property_measure']`
- Feature: Face-connected locations form one contiguous high-speed flow region.
- Criterion: Retain locations at or above the 75th percentile of non-zero recorded speed values.
- Measure: Use mean speed across retained locations as the strength of each retained region.
- Representation: Represent the selected region by the arithmetic mean spatial location of its retained high-speed locations.
- Adjudication: **PROVISIONALLY_ACCEPTED** — Provisional: a distinct distribution-informed criterion/measure bundle; grounding remains pending.
- Candidate: `o2_q75_mean` (provisionally_accepted)
- Criterion: `>= 75th percentile (934.449576504776)`; retained points: `10946`
- Connected regions: `6`; sizes: `[10840, 18, 52, 20, 9, 7]`
- Selected region size: `10840`
- Reported location (mean_spatial_location): `[13.448981669732126, 0.24563814047242882, 32.80086336329414]`
- Measure kind: `mean`
- Strength: `1118.6772287108902`

### `o2_q90_integrated_excess`

- Dimensions differing from baseline candidate: `['property_measure']`
- Feature: Face-connected locations form one contiguous high-speed flow region.
- Criterion: Retain locations at or above the 90th percentile of non-zero recorded speed values.
- Measure: Use the point-summed speed excess above the selected criterion as region strength.
- Representation: Represent the selected region by the arithmetic mean spatial location of its retained high-speed locations.
- Adjudication: **REJECTED** — Rejected: point-summed strength is sampling-density dependent without a cell-volume contract.
- Candidate: `o2_q90_integrated_excess` (rejected)
- Criterion: `>= 90th percentile (1135.982844645889)`; retained points: `4379`
- Connected regions: `4`; sizes: `[54, 116, 2725, 1484]`
- Selected region size: `2725`
- Reported location (mean_spatial_location): `[13.673052853925512, 2.427423816624038, 32.3577929498515]`
- Measure kind: `integrated_excess`
- Strength: `298278.8746638809`

## O2 unresolved-dimension review

Both criterion and property measure were explored using complete candidate bundles; no Cartesian product was generated. The q90 and q75 criteria remain provisional under the same grounding standard.

Scientific Grounding for this dataset-specific operationalization remains PENDING; Formal Release is HOLD / CONDITIONAL.
