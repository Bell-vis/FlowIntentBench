# Office office_speed_zones_o2_f1 Reference Analysis

This construction artifact records executed candidate analyses and curator adjudication.
It is not model-facing input and does not add a tool or scientific-operation schema.

## Authored question

Which high-speed airflow region is strongest in the office, and where is it located? Identify contiguous high-speed regions using an appropriate speed-based criterion and use an appropriate measure to compare their strength. Treat locations connected through shared grid faces as belonging to the same region, represent the selected region by the mean spatial location of its high-speed locations, and report its location and strength.

## Reference execution baseline

The Office legacy VTK file was read with `vtkDataSetReader` and the existing canonical
read-all-scalars/read-all-vectors configuration. Mean locations are arithmetic means
of retained locations, while peak-point candidates report the location of the selected
region's peak-speed location. Candidate
analyses below use only the supplied velocity field and face-connected structured-grid
regions; no units are inferred for the stored velocity scale.

The provisionally accepted O space is the smallest candidate set judged defensible for this pilot.
Every provisionally accepted candidate has an independent G(O) branch in ground_truth.json.

## Baseline candidate result

- Candidate: `o2_peak_speed` (provisionally accepted)
- Criterion: `> 0.20`; retained points: `200`
- Regions: `8`; sizes: `[20, 33, 64, 4, 1, 32, 9, 37]`
- Selected region size: `32`
- Mean spatial location: `[0.06812500045634806, 2.343749985098839, 2.389375001192093]`
- Reported location (mean_spatial_location): `[0.06812500045634806, 2.343749985098839, 2.389375001192093]`
- Candidate strength: `0.8049350186348156`
- Selected-region peak speed: `0.8049350186348156`
- Selected-region mean speed: `0.3364087523075579`
- Peak point: `[0.009999998845160007, 2.549999952316284, 2.430000066757202]`
- Measure kind: `peak_speed`

## Candidate-O exploration and adjudication

### `o2_peak_speed`

- Dimensions differing from baseline candidate: `['none']`
- Feature: Face-connected (6-neighbor) regions of retained high-speed locations form the analysis regions.
- Criterion: Retain locations whose speed magnitude is strictly greater than 0.20.
- Measure: Use peak speed as the strength of each retained region.
- Representation: Represent each selected region by the arithmetic mean spatial location of its retained high-speed locations.
- Adjudication: **PROVISIONALLY_ACCEPTED** — Accepted: it directly represents the most intense local airflow within each thresholded region.
- Candidate: `o2_peak_speed` (provisionally accepted)
- Criterion: `> 0.20`; retained points: `200`
- Regions: `8`; sizes: `[20, 33, 64, 4, 1, 32, 9, 37]`
- Selected region size: `32`
- Mean spatial location: `[0.06812500045634806, 2.343749985098839, 2.389375001192093]`
- Reported location (mean_spatial_location): `[0.06812500045634806, 2.343749985098839, 2.389375001192093]`
- Candidate strength: `0.8049350186348156`
- Selected-region peak speed: `0.8049350186348156`
- Selected-region mean speed: `0.3364087523075579`
- Peak point: `[0.009999998845160007, 2.549999952316284, 2.430000066757202]`
- Measure kind: `peak_speed`

### `o2_mean_speed`

- Dimensions differing from baseline candidate: `['property_measure']`
- Feature: Face-connected (6-neighbor) regions of retained high-speed locations form the analysis regions.
- Criterion: Retain locations whose speed magnitude is strictly greater than 0.20.
- Measure: Use the mean speed across retained locations as the strength of each region.
- Representation: Represent each selected region by the arithmetic mean spatial location of its retained high-speed locations.
- Adjudication: **PROVISIONALLY_ACCEPTED** — Accepted: it represents sustained regional airflow intensity and is computable from the supplied velocity field.
- Candidate: `o2_mean_speed` (provisionally accepted)
- Criterion: `> 0.20`; retained points: `200`
- Regions: `8`; sizes: `[20, 33, 64, 4, 1, 32, 9, 37]`
- Selected region size: `64`
- Mean spatial location: `[4.5, 2.4312500208616257, 2.1737499833106995]`
- Reported location (mean_spatial_location): `[4.5, 2.4312500208616257, 2.1737499833106995]`
- Candidate strength: `0.3577313150236388`
- Selected-region peak speed: `0.4289544897909436`
- Selected-region mean speed: `0.3577313150236388`
- Peak point: `[4.5, 1.600000023841858, 2.3499999046325684]`
- Measure kind: `mean_speed`

### `o2_q90_mean_speed`

- Dimensions differing from baseline candidate: `['criterion', 'property_measure']`
- Feature: Face-connected (6-neighbor) regions of retained high-speed locations form the analysis regions.
- Criterion: Retain locations whose speed is at or above the 90th percentile of non-zero speed values as a distribution-informed high-speed criterion.
- Measure: Use the mean speed across retained locations as the strength of each region.
- Representation: Represent each selected region by the arithmetic mean spatial location of its retained high-speed locations.
- Adjudication: **PROVISIONALLY_ACCEPTED** — Provisionally retained: a complete, executable distribution-informed speed criterion that is materially different from the fixed threshold; its scientific grounding remains pending under the same gate as 0.20.
- Candidate: `o2_q90_mean_speed` (provisionally accepted)
- Criterion: `>= NONZERO_SPEED_POINTS 90th percentile (0.120388175490)`; retained points: `817`
- Regions: `7`; sizes: `[247, 154, 296, 93, 11, 12, 4]`
- Selected region size: `93`
- Mean spatial location: `[4.497741883800876, 2.421505397365939, 2.1120429859366467]`
- Reported location (mean_spatial_location): `[4.497741883800876, 2.421505397365939, 2.1120429859366467]`
- Candidate strength: `0.2919336353878951`
- Selected-region peak speed: `0.4289544897909436`
- Selected-region mean speed: `0.2919336353878951`
- Peak point: `[4.5, 1.600000023841858, 2.3499999046325684]`
- Measure kind: `mean_speed`

### `o2_integrated_speed_excess`

- Dimensions differing from baseline candidate: `['property_measure']`
- Feature: Face-connected (6-neighbor) regions of retained high-speed locations form the analysis regions.
- Criterion: Retain locations whose speed magnitude is strictly greater than 0.20.
- Measure: Use the point-summed speed excess above 0.20 as the strength of each region.
- Representation: Represent each selected region by the arithmetic mean spatial location of its retained high-speed locations.
- Adjudication: **REJECTED** — Rejected: a point sum is sampling-density dependent and no cell-volume integration contract is available for this point-based task.
- Candidate: `o2_integrated_speed_excess` (rejected)
- Criterion: `> 0.20`; retained points: `200`
- Regions: `8`; sizes: `[20, 33, 64, 4, 1, 32, 9, 37]`
- Selected region size: `64`
- Mean spatial location: `[4.5, 2.4312500208616257, 2.1737499833106995]`
- Reported location (mean_spatial_location): `[4.5, 2.4312500208616257, 2.1737499833106995]`
- Candidate strength: `10.094804161512885`
- Selected-region peak speed: `0.4289544897909436`
- Selected-region mean speed: `0.3577313150236388`
- Peak point: `[4.5, 1.600000023841858, 2.3499999046325684]`
- Measure kind: `integrated_excess`

## Unresolved criterion and measure sensitivity

The unresolved `property_measure` is consequential on this dataset: changing
from peak speed to mean speed changes both the selected strongest region and
the resulting reference strength/location (peak: 0.8049350186 at [0.06812500045634806, 2.343749985098839, 2.389375001192093]; mean: 0.3577313150 at [4.5, 2.4312500208616257, 2.1737499833106995]).
The unresolved `criterion` was also genuinely explored. The distribution-informed
non-zero 90th-percentile criterion retains a different region and is provisionally
retained for comparison (strength: 0.2919336354 at [4.497741883800876, 2.421505397365939, 2.1120429859366467]).
Both the fixed 0.20 and q90 criteria remain subject to the same pending Scientific
Grounding Gate; q90 is not rejected merely because grounding is pending.

The numerical results are reference-analysis outputs from the supplied flow data.
They do not establish publication-level scientific grounding for the candidate
speed criteria or the connected-region interpretation; that release gate remains pending.
