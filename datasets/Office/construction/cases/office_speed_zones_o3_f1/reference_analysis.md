# Office office_speed_zones_o3_f1 Reference Analysis

This construction artifact records executed candidate analyses and curator adjudication.
It is not model-facing input and does not add a tool or scientific-operation schema.

## Authored question

Which airflow region in the office is strongest, and where is it located? Identify and characterize the strongest region using an appropriate scientific analysis, and report its location and strength.

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

- Candidate: `o3_fixed_peak_speed` (provisionally accepted)
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

### `o3_fixed_peak_speed`

- Dimensions differing from baseline candidate: `['none']`
- Feature: Face-connected (6-neighbor) regions of retained high-speed locations form the analysis regions.
- Criterion: Retain locations whose speed magnitude is strictly greater than 0.20.
- Measure: Use peak speed as the strength of each retained region.
- Representation: Represent each selected region by the arithmetic mean spatial location of its retained high-speed locations.
- Adjudication: **PROVISIONALLY_ACCEPTED** — Accepted: a complete, executable interpretation of the strongest region in the Office velocity field.
- Candidate: `o3_fixed_peak_speed` (provisionally accepted)
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

### `o3_fixed_mean_speed`

- Dimensions differing from baseline candidate: `['property_measure']`
- Feature: Face-connected (6-neighbor) regions of retained high-speed locations form the analysis regions.
- Criterion: Retain locations whose speed magnitude is strictly greater than 0.20.
- Measure: Use the mean speed across retained locations as the strength of each region.
- Representation: Represent each selected region by the arithmetic mean spatial location of its retained high-speed locations.
- Adjudication: **PROVISIONALLY_ACCEPTED** — Accepted: a complete, meaningfully distinct sustained-intensity interpretation that is executable on the supplied data.
- Candidate: `o3_fixed_mean_speed` (provisionally accepted)
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

### `o3_q90_mean_speed`

- Dimensions differing from baseline candidate: `['criterion', 'property_measure']`
- Feature: Face-connected (6-neighbor) regions of retained high-speed locations form the analysis regions.
- Criterion: Retain locations whose speed is at or above the 90th percentile of non-zero speed values as a distribution-informed high-speed criterion.
- Measure: Use the mean speed across retained locations as the strength of each region.
- Representation: Represent each selected region by the arithmetic mean spatial location of its retained high-speed locations.
- Adjudication: **PROVISIONALLY_ACCEPTED** — Provisionally retained: it is a complete, executable criterion/measure combination and receives the same pending grounding status as the 0.20 criterion rather than being rejected by an asymmetric standard.
- Candidate: `o3_q90_mean_speed` (provisionally accepted)
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

### `o3_peak_point_speed`

- Dimensions differing from baseline candidate: `['aggregation_or_representation']`
- Feature: Face-connected (6-neighbor) regions of retained high-speed locations form the analysis regions.
- Criterion: Retain locations whose speed magnitude is strictly greater than 0.20.
- Measure: Use peak speed as the strength of each retained region.
- Representation: Represent the selected region by the spatial location of its peak-speed location.
- Adjudication: **PROVISIONALLY_ACCEPTED** — Provisionally retained: peak-point location is a complete and interpretable alternative representation of the strongest region.
- Candidate: `o3_peak_point_speed` (provisionally accepted)
- Criterion: `> 0.20`; retained points: `200`
- Regions: `8`; sizes: `[20, 33, 64, 4, 1, 32, 9, 37]`
- Selected region size: `32`
- Mean spatial location: `[0.06812500045634806, 2.343749985098839, 2.389375001192093]`
- Reported location (peak_speed_location): `[0.009999998845160007, 2.549999952316284, 2.430000066757202]`
- Candidate strength: `0.8049350186348156`
- Selected-region peak speed: `0.8049350186348156`
- Selected-region mean speed: `0.3364087523075579`
- Peak point: `[0.009999998845160007, 2.549999952316284, 2.430000066757202]`
- Measure kind: `peak_speed`

### `o3_local_peak_features`

- Dimensions differing from baseline candidate: `['feature_definition', 'criterion', 'property_measure', 'aggregation_or_representation']`
- Feature: Treat isolated local speed maxima above the selected cutoff as the airflow features rather than contiguous regions.
- Criterion: Retain speed locations above 0.20 that are strictly greater than their face-connected neighbors.
- Measure: Use the speed at each local maximum as feature strength.
- Representation: Represent each feature by the spatial location of its local maximum.
- Adjudication: **REJECTED** — Rejected: execution is possible, but isolated point features do not address the family target of characterizing a contiguous airflow region.
- Candidate: `o3_local_peak_features` (rejected)
- Criterion: `> 0.20 and strictly greater than face-connected neighbors`; retained points: `10`
- Regions: `10`; sizes: `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]`
- Selected region size: `1`
- Mean spatial location: `[0.009999998845160007, 2.549999952316284, 2.430000066757202]`
- Reported location (peak_speed_location): `[0.009999998845160007, 2.549999952316284, 2.430000066757202]`
- Candidate strength: `0.8049350186348156`
- Selected-region peak speed: `0.8049350186348156`
- Selected-region mean speed: `0.8049350186348156`
- Peak point: `[0.009999998845160007, 2.549999952316284, 2.430000066757202]`
- Measure kind: `peak_speed`

## Full-openness breadth review

The exploration includes complete candidates that vary criterion, property measure,
and location representation, plus a feature-definition candidate based on isolated
local maxima. The local-peak feature was executed and rejected because it does not
preserve the family target of a contiguous airflow region. Accepted candidates are
therefore provisional and remain the smallest defensible space found in this pilot;
they are not a pre-frozen O3 registry.

The numerical results are reference-analysis outputs from the supplied flow data.
They do not establish publication-level scientific grounding for the candidate
speed criteria or the connected-region interpretation; that release gate remains pending.
