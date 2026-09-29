# Office office_speed_zones_o1_f1 Reference Analysis

This construction artifact records executed candidate analyses and curator adjudication.
It is not model-facing input and does not add a tool or scientific-operation schema.

## Authored question

Which high-speed airflow region is strongest in the office, and where is it located? For this analysis, define high-speed airflow as velocity magnitude above 0.20 in the stored velocity scale, treat locations connected through shared grid faces as one contiguous region, use peak speed as the measure of strength, and represent the strongest region by the mean spatial location of its retained high-speed locations. Identify the strongest region and report its location and strength.

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

- Candidate: `o1_peak_speed` (provisionally accepted)
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

The numerical results are reference-analysis outputs from the supplied flow data.
They do not establish publication-level scientific grounding for the candidate
speed criteria or the connected-region interpretation; that release gate remains pending.
