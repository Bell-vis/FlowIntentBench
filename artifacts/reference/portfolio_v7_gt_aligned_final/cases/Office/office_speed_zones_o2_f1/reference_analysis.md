# office_speed_zones_o2_f1 Reference Analysis

Status: AGENT_READY construction artifact; values below are rendered from the structured deterministic materialization.

## Effective O `o2_peak_speed`

- Execution status: `MATERIALIZED`
- Reproducible: `True`
- Dataset manifest: `13723df1cea9f5ceaabb7c1a98c2661070be91cc96f334ec6ffbe0d57784be3f`

### G(O)

- `operation_id`: `o2_peak_speed`
- `criterion_kind`: `fixed_threshold`
- `criterion`: `> 0.2`
- `comparator`: `GT`
- `threshold`: `0.2`
- `measure_kind`: `peak`
- `representation`: `mean`
- `retained_points`: `200`
- `region_count`: `8`
- `region_sizes`: `[20,33,64,4,1,32,9,37]`
- `selected_region_size`: `32`
- `reported_location`: `[0.06812500045634806,2.343749985098839,2.389375001192093]`
- `mean_spatial_location`: `[0.06812500045634806,2.343749985098839,2.389375001192093]`
- `peak_point`: `[0.009999998845160007,2.549999952316284,2.430000066757202]`
- `coordinate_span`: `[0.24000000115484,0.5999999046325684,0.24000000953674316]`
- `strength`: `0.8049350186348156`
- `peak_speed`: `0.8049350186348156`
- `mean_speed`: `0.3364087523075579`

## Effective O `o2_mean_speed`

- Execution status: `MATERIALIZED`
- Reproducible: `True`
- Dataset manifest: `13723df1cea9f5ceaabb7c1a98c2661070be91cc96f334ec6ffbe0d57784be3f`

### G(O)

- `operation_id`: `o2_mean_speed`
- `criterion_kind`: `fixed_threshold`
- `criterion`: `> 0.2`
- `comparator`: `GT`
- `threshold`: `0.2`
- `measure_kind`: `mean`
- `representation`: `mean`
- `retained_points`: `200`
- `region_count`: `8`
- `region_sizes`: `[20,33,64,4,1,32,9,37]`
- `selected_region_size`: `64`
- `reported_location`: `[4.5,2.4312500208616257,2.1737499833106995]`
- `mean_spatial_location`: `[4.5,2.4312500208616257,2.1737499833106995]`
- `peak_point`: `[4.5,1.600000023841858,2.3499999046325684]`
- `coordinate_span`: `[0.0,2.5,0.9900000095367432]`
- `strength`: `0.3577313150236388`
- `peak_speed`: `0.4289544897909436`
- `mean_speed`: `0.3577313150236388`

## Effective O `o2_q90_mean_speed`

- Execution status: `MATERIALIZED`
- Reproducible: `True`
- Dataset manifest: `13723df1cea9f5ceaabb7c1a98c2661070be91cc96f334ec6ffbe0d57784be3f`

### G(O)

- `operation_id`: `o2_q90_mean_speed`
- `criterion_kind`: `quantile`
- `criterion`: `>= non-zero 90th percentile (0.120388175490)`
- `comparator`: `GE`
- `threshold`: `0.12038817548967597`
- `measure_kind`: `mean`
- `representation`: `mean`
- `retained_points`: `817`
- `region_count`: `7`
- `region_sizes`: `[247,154,296,93,11,12,4]`
- `selected_region_size`: `93`
- `reported_location`: `[4.497741883800876,2.421505397365939,2.1120429859366467]`
- `mean_spatial_location`: `[4.497741883800876,2.421505397365939,2.1120429859366467]`
- `peak_point`: `[4.5,1.600000023841858,2.3499999046325684]`
- `coordinate_span`: `[0.010000228881835938,2.5,1.2400000095367432]`
- `strength`: `0.2919336353878951`
- `peak_speed`: `0.4289544897909436`
- `mean_speed`: `0.2919336353878951`
