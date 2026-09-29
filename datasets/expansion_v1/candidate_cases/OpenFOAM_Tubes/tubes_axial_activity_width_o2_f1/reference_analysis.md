# tubes_axial_activity_width_o2_f1 Reference Analysis

Status: **CONSTRUCTION PILOT / NOT FORMAL RELEASE**.

Every accepted complete operationalization was executed on the reviewed reader-visible dataset before its G(O) branch was written.

## Accepted operationalizations

- `tubes_axial_activity_width_o2_f1_rms`: Use all cells, with arithmetic means of vertex values for point fields. Use the magnitude of the stated velocity field as a nonnegative spatial weight.; Measure width along the x axis as the weighted root-mean-square distance from its weighted center.; Multiply the stated spatial weight by cell volume and normalize by its sum.
- `tubes_axial_activity_width_o2_f1_interdecile`: Use all cells, with arithmetic means of vertex values for point fields. Use the magnitude of the stated velocity field as a nonnegative spatial weight.; Measure width along the x axis as the weighted 90th-minus-10th coordinate percentile range.; Multiply the stated spatial weight by cell volume and normalize by its sum.

The artifact does not assert a formal release or a model-evaluation result.
