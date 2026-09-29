# mhd_velocity_magnetic_alignment_o3_f1 Reference Analysis

Status: **CONSTRUCTION PILOT / NOT FORMAL RELEASE**.

Every accepted complete operationalization was executed on the reviewed reader-visible dataset before its G(O) branch was written.

## Accepted operationalizations

- `mhd_velocity_magnetic_alignment_o3_f1_mean_cosine`: Use all cells, with stored cell values directly and arithmetic means of vertex values for point fields.; Average the local cosine of the angle between the cell velocity and magnetic vectors.; Weight cell statistics by cell volume and normalize by the included volume.
- `mhd_velocity_magnetic_alignment_o3_f1_normalized_dot`: Use all cells, with stored cell values directly and arithmetic means of vertex values for point fields.; Divide the volume-mean dot product by the square root of the product of the two volume-mean squared vector magnitudes.; Weight cell statistics by cell volume and normalize by the included volume.

The artifact does not assert a formal release or a model-evaluation result.
