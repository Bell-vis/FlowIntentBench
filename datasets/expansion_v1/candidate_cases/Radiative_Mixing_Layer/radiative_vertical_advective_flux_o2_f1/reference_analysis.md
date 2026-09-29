# radiative_vertical_advective_flux_o2_f1 Reference Analysis

Status: **CONSTRUCTION PILOT / NOT FORMAL RELEASE**.

Every accepted complete operationalization was executed on the reviewed reader-visible dataset before its G(O) branch was written.

## Accepted operationalizations

- `radiative_vertical_advective_flux_o2_f1_cell_product`: Use the complete density and vertical-velocity fields on the supplied mesh.; First average density and vertical velocity to cells, then multiply their cell values.; Weight cell statistics by cell volume and normalize by the included volume.
- `radiative_vertical_advective_flux_o2_f1_point_product`: Use the complete density and vertical-velocity fields on the supplied mesh.; Multiply density by vertical velocity at each stored point, then average that product to cells.; Weight cell statistics by cell volume and normalize by the included volume.

The artifact does not assert a formal release or a model-evaluation result.
