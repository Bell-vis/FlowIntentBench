# Combustor construction notes

Reviewed sources:

- VTK Examples, `WarpCombustor` (https://examples.vtk.org/site/Cxx/VisualizationAlgorithms/WarpCombustor/)
- VTK Examples, `CombustorIsosurface` (https://examples.vtk.org/site/Cxx/VisualizationAlgorithms/CombustorIsosurface/)

The WarpCombustor code explicitly uses `combxyz.bin` and `combq.bin`, identifies the data as an annular combustor, and describes the gas-turbine fuel/air setting. Those static statements are context evidence. Computational-plane extraction, scalar warping, and density isosurface generation are analysis procedures and therefore remain operationalization evidence. No example visualization result is frozen as a finding.
