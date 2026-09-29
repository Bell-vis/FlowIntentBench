# Version update

Plot Data for the accepted paper is updated.


# Simulation Data

Each folder contains the results at the last time step.

The folder name format: `[Simulation number]-Q[Flow rate in ml/h]I[Applied current in mA]R[Hydrogen bubble nucleation radius in um][With or without surfactant]`

Example: `01-Q300I45R202NoSurfactant` -> Flow rate: 300 ml/h, Current: 45 mA, Hydrogen bubble nucleation radius: 202 μm, Without surfactant

The containers in each folder can be opened using Paraview. The `.vtk` files can be opened directly. For files with `raw` and `xmf` formats, the `xmf` file should be opened, and then the "XDMF Reader" in Paraview should be selected.

Each folder contains:

- `bc.vtk`: Boundary conditions
- `omz_#.raw` and `omz_#.xmf`: Vorticity in the z-direction
- `p_#.raw` and `p_#.xmf`: Pressure
- `potsolver_#.vtk`: Electrical potential and Current density vector
- `s_#.vtk` and `sm_#.vtk`: Bubbles interface
- `vf_#.raw` and `vf_#.xmf`: Volume fraction
- `vx_#.raw` and `vx_#.xmf`: Velocity in the x-direction
- `vy_#.raw` and `vy_#.xmf`: Velocity in the y-direction
- `vz_#.raw` and `vz_#.xmf`: Velocity in the z-direction

# Plot Data

The files contain the data used to draw the figures in the paper. The name of each filed indicate the figure number.