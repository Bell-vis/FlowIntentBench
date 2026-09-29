# FireFlow construction notes

Reviewed sources:

- VTK Examples, `FireFlow` (https://examples.vtk.org/site/Cxx/VisualizationAlgorithms/FireFlow/)
- Mayavi documentation, “Visualizing rich datasets: the fire_ug.vtu example” (https://docs.enthought.com/mayavi/mayavi/example_fire.html)

Mayavi directly identifies `fire_ug.vtu` as an unstructured VTK XML grid for a room with a fire in one corner and identifies `room_vis.wrl` as the room-context geometry. The official VTK example independently states that the `.wrl` geometry and `.vtu` solution are combined. These statements populate static context. The reader, stream-tracer, and isosurface pipeline is operationalization evidence. No flow-derived finding is emitted.
