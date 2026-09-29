# Data provenance

The release contains all 15 numerical inputs referenced by the 96-case manifest,
including source assets and construction recipes retained in the original project.
The 15 inputs represent 14 parent source groups: Electrolyzer and
Electrolyzer_Electric come from the same source release and have different meshes.

| Inputs | Source records retained in the dataset directories |
|---|---|
| Blunt_Fin, Carotid, Combustor, FireFlow, Kitchen, NASA_LOx_Post, Office | VTK example provenance and reader conventions |
| AIDEAS_Blow_Mold | Zenodo record 15720265; selected blow-molding VTU |
| Double_Fin | Zenodo record 10697026; pressure field and geometry |
| Electrolyzer, Electrolyzer_Electric | Zenodo record 14935434; selected flow snapshot and separate electric-field mesh |
| Rayleigh_Taylor, Radiative_Mixing_Layer, MHD_Turbulence | The Well; fixed source revisions and selected snapshots |
| OpenFOAM_Tubes | PyVista example data; fixed source commit and conversion settings |

Inspect `dataset_manifest.json` for paths, byte sizes, checksums, reader configuration,
and provenance, and `data_metadata.json` for field/mesh descriptions. Where present,
`sources/`, `construction/source_assets.json`, and `construction/source_import.json`
provide source metadata, source-specific license information, transformations, and
selected-file identity. Public source-author attribution remains intact.

The Well snapshots and selected ZIP members include retained byte-range data for
conversion reproducibility. A selected-snapshot checksum does not claim to verify an
entire multi-GB upstream object. Source assets are not silently substituted or
re-downloaded during normal evaluation.

```bash
python scripts/verify_release.py --read-data
```

Reconstruction helpers are in `scripts/build_context_construction.py` and
`scripts/reference_construction/`. Reconstruct into a separate working copy: the
reference inputs in this release are checksum-bound, and changing scientific
content requires a newly versioned reference package.
