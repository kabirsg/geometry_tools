""" Tools for preparing vessel geometries for CFD: clipping, centerlines, mesh sizing, volume
meshing and solver input files. Built on VMTK and PyVista.

Library modules:

- `geometry_tools.surface`: `Surfer`, for surface preparation (inlets and outlets, centerlines,
  clipping, branch labelling).
- `geometry_tools.meshing`: `Mesher`, which extends `Surfer` with volume meshing and the solver's
  h5, info and xml.gz files.
- `geometry_tools.vmtk_wrapper`: thin wrappers around `vmtkscripts` that take and return PyVista
  objects.
- `geometry_tools.common`: shared helpers, including the interactive selection tools, sizing
  arrays and flow extensions.
- `geometry_tools.resample_surface`: `Resampler`, for repairing messy segmentations.
- `geometry_tools.make_submission_file`: the job script template for the solver.
- `geometry_tools.volume_mesh`: `VolumeMesh`, for working with an existing volume mesh.
- `geometry_tools.legacy`: Christophe Chnafa's network flow-splitting code, used by
  `Mesher.generate_flow_rates_legacy`.

The command-line pipeline lives in `geometry_tools.scripts`. Each script's usage is in its module
docstring. For the PT pipeline, see `examples/PT_example/PT_example.sh` and Meshing_Pipeline.pdf
in the repository.
"""
from .legacy import *