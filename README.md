# Geometry tools 

The module `vmtk_wrapper` is largely a wrapper for `vmtkscripts`, incorporating the ease-of-use of `PyVista`. 

Moving forward, it would be nice to re-write this using `vtkvmtk` and `PyVista` directly, but wrapping `vmtkscripts` works for now. 

- `surface` contains the `Surfer()` class, which incorporates surface-based operations.
- `meshing` contains the `Mesher()` class for generating volume meshes and files necessary for simulation. Inherits from `Surfer()`. Note: VMTK TetGen is reallly buggy, YMMV.
- `resample_surface` contains `Resampler()` to sample ugly meshes to an image, then re-contour the image to get a surface. 
- `make_submission_file` makes the bash submission file formatted for the BSL solver on Mehdi's niagara.
- `common` contains some useful odds and ends.

VMTK has a lot of out-of-date requirements, so `surface`, `meshing`, and `vmtk_wrapper` rely on VMTK, but `common` and `resample_surface` may have different requirements. 

The `Scripts` directory contains a number of useful scripts and information on how to call them.

Requirements:
- vtk
- numpy
- pyvista=0.29
- scipy
- h5py
- pymeshfix
- vmtk
- networkx
- matplotlib
- TubeClipper (https://github.com/Biomedical-Simulation-Lab/tubeclipper)
- pygeodesic

# Environment
First, conda env as here: http://www.vmtk.org/download/

Then 
`conda install -c conda-forge pyvista networkx scipy ipython h5py matplotlib`

Then clone and install `tubeclipper` using pip. 

When installing on workstation (ubuntu), also had to `conda install llvm=3.3` and make sure `pyvista=0.29`

# Meshing
For an example of using this for meshing, see the `meshing_example.sh` file in `scripts`.

# To install with VMTK 1.5 on ubuntu 
(WARNING: There are issues with the rendering)

1) clean up tarballs and unused packages
`conda clean -a`

2) create conda environment for vmtk=1.5.0 
`conda create -n vmtk15 -c conda-forge python=3.7 itk vtk`  
`conda activate vmtk15`
`conda install -c conda-forge vmtk`

3) install some packages
`conda install -c conda-forge scipy ipython pyvista=0.34.0 networkx matplotlib`
`pip install pygeodesic`

4) install tubeclipper
`cd /path/to/tubeclipper`
`pip install -e .`

5) install geometry_tools
`cd /path/to/geometry_tools/`
`pip install -e .`

# Warning
There are newer versions of pyvista that have more functionality. Do not be tempted! This will have to be updated at some point in the near future.

# How To Run
The Meshing_Pipeline.pdf file was written by Gurnish and Anna and is up to date, except for the map_info_direct file that takes in arguments differently now. Please look at docstring at the top of the file or ask Kabir for details.