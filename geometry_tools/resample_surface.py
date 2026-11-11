""" Repair messy surfaces by resampling them to an image and re-contouring.

`Resampler` decimates a surface, voxelizes it, contours the voxels back into a surface, smooths it,
and cleans it up with pymeshfix. Use it on segmentations with holes, self-intersections or junk
points before running `surface_prep.py`. Needs pymeshfix but not VMTK.
"""
import pyvista as pv 
import numpy as np 
from pathlib import Path
import pymeshfix
from geometry_tools import common as cc

class Resampler():
    """ Fix a messy mesh using resampling and pymeshfix
    """
    def __init__(self, surf, resample_spacing=0.05):
        self.surf = surf 
        self.resample_spacing = resample_spacing

    def remove_junk_points(self, thresh=1e4):
        """ Remove points with any coordinate beyond +/- `thresh`, then keep the largest connected piece. """
        junk_points = np.unique(np.where(np.abs(self.surf.points) > thresh)[0])
        self.surf, _ = self.surf.remove_points(junk_points)
        self.surf = self.surf.extract_largest()

    def fix(self):
        """ Resample `self.surf` and store the repaired surface back in `self.surf`.

        Decimates to at most about 75,000 points, voxelizes with spacing `resample_spacing`,
        contours, applies Taubin smoothing, decimates again and repairs the result with pymeshfix.
        The grid is kept in `self.grid`.
        """
        # Decimate the surface
        surf = self.surf 

        if surf.n_points > 75000:
            decimate_ratio = 1 - 75000 / surf.n_points
        else:
            decimate_ratio = 0.05
        surf = surf.decimate(decimate_ratio)
        surf = surf.clean(tolerance=1e-4)

        # Resample the surface to a grid, recontour
        self.grid = cc.vtk_generate_img_stencil(surf.fill_holes(15.), spacing=self.resample_spacing)

        surf_r = self.grid.contour([0.5])
        surf_smooth = cc.vtk_taubin_smooth(surf_r, pass_band=0.03, iterations=100)

        # Decimate again
        if surf_smooth.n_points > 75000:
            decimate_ratio = 1 - 75000 / surf_smooth.n_points
        else:
            decimate_ratio = 0.05
        surf_decimate = surf_smooth.decimate(decimate_ratio)
        
        # Use pymeshfix to get rid of junk
        surf = surf_decimate.triangulate()
        meshfix = pymeshfix.MeshFix(surf)
        meshfix.repair(verbose=True)
        surf = meshfix.mesh
        surf = surf.clean()
        self.surf = surf

def combine_surfaces_as_image(surf, roi, spacing=0.05, bounds=None):
    """ Resample surfs as img then merge and contour.
    """
    if bounds is None:
        roi_bounds = [roi.bounds]
    else:
        roi_bounds = bounds

    grid = cc.vtk_generate_img_stencil(surf.fill_holes(15.), spacing=spacing, 
        bounds=surf.bounds)
    grid_roi = cc.vtk_generate_img_stencil(roi.fill_holes(15.), spacing=spacing,
        bounds=surf.bounds)
    
    bounds_idx = []
    for bound in roi_bounds:
        ii = bounds_to_indicies(bound, grid.origin, spacing)
        bounds_idx.append(ii)

    # Replace
    img = grid.point_data['ImageScalars'].reshape(grid.dimensions, order='F').copy()
    img_roi = grid_roi.point_data['ImageScalars'].reshape(grid_roi.dimensions, order='F')

    for ii in bounds_idx:
        img[ii[0]:ii[1], ii[2]:ii[3], ii[4]:ii[5]] = img_roi[ii[0]:ii[1], ii[2]:ii[3], ii[4]:ii[5]]

    merged = pv.wrap(img)
    merged.origin = grid.origin
    merged.dimensions = grid.dimensions
    merged.spacing = grid.spacing

    surf = grid.contour([0.5])
    surf = cc.vtk_taubin_smooth(surf, pass_band=0.03, iterations=100)

    surf_r = merged.contour([0.5])
    surf_r = cc.vtk_taubin_smooth(surf_r, pass_band=0.03, iterations=100)

    return surf, surf_r

def combine_surfaces(surf, roi, bounds=None):
    """ Merge two surfaces.
    """
    if bounds is None:
        roi_bounds = [roi.bounds]
    else:
        roi_bounds = bounds

    surf_rois = [roi.clip_box(bounds, invert=False) for bounds in roi_bounds]
    surf_r = surf.copy()

    for bounds in roi_bounds:
        surf_r = surf_r.clip_box(bounds, invert=True)
    
    for roi in surf_rois:
        surf_r = surf_r.merge(roi)
    
    surf_r = surf_r.triangulate()
    surf_r = pv.PolyData(surf_r.points, faces=surf_r.cells)
    surf_r = surf_r.clean(tolerance=0.05)
    
    return surf, surf_r

def bounds_to_indicies(bounds, origin, spacing):
    """ Convert vtk bounds to array indicies.

    Args:
        bound (array) : bound of ROI
        origin (array) : origin of full image
        spacing (array) : spacing of full image

    Say you have a small box B inside a bigger box A, 
    and you want to get the indicies of A that
    correspond to B.

    "bounds" are the bounds of box B
    "origin" is the origin of box A
    "spacing" is the spacing of box A

    For reverse operation, see indicies_to_bounds.

    Note: originally from the surge_3 project.
    """

    # Shift bounding box to origin
    bounds = np.array(bounds).astype(float)
    origin = np.array(origin).astype(float)
    spacing = np.array(spacing).astype(float)

    bounds[0:2] -= origin[0]
    bounds[2:4] -= origin[1]
    bounds[4:6] -= origin[2]

    bounds = bounds.reshape(3,2)

    # Then scale mm to unit 1 (pixels)
    indicies = np.round(bounds / np.array(spacing).reshape(-1,1), 0)
    return indicies.flatten().astype(int)