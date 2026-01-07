#!/usr/bin/env python3
"""
This is a script that processes a segmented mesh by cutting off the endcaps on the vessel(s) and 
adding flow extensions. Needed if you intend to use any of the scripts in this directory.

Call this script using:
surface_prep.py surf_file proj_dir surf_type

Where
-surf_file is the stl of your segmented geometry (eg. surf.stl)
-proj_dir is the name of the directory you want your files to be stored in
-surf_type is pt or a for PT or aneurysm case respectively

"""

from pathlib import Path 
import pyvista as pv 
from geometry_tools.meshing import Mesher
from geometry_tools import common as cc
from geometry_tools import vmtk_wrapper as vmtk
import time
from datetime import timedelta
import sys 
import numpy as np

def surface_prep(surf_file, proj_dir, surf_type):
    """ Basic surface prep.
    """
    surf_file = Path(surf_file)
    proj_dir = Path(proj_dir)
    if not proj_dir.exists():
        proj_dir.mkdir()
    
    '''
    surf_output_dir = proj_dir / '01_clipped'  
    points_output_dir = proj_dir / '01_points' 
    neckpoints_output_dir = proj_dir / '01_neckpoints' 

    for f in [surf_output_dir, points_output_dir, neckpoints_output_dir]:
        if not f.exists():
            f.mkdir(parents=True)
    '''
    # Output files
    surf_file_out = proj_dir / (surf_file.stem + '_cl.vtp')
    clipped_surf = proj_dir / (surf_file.stem + '_noext.vtp')
    if surf_type=='a':
        neck_file_out = proj_dir / (surf_file.stem + '_cl_neckpoints.vtm')
    #else:
        #we may want to choose some points to identify some important pt features (eg torcula)
    points_file_out = proj_dir / (surf_file.stem + '_cl_endpoints.vtm') #Multiblock object

    if not surf_file_out.exists():
        case_start = time.time()

        surf = pv.read(surf_file)
        surf = surf.compute_normals(auto_orient_normals=False)
        if surf_type=='a': 
            m = Mesher(surf, include_aneurysms=True)
            anubool=True
        else:
            m = Mesher(surf) 
            anubool=False

        if surf_type=='a':
            m.clip_boundaries()
            m.set_inlets_outlets()
            m.pick_aneurysm()
            m.copy_structure() #this makes the surface mesh (m.surf) into a pv.PolyData object  
            # Select aneurysms
            s = cc.SelectGeodesic(m.surf)
            s.interact(title='Isolate aneurysms.')
            s.save_stored_points(neck_file_out)
            m.surf = s.mesh 
            m.generate_centerlines(include_aneurysms=True)
            m.surf, m.centerlines = vmtk.flow_extensions(m.surf, m.centerlines)
        else:
            accept = False
            while not accept:
                if not clipped_surf.exists(): 
                    m.clip_boundaries(method='box')
                else:
                    m.surf = pv.read(clipped_surf)
                m.surf.save(clipped_surf)
                m.set_inlets_outlets()
                m.generate_centerlines_multi(proj_dir)  
                p=pv.Plotter()
                p.add_mesh(m.surf,opacity=0.3)
                p.add_points(m.centerlines.points, color='red', render_points_as_spheres=True)
                p.add_axes()
                p.show()
                surf = vmtk.flow_ext(m.surf, m.centerlines, m.inlet_ids)
                extender = cc.Flow_Extender(pv.wrap(surf), m.centerlines,inlet_points=m.inlet_points, outlet_points=m.outlet_points)
                accept = extender.accept
            m.surf = extender.surf
            m.update_inlets_outlets()         

        m.surf.save(surf_file_out) #saves the clipped surface with extensions
        m.save_inlet_outlet_points(points_file_out, include_aneurysms=anubool, include_normals = True)

        time_spent = time.time() - case_start
        print('Case done', timedelta(seconds=time_spent))

    else:
        print('Output file exists.')


if __name__ == "__main__":
    if len(sys.argv) == 1: 
        surf_file = "/home/kabir/PT/PTSeg106_v8/PTSeg106_base_0p64.stl"
        proj_dir = "/home/kabir/PT/PTSeg106_v8/PTSeg106_clip"
        surf_type = 'pt' #options: a or pt
    else:
        surf_file = sys.argv[1]
        proj_dir = sys.argv[2]
        if len(sys.argv) > 3:
            surf_type = sys.argv[3] #options: a or pt
        else:
            surf_type = 'pt' #default is a pulsatile tinnitus surface file
    surface_prep(surf_file, proj_dir, surf_type)
