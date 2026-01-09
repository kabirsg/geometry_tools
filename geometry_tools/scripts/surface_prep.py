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

def surface_prep(surf_file, proj_dir, surf_type):
    """ Basic surface prep.
    """
    surf_file = Path(surf_file)
    proj_dir = Path(proj_dir)
    if not proj_dir.exists():
        proj_dir.mkdir()
    
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
            mesher = Mesher(surf, include_aneurysms=True)
            anubool=True
        else:
            mesher = Mesher(surf) 
            anubool=False

        if surf_type=='a':
            mesher.clip_boundaries()
            mesher.set_inlets_outlets()
            mesher.pick_aneurysm()
            mesher.copy_structure() #this makes the surface mesh (mesher.surf) into a pv.PolyData object  
            # Select aneurysms
            s = cc.SelectGeodesic(mesher.surf)
            s.interact(title='Isolate aneurysms.')
            s.save_stored_points(neck_file_out)
            mesher.surf = s.mesh 
            mesher.generate_centerlines(include_aneurysms=True)
            mesher.surf, mesher.centerlines = vmtk.flow_extensions(mesher.surf, mesher.centerlines)
        else:
            accept = False
            while not accept:
                if not clipped_surf.exists(): 
                    mesher.clip_boundaries(method='box')
                else:
                    mesher.surf = pv.read(clipped_surf)
                mesher.surf.save(clipped_surf)
                mesher.set_inlets_outlets()
                mesher.generate_centerlines_multi(proj_dir)  
                plotter=pv.Plotter()
                plotter.add_mesh(mesher.surf,opacity=0.3)
                plotter.add_points(mesher.centerlines.points, color='red', render_points_as_spheres=True)
                plotter.add_axes()
                plotter.show()
                surf = vmtk.flow_ext(mesher.surf, mesher.centerlines, mesher.inlet_ids) #Adds inlet flow extension only
                extender = cc.Flow_Extender(pv.wrap(mesher.surf), mesher.centerlines,inlet_points=mesher.inlet_points, outlet_points=mesher.outlet_points) #Adds outlet flow extension only
                accept = extender.accept
            mesher.surf = extender.surf
            mesher.update_inlets_outlets()         

        mesher.surf.save(surf_file_out) #saves the clipped surface with extensions
        mesher.save_inlet_outlet_points(points_file_out, include_aneurysms=anubool, include_normals = True)

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
