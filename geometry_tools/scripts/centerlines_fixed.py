"""
This file includes a method to create vmtk centerlines for a surface mesh, which contains the VMTK
attributes.

Call this using:

centerlines_fixed.py prep_dir case_name

where prep_dir is the directory you are looking for the surface file in (should end in "cl.vtp"),
and the case_name is what you want your centerline to be called ("case_name_centerline_graph_vmtk.vtp")
"""

import numpy as np
import pyvista as pv
from geometry_tools.meshing import Mesher
from geometry_tools import vmtk_wrapper as vmtk
from scipy.spatial import cKDTree as KDTree 
from pathlib import Path
import sys

def make_cl(prep_dir, case_name, edge_length, iters, r, resample_step_length):
    out_dir = prep_dir.parent
    surf_file = sorted(prep_dir.glob('*cl.vtp'))[0]
    surf=pv.read(surf_file)
    remeshed_file = out_dir/(case_name+'_remeshed.vtp')
    graphed_cl_file = out_dir/(case_name+'_centerline_graph_vmtk.vtp')
    resampled_file =  out_dir/(case_name+'_centerline_resampled.vtp')
    if not remeshed_file.exists():
        surf = vmtk.surface_remeshing(surf, edgelength=edge_length, iterations = iters)
        surf.save(remeshed_file)
        
    m = Mesher(include_aneurysms=False)
    
    if not graphed_cl_file.exists():
        centerlines, graph = vmtk.network_extractor(surf, ratio = r)
        print(centerlines.cell_data)
        centerlines = vmtk.resample_cl(centerlines, length=0.2)
        # centerlines = vmtk.centerline_geometry(centerlines)
        # m.centerlines.save(out_dir/(case_name+'_centerline.vtp'))
        # m.centerlines = vmtk.centerlines_smooth(m.centerlines, iterations=1, sm_factor=0.1)
        graph.save(out_dir/(case_name+'_graph.vtp'))
        val = input("Is there a fenestration in this case? [y/n]: ")
        centerlines = vmtk.centerline_geometry(centerlines)
        centerlines.save(out_dir/(case_name+'_cl_centerline_graph_vmtk.vtp'))
        if val == 'y':
            print_next_step()
            sys.exit()
        
        #use vmtk for each segment
        m.centerlines = pv.PolyData()
        for idx in range(1, len(graph.points), 2):
            inlet_id = [idx-1]
            inlet_points = graph.points[inlet_id]
            #print(inlet_points)
            outlet_id = [idx]
            outlet_points = graph.points[outlet_id]

            #surf_capped = pv.PolyData()
            #surf_capped.copy_structure(vmtk.surface_capper(surf))
            tree = KDTree(surf.points)
            inlet_ids = [tree.query(i, k=1)[1] for i in inlet_points]
            outlet_ids = [tree.query(o, k=1)[1] for o in outlet_points]
            centerlines_seg = vmtk.centerlines(
	            surf, 
	            seed_selector='idlist', 
	            resampling_step_length = resample_step_length,
	            src_ids=inlet_ids,
	            target_ids=outlet_ids,
	            )
	        #make sure to set the first and last points to the inlet and outlet points
            centerlines_seg.points[0]=outlet_points
            centerlines_seg.points[-1]=inlet_points
            if idx == 1:
                centerline = centerlines_seg
            else:
                centerline += centerlines_seg
                
        centerline = vmtk.centerline_geometry(centerline)
        m.centerlines = centerline
        m.centerlines.save(graphed_cl_file)

    else:
        m.centerlines =  pv.read(graphed_cl_file)
        centerline_resampled = vmtk.resample_cl(m.centerlines, length=resample_step_length) #Was originally 1.5 from Anna
        centerline_resampled.save(resampled_file)

def print_next_step(): 
    print("Completed centerline generation")
    print("Next step: Map info")
    print("Usage: Gives parameters that can be looked at in Paraview")
    print("Scripts:\nmap_info_direct.py: For simpler cases - extracts metrics directly from the centerlines\t-->\tif this doesn't work go to map_info")
    print("map_info.py: For more complicated geometries with intersecting planes - Creates planes and extracts metrics based on the planes")
    print("map_info_bilateral.py: For cases with bilateral geometry")
    print("Command: map_info_direct.py [path/to/clipped/folder] [flowrate_at_inlet_1] [flowrate_at_inlet_2] False False False False False")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        try:
            import config
            prep_dir = Path(config.cf_prep_dir)
            print(f"Using the settings outlined in the config.py file. Prep Directory: {prep_dir}")
            case_name = config.cf_case_name
            edge_length = config.cf_edge_length
            iters = config.cf_iters
            ratio = config.cf_ratio
            resample_step_length = config.cf_resample_step_length
        except:
            prep_dir = Path("")
            case_name = ""
            edge_length = 0.4 #Edge length of the remeshed surface - Default is 0.4
            iters= 10 #Number of iterations for surface remeshing - Default is 10
            ratio = 1.2 #Ratio between the sphere step and the local maximum radius - Default not set, somewhere around 1.01 -> 1.2 is usually good
            resample_step_length = 1.5 #Primary parameter for density - Default is 1.5 -> places a centerline point every _ [units of the file] along the centerline
    else:
        prep_dir = Path(sys.argv[1])
        if prep_dir == "info": 
            print(f'Usage of centerlines_fixed.py file:\npython centerlines_fixed.py [path/to/prep/dir] [case_name] [optional: edge_length] [optional: number_of_iterations] [optional: ratio] [optional: resample_step-length]')
            sys.exit()
        case_name = sys.argv[2]
        edge_length = sys.argv[3]
        iters = sys.argv[4]
        ratio = sys.argv[5]
        resample_step_length = sys.argv[6]
    
    make_cl(prep_dir=prep_dir, case_name=case_name, edge_length=edge_length, iters=iters, r=ratio, resample_step_length=resample_step_length)
    print_next_step()