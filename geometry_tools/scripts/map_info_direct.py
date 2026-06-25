"""
This file contains a method for preparing a segmented surface mesh that has been run through 'surface_prep.py'
for creating the mappings for a PT surface mesh. Currently only does unilateral.

Call this file using:
map_info_direct.py [path/to/prep/dir] -sss [sss_flow_rate] -ss [ss_flow_rate] -l [labbe_flow_rate] -f [True/False] -s [sylvian_flow_rate] -e [emissary_flow_rate] -c [condylar_flow_rate]

Where
The values in the [square_brackets] are values that have to be filled in by the user (You!)
-prep_dir is the path to the directory your surface mesh from 'surface_prep.py' is stored
-sss_flow_rate is a float indicating the Superior Saggital Sinus flow rate at peak systole in mL/s
-ss_flow_rate is a float indicating the Straight Sinus flow rate at peak systole in mL/s
-labbe_flow_rate is a float indicating the Labbe flow rate at peak systole mL/s
-f indicates True or False if there is a fenestration (currently only set up for one)
-sylvian_flow_rate is a float indicating the Sylvian vein flow rate at peak systole mL/s
-emissary_flow_rate is a float indicating the Emissary vein outlet ratio
-condylar_flow_rate is a float indicating the Condylar vein outlet ratio

defaults to one flow rate for the whole geometry, which is 6.816019219 for peak systolic Superior Sinus inflow

This will produce a number of useful attributes, including the boundary layer width to get y+<1

"""
import numpy as np
import pyvista as pv
from geometry_tools.meshing import Mesher
import geometry_tools.vmtk_wrapper as vmtk
import geometry_tools.common as cc
from scipy.spatial import cKDTree as KDTree
from scipy.interpolate import interp1d
from pathlib import Path
import sys
import argparse

def define_fr(obj_pt, flowrate=5.578888889):
    cell_ids = obj_pt.surf.faces.reshape(-1, 4)[obj_pt.surf.cell_data[obj_pt.name]==1][:,1:]
    ids = cell_ids.flatten()
    obj_pt.surf.point_data[obj_pt.name][ids]=flowrate
    return obj_pt.surf

def mapped_info(prep_dir, sss, ss, lab, fen, syl, emissary, condylar):
    out_dir = prep_dir.parent
    #surf0_file = sorted(prep_dir.glob('*_noext.vtp'))[0]
    surf_file = sorted(prep_dir.glob('*_cl.vtp'))[0]
    remeshed_file = out_dir/(surf_file.stem  +'_remeshed.vtp')
    dec = str(int(round(float(sss) - int(float(sss)), 1)*10))
    #cent_graph_file = out_dir/(surf_file.stem  +'_centerline_graph_' + str(int(float(sss))) + 'p' + dec + '.vtp')
    #graphed_cl_file = out_dir/(surf_file.stem +'_centerline_graph.vtp')
    cent_graph_vmtk = out_dir/(surf_file.stem +'_centerline_graph_vmtk.vtp')
    cent_file = out_dir/(surf_file.stem + '__' + str(int(float(sss))) + 'p' + dec + 'centerline_mapped.vtp')
    mapped_file = out_dir/(surf_file.stem + '_' + str(int(float(sss))) + 'p' + dec + '_mappedsys.vtp')
    planes_files = out_dir/(surf_file.stem + '_planes_' + str(int(float(sss))) + 'p' + dec + '.vtm')
    if not mapped_file.exists():
        print("No mapped file")
        #surf = pv.read(surf0_file) #use unprepped surface for the centerline map
        if not remeshed_file.exists(): #use remeshed surface
            surf = vmtk.surface_remeshing(pv.read(surf_file), edgelength=0.5, iterations=5)
            surf.save(remeshed_file)
        else:
            surf=pv.read(remeshed_file)

        m = Mesher(
                surf,
                include_aneurysms=False
                )
        #m.clip_boundaries()
        #print(cent_file)
        if not cent_file.exists():
            print("Remaking centerlines")
            m.centerlines = pv.read(cent_graph_vmtk)
            m.centerlines = vmtk.resample_cl(m.centerlines, length=1.4) #so we don't have as many planes and points are equispaced
            #Use the centerline points to create planes
            planes = pv.MultiBlock()
            m.centerlines.point_data['CSA']=np.array(m.centerlines.n_points) #Initializing the numpy array
            m.centerlines.point_data['perimeter']=np.array(m.centerlines.n_points) #Initializing the numpy array
            points = m.centerlines.points
            normals = m.centerlines.point_data['FrenetTangent'] #Using the centerline point tangent as the normal for the plane 
            for ndx, pt in enumerate(points):
                plane=pv.Plane(center = pt, direction = normals[ndx], i_size=20, j_size=20, i_resolution=100, j_resolution=100).triangulate() #Creates a plane of triangular mesh elements - size is 20 just to be bigger than any vessel that we would use
                
                #Trying to clip with the default invert behaviour
                plane_split = plane.clip_surface(surf, invert=True)
                #Automation check: Is the original centerline point inside the resulting mesh bounds of the plane
                closest_pt_idx = plane_split.find_closest_point(pt)
                closest_pt = plane_split.points[closest_pt_idx] #Numpy array containing [x,y,z] coordinates of the closest point
                distance = np.linalg.norm(closest_pt - pt)
                if distance > 1e-4: #If the distance is less than this distance, then it is probably inside the plane boundaries
                    plane_split = plane.clip_surface(surf, invert=False) #Redo the plane clipping but inverted
                    print("inverting")
                
                split = plane_split.split_bodies() #Disconnects the planes if there are multiple (means that it intersected more than 1 branch)
                if len(split)>1:
                    cm = np.zeros((len(split),3))
                    for i in range(len(split)):
                        cm[i, :] = split[i].center_of_mass() #Returns the [x,y,z] coordinates of the center of mass for each split plane
                    tree = KDTree(cm) #KDTree for nearest neighbour lookup
                    _, j = tree.query(pt) # Plane with COM closest to the centerline point is the corresponding plane
                    plane_split = split[j] #Making sure the plane that will be used is only that one branch's plane
                
                CSsurf = plane_split.extract_surface()
                area = CSsurf.area
                edges = CSsurf.extract_feature_edges(boundary_edges=True, non_manifold_edges=False, feature_edges=False, manifold_edges=False)    
                #p=pv.Plotter()
                #p.add_mesh(CSsurf)
                #p.add_mesh(edges, color='red')
                #p.show()
                sized = edges.compute_cell_sizes()
                #print(sized)
                perimeter = sum(sized['Length'])
                m.centerlines.point_data['CSA'][ndx]=area
                m.centerlines.point_data['perimeter'][ndx]=perimeter
                CSsurf.point_data['centerline_id']=ndx
                planes.append(CSsurf)
            m.centerlines.save(cent_file)
            planes.save(planes_files)
        else:
            print("Found existing mapped file - continuing using this")
            m.centerlines = pv.read(cent_file)
            planes = pv.read(planes_files)
        #Create mapping to surface
        #Label segments
        #How many main segments do we have? Assuming only unilateral wiht current options
        main_segs=1
        branches = []
        if ss != 'False':
            branches.append('Straight_Sinus')
            main_segs += 1
        if lab != 'False':
            branches.append('Labbe')
            main_segs += 1
        if syl != 'False':
            branches.append('Sylvian_Vein')
            main_segs += 1
        if emissary != 'False':
            branches.append('Emissary_Vein')
            main_segs += 1
        if condylar != 'False':
            branches.append('Condylar_Vein')
            main_segs += 1 

        #Some parts of the centerline shouldn't be used to calculate parameters, so take these bits out
        #While we are at it, identify the other branches for later 
        labelled_centerlines = cc.Label_CLs(m.surf, m.centerlines, main_segs, branches, fen)
        m.centerlines.point_data['unusable']=labelled_centerlines.centerline.point_data['branch_centerlines']
        for b in branches: #get all labelled branch centerline points
            m.centerlines.point_data[b]=labelled_centerlines.centerline.point_data[b]
        if fen == 'True':
            m.centerlines.point_data['fen1']=labelled_centerlines.centerline.point_data['fen1']
            m.centerlines.point_data['fen2']=labelled_centerlines.centerline.point_data['fen2']
        m.centerlines.point_data['main_branch']=labelled_centerlines.centerline.point_data['main_branch']

        #don't need to do this next line anymore since we remeshed the surface already
        usable_ids = np.asarray(np.where(m.centerlines.point_data['unusable']==0))[0]
        usable_CL=m.centerlines.points[usable_ids]
        usable_CL_CSA=m.centerlines.point_data['CSA'][usable_ids]
        usable_CL_perimeter=m.centerlines.point_data['perimeter'][usable_ids]

        tree = KDTree(usable_CL) #only include usable centerlines
        _, idx_c = tree.query(m.surf.points) #get usable centerline points closest to surf points
        
        #assign the CSA at that centerline point to the surface
        m.surf.point_data['CSA']=m.centerlines.point_data['CSA'][idx_c]
        m.surf.point_data['perimeter']=m.centerlines.point_data['perimeter'][idx_c]
        
        #print(np.isnan(np.sum(m.surf.point_data['CSA'])), np.isnan(np.sum(m.surf.point_data['perimeter'])))
        m.surf, neighbour_pts = cc.smooth_mesh_data_local_alt(m.surf, array='CSA', iterations = 10)
        m.surf, _ = cc.smooth_mesh_data_local_alt(m.surf, array='perimeter', neighbour_pt_ids = neighbour_pts, iterations = 10)
        #print(np.isnan(np.sum(m.surf.point_data['CSA'])), np.isnan(np.sum(m.surf.point_data['perimeter'])))
        
        m.surf.save(mapped_file)
        m.centerlines.save(cent_file)

    surf=pv.read(mapped_file) #for some reason have to read in again. boolean array dimension error. Fix this.
    
    m = Mesher(
            surf,
            include_aneurysms=False,
            )
    m.centerlines=pv.read(cent_file)
    planes = pv.read(planes_files)
    #Select flowrate regions assuming unilateral
    flowrate = float(sss) #Superior Saggital Sinus at Peak systolic
    m.surf.point_data['flowrate'] = np.zeros(m.surf.n_points)
    m.centerlines.point_data['flowrate']=np.zeros(m.centerlines.n_points)
    #Superior Saggital Sinus branch:
    #Use the first main branch segment to define SSS flowrate
    main_branch_seg=1
    m.centerlines.point_data['flowrate'][m.centerlines.point_data['main_branch']==main_branch_seg]=flowrate
    if ss != 'False':
        m.centerlines.point_data['flowrate'][m.centerlines.point_data['Straight_Sinus']==1]=float(ss)
        flowrate += float(ss) #this is the current flowrate of the main branch
        main_branch_seg +=1
        m.centerlines.point_data['flowrate'][m.centerlines.point_data['main_branch']==main_branch_seg]=flowrate
    if lab !='False':
        m.centerlines.point_data['flowrate'][m.centerlines.point_data['Labbe']==1]=float(lab)
        flowrate += float(lab) #this is the current flowrate of the main branch
        main_branch_seg +=1
        m.centerlines.point_data['flowrate'][m.centerlines.point_data['main_branch']==main_branch_seg]=flowrate
    if syl !='False':
        m.centerlines.point_data['flowrate'][m.centerlines.point_data['Sylvian_Vein']==1]=float(syl)
        flowrate += float(syl) #this is the current flowrate of the main branch
        main_branch_seg +=1
        m.centerlines.point_data['flowrate'][m.centerlines.point_data['main_branch']==main_branch_seg]=flowrate
    if emissary !='False':
        m.centerlines.point_data['flowrate'][m.centerlines.point_data['Emissary_Vein']==1]=flowrate*float(emissary)
        flowrate -=flowrate*float(emissary) #this is the current flowrate of the main branch
        main_branch_seg +=1
        m.centerlines.point_data['flowrate'][m.centerlines.point_data['main_branch']==main_branch_seg]=flowrate
    if condylar !='False':
        m.centerlines.point_data['flowrate'][m.centerlines.point_data['Condylar_Vein']==1]=flowrate*float(condylar)
        flowrate -=flowrate*float(condylar) #this is the current flowrate of the main branch
        main_branch_seg +=1
        m.centerlines.point_data['flowrate'][m.centerlines.point_data['main_branch']==main_branch_seg]=flowrate
    if fen != 'False':
        #get average CSA ratio between branches:
        fen1_CSA = np.mean(m.centerlines.point_data['CSA'][m.centerlines.point_data['fen1']==1])
        fen2_CSA = np.mean(m.centerlines.point_data['CSA'][m.centerlines.point_data['fen2']==1])
        ratio1 = fen1_CSA/(fen1_CSA+fen2_CSA)
        ratio2 = fen2_CSA/(fen1_CSA+fen2_CSA)
        m.centerlines.point_data['flowrate'][m.centerlines.point_data['fen1']==1]=ratio1*m.centerlines.point_data['flowrate'][m.centerlines.point_data['fen1']==1]
        m.centerlines.point_data['flowrate'][m.centerlines.point_data['fen2']==1]=ratio2*m.centerlines.point_data['flowrate'][m.centerlines.point_data['fen2']==1]
    
    #get the dP value at every point on the entire centerline
    U_c = m.centerlines.point_data['flowrate']/m.centerlines.point_data['CSA']
    rho = 1057
    m.centerlines.point_data['dP']=0.5*rho*(3/2*U_c)**2/133.322 #dP based on max centerline velocity in mmHg calculated in every branch

    usable_ids = np.asarray(np.where(m.centerlines.point_data['unusable']==0))[0]
    usable_CL_flowrate=m.centerlines.point_data['flowrate'][usable_ids]

    tree3 = KDTree(m.centerlines.points[usable_ids]) #only include usable centerline points
    _, idx_c3 = tree3.query(m.surf.points)

    m.surf.point_data['flowrate']=m.centerlines.point_data['flowrate'][idx_c3]
    #smooth data
    m.surf, _ = cc.smooth_mesh_data_local_alt(m.surf, array='flowrate', iterations = 5)

    nu = (0.0037/1057) #viscosity
    L = 4*m.surf.point_data['CSA']/m.surf.point_data['perimeter']*0.001#Hydraulic diameter (m)
    Deff=2*np.sqrt(m.surf.point_data['CSA']/(np.pi))*0.001 #Effective diameter (m)
    U = (m.surf.point_data['flowrate']/m.surf.point_data['CSA']) #peak systolic velocity in m/s
    Re = U*L/nu
    Cf = 0.026/(Re**(1/7))
    Tw_rho=Cf*(U**2)/2
    Uf=np.sqrt(Tw_rho)
    m.surf.point_data['Dh']=L
    m.surf.point_data['Deff']=Deff
    m.surf.point_data['mean_velocity']=U
    m.surf.point_data['kolmog_len']=((nu**3)*L/(U**3))**(1/4)
    m.surf.point_data['taylor_len']=np.sqrt(10)*(Re**(1/4))*(((nu**3)*L/(U**3))**(1/4))
    m.surf.point_data['ds_max(y+=1)']=nu/Uf #  
    m.surf.point_data['dP']=0.5*rho*(3/2*U)**2/133.322 #dP based on max centerline velocity in mmHg calculated in every branch
    m.centerlines.save(cent_file) 
    m.surf.save(mapped_file)

if __name__ == "__main__":
    # prep_dir = Path(sys.argv[1]) 
    # print(type(prep_dir))
    # if len(sys.argv)>3:
    #     sss = sys.argv[2]
    #     ss=sys.argv[3]
    #     lab=sys.argv[4]
    #     fen=sys.argv[5]
    #     syl= sys.argv[6]
    #     emissary = sys.argv[7]
    #     condylar = sys.argv[8]
    # else:
    #     sss = sys.argv[2]#6.816019219
    #     ss='False'
    #     lab='False'
    #     fen='False'
    #     syl = 'False'
    #     emissary='False'
    #     condylar = 'False'
    use_parser = True
    if len(sys.argv) == 1:
        try:
            import config
            prep_dir = Path(config.mid_prep_dir)
            print(f"Using the settings outlined in the config.py file - prep directory: {prep_dir}")
            sss = config.mid_sss
            ss = config.mid_ss
            labbe = config.mid_lab
            fen = config.mid_fen
            syl = config.mid_syl
            emi = config.mid_emissary
            con = config.mid_condylar
            use_parser = False
        except Exception as e:
            print(f"Error: {e}")
            print("Failed to use config file for settings - switching to using the arguments provided")
    if use_parser:
        parser = argparse.ArgumentParser(description="Arguments for map_info_direct.\nClip folder location must be provided\nFor all present vessels, an inlet flow rate must be provided. If the vessel is not in the model, either don't enter it's flag or enter False for that vessel")
        parser.add_argument('prep_dir', help='Location to the clip folder (output of surface prep file)')
        parser.add_argument('-sss', '--superior-saggital-sinus', required=False, type=float, default=6.816019219, help='Flow rate through the inlet superior saggital sinus (the peak systolic inlet flow rate)')
        parser.add_argument('-ss', '--sigmoid-sinus', required=False, default='False', help='Flow rate through the sigmoid sinus')
        parser.add_argument('-l', '--labbe', required=False, default='False', help='Flow Rate through Labbe')
        parser.add_argument('-sy', '--sylvian-vein', required=False, default='False', help='Flow rate through the Sylvian Vein')
        parser.add_argument('-e', '--emissary-vein', required=False,  default='False', help='Flow rate through Emissary Vein')
        parser.add_argument('-c', '--condylar-vein', required=False, default='False', help='Flow rate through the condylar vein')
        parser.add_argument('-f', '--fenestration-stenosis', required=False, default='False', help='Boolean (True or False) indicating the presence of a fenestration stenosis')
        args = parser.parse_args()
        mapped_info(prep_dir=Path(args.prep_dir), sss = args.superior_sigmoid_sinus, ss = args.sigmoid_sinus, lab = args.labbe, fen = args.fenestration_stenosis, syl = args.sylvian_vein, emissary=args.emissary_vein, condylar=args.condylar_vein)    
    else:
        mapped_info(prep_dir=prep_dir,sss=sss,ss=ss,lab=labbe,fen=fen,syl=syl,emissary=emi,condylar=con)