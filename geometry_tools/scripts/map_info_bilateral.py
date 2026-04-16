"""
This file contains a method for preparing a segmented surface mesh that has been run through 'surface_prep.py'
for creating the mappings for a PT surface mesh. Bilateral models.

Call this file using:
map_info_bilateral.py prep_dir [-sss] [-ss] [-lab] [-fen] [-syl] [-emissary] [-condylar]

Example:
map_info_bilateral.py prep_dir -sss 6.8 -ss 1.7 r -lab 1.2 1.4 -fen False True -emissary 0.1 False -condylar False 0.2

NOTE that these are keyword arguments, so calling this function requires the -arg before the argument list, and is slightly different
than the other scripts. Arguments in brackets are optional, and the defaults can be seen at the bottom of this script.

Where
-prep_dir is the directory your surface mesh from 'surface_prep.py' is stored. This is not an optional argument
-sss is a float indicating the Straight Sinus flow rate at peak systole in mL/s
-ss is a list of strings indicating the Straight Sinus flow rate at peak systole in mL/s or False if there is no ss and whether it is to the left (l), right (r), or at the junction (j) of the sss
-lab is a list of two float/strings indicating the Labbe flow rate at peak systole mL/s or False if there is no Labbe. Left Right.
-fen is a list of two strings indicating True or False if there is a fenestration (currently only set up for one on both sides). Left Right.
-syl is a list of two strings indicating the Sylvian vein flow rate at peak systole mL/s or False if there is no Sylvian. Left Right.
-emissary is a list of two strings indicating the Emissary vein outlet ratios or False if there is no emissary vein. Left Right.
-condylar is a float indicating the Condylar vein outlet ratios or False if there is no condylar vein. Left Right.

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

def mapped_info(prep_dir, sss, ss, split_flow, lab, fen, syl, emissary, condylar):
    out_dir = prep_dir.parent
    dec = str(int(round(float(sss) - int(float(sss)), 1)*10))
    #surf0_file = sorted(prep_dir.glob('*_noext.vtp'))[0]
    surf_file = sorted(prep_dir.glob('*_cl.vtp'))[0]
    remeshed_file = out_dir/(surf_file.stem  +'_remeshed.vtp')
    cent_graph_file = out_dir/(surf_file.stem  +'_graph.vtp')
    cent_graph_vmtk = out_dir/out_dir/(surf_file.stem +'_centerline_graph_vmtk.vtp')
    cent_file = out_dir/(surf_file.stem + '__' + str(int(float(sss))) + 'p' + dec + 'centerline_mapped.vtp')
    #newcent_file = out_dir/(surf_file.stem + '_centerline_cm.vtp')
    mapped_file = out_dir/(surf_file.stem + '_' + str(int(float(sss))) + 'p' + dec + '_mappedsys.vtp')
    planes_files = out_dir/(surf_file.stem + '_planes_' + str(int(float(sss))) + 'p' + dec + '.vtm')
    if not mapped_file.exists():
        #surf = pv.read(surf0_file) #use unprepped surface for the centerline map
        if not remeshed_file.exists(): #use remeshed surface for the centerline map
            surf = vmtk.surface_remeshing(pv.read(surf_file), edgelength=0.5, iterations=5)
            surf.save(remeshed_file)
        else:
            surf=pv.read(remeshed_file)
        m = Mesher(
                surf,
                include_aneurysms=False
                )

        if not cent_file.exists():  
            m.centerlines = pv.read(cent_graph_vmtk)     
            '''
            _ , graph = vmtk.network_extractor(m.surf)#vmtk.centerline_geometry(m.centerlines)
            graph.save(cent_graph_file)
            #use vmtk for each segment
            centerline = pv.PolyData()
            for idx in range(1, len(graph.points), 2):
                    inlet_id = [idx-1]
                    m.inlet_points = graph.points[inlet_id]
                    outlet_id = [idx]
                    m.outlet_points = graph.points[outlet_id]
                    m.generate_centerlines(include_aneurysms=False)
                    if idx == 1:
                        centerline = m.centerlines
                    else:
                        centerline += m.centerlines
            m.centerlines = centerline
            #m.centerlines = vmtk.centerline_geometry(m.centerlines)
            '''
            m.centerlines = vmtk.resample_cl(m.centerlines, length=1.5) #so we don't have as many planes and points are equispaced
            #Use the centerline points to create planes
            planes = pv.MultiBlock()
            m.centerlines.point_data['CSA']=np.array(m.centerlines.n_points)
            m.centerlines.point_data['perimeter']=np.array(m.centerlines.n_points)
            points = m.centerlines.points
            normals = m.centerlines.point_data['FrenetTangent'] 
            for ndx, pt in enumerate(points):
                plane=pv.Plane(center = pt, direction = normals[ndx], i_size=20, j_size=20, i_resolution=100, j_resolution=100).triangulate()
                plane_split = plane.clip_surface(surf)
                split = plane_split.split_bodies()
                if len(split)>1:
                    cm = np.zeros((len(split),3))
                    for i in range(len(split)):
                        cm[i, :] = split[i].center_of_mass()
                    tree = KDTree(cm)
                    _, j = tree.query(pt) #closest center of mass to the centerline point
                    plane_split = split[j]
                CSsurf = plane_split.extract_surface() 
                #new_centerline.points[ndx]=plane_split.center_of_mass()
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
            #new_centerline.save(newcent_file)
        else:
            m.centerlines = pv.read(cent_file)
            planes = pv.read(planes_files)
        #Create mapping to surface

        #Label segments
        #How many main segments do we have? Assuming anything to the left of the SSS is 'left'
        #Left segments:
        main_segs_l=1
        branches_l = []        
        if lab[0] != 'False':
            branches_l.append('Labbe_l')
            main_segs_l += 1
        if syl[0] != 'False':
            branches_l.append('Sylvian_Vein_l')
            main_segs_l += 1
        if emissary[0] != 'False':
            branches_l.append('Emissary_Vein_l')
            main_segs_l += 1
        if condylar[0] != 'False':
            branches_l.append('Condylar_Vein_l')
            main_segs_l += 1 

        #Right segments:
        main_segs_r=1 
        branches_r = ['Superior_Saggital'] #automatically add the SSS  
        if ss != 'False':
            branches_r.append('Straight_Sinus') #assume there is always a junction with the SSS
            main_segs_r += 1     
        if lab[1] != 'False':
            branches_r.append('Labbe_r')
            main_segs_r += 1
        if syl[1] != 'False':
            branches_r.append('Sylvian_Vein_r')
            main_segs_r += 1
        if emissary[1] != 'False':
            branches_r.append('Emissary_Vein_r')
            main_segs_r += 1
        if condylar[1] != 'False':
            branches_r.append('Condylar_Vein_r')
            main_segs_r += 1 

        #Some parts of the centerline shouldn't be used to calculate parameters, so take these bits out
        #While we are at it, identify the other branches for later 
        print("Labelling the Left Side - anything to the Left of the Superior Saggital Sinus.")
        labelled_centerlines_l = cc.Label_CLs(m.surf, m.centerlines, main_segs_l, branches_l, fen[0])
        print("Labelling the Right Side - anything to the Right of the Superior Saggital Sinus, inclusive.")
        labelled_centerlines_r = cc.Label_CLs(m.surf, m.centerlines, main_segs_r, branches_r, fen[1])
        m.centerlines.point_data['unusable']=labelled_centerlines_l.centerline.point_data['branch_centerlines']+labelled_centerlines_r.centerline.point_data['branch_centerlines']
        for b in branches_l: #get all labelled branch centerline points on left
            m.centerlines.point_data[b]=labelled_centerlines_l.centerline.point_data[b]
        for b in branches_r: #get all labelled branch centerline points on right
            m.centerlines.point_data[b]=labelled_centerlines_r.centerline.point_data[b]
        if fen[0] == 'True':
            m.centerlines.point_data['fen1_l']=labelled_centerlines_l.centerline.point_data['fen1']
            m.centerlines.point_data['fen2_l']=labelled_centerlines_l.centerline.point_data['fen2']
        if fen[1] == 'True':
            m.centerlines.point_data['fen1_r']=labelled_centerlines_r.centerline.point_data['fen1']
            m.centerlines.point_data['fen2_r']=labelled_centerlines_r.centerline.point_data['fen2']
        m.centerlines.point_data['main_branch_l']=labelled_centerlines_l.centerline.point_data['main_branch']
        m.centerlines.point_data['main_branch_r']=labelled_centerlines_r.centerline.point_data['main_branch']

        #don't need to do this next line anymore since we remeshed the surface already
        #m.surf = pv.read(surf_file) #replace surface with flow extension surface to avoid the flow extension issues

        '''
        usable_CL=m.centerlines.points[m.centerlines.point_data['unusable']==0]
        usable_CL_CSA=m.centerlines.point_data['CSA'][m.centerlines.point_data['unusable']==0]
        usable_CL_perimeter=m.centerlines.point_data['perimeter'][m.centerlines.point_data['unusable']==0]
        tree = KDTree(usable_CL) #only include the usable centerlines
        _, idx_c = tree.query(m.surf.points)
        m.surf.point_data['CSA']=usable_CL_CSA[idx_c]
        m.surf.point_data['perimeter']=usable_CL_perimeter[idx_c]
        '''
        usable_CL=m.centerlines.points[m.centerlines.point_data['unusable']==0]
        usable_CL_CSA=m.centerlines.point_data['CSA'][m.centerlines.point_data['unusable']==0]
        usable_CL_perimeter=m.centerlines.point_data['perimeter'][m.centerlines.point_data['unusable']==0]
        usable_ids = np.asarray(np.where(m.centerlines.point_data['unusable']==0))[0]
        usable_planes = pv.MultiBlock()
        for i in usable_ids:
            usable_planes.append(planes[i])
        #usable_planes.save(out_dir/(surf_file.stem + '_usable_planes.vtm'))
        merged_usable_planes=usable_planes.combine()   
        planes_points = merged_usable_planes.points
        
        tree = KDTree(planes_points) #only include usable planes
        _, idx_p = tree.query(m.surf.points) #get plane points closest to surf points

        cntr_ids = merged_usable_planes.point_data['centerline_id'][idx_p].astype(int) #centerline ids corresponding to the plane on the surface
        m.surf.point_data['CSA']=m.centerlines.point_data['CSA'][cntr_ids]
        m.surf.point_data['perimeter']=m.centerlines.point_data['perimeter'][cntr_ids]

        #if wonky planes were deleted, we need to remove the data at those centerline points and replace with weighted average data between the two neighbouring points
        cntr_ids_avg = np.asarray([x for x in range(len(m.centerlines.points)) if x not in cntr_ids.tolist()])
        if cntr_ids_avg.size != 0:
            tree_ctr_avg = KDTree(planes_points) #look at the closest planes
            dist_avg, cind = tree_ctr_avg.query(m.centerlines.points[cntr_ids_avg], k=2) #get two closest points on planes
            #inverse distance average
            m.centerlines.point_data['CSA'][cntr_ids_avg]=((1/dist_avg[:, 0])*m.centerlines.point_data['CSA'][merged_usable_planes.point_data['centerline_id'][cind[:, 0]].astype(int)]+(1/dist_avg[:, 1])*m.centerlines.point_data['CSA'][merged_usable_planes.point_data['centerline_id'][cind[:, 1]].astype(int)])/((1/dist_avg[:, 0])+(1/dist_avg[:, 1]))
            m.centerlines.point_data['perimeter'][cntr_ids_avg]=((1/dist_avg[:, 0])*m.centerlines.point_data['perimeter'][merged_usable_planes.point_data['centerline_id'][cind[:, 0]].astype(int)]+(1/dist_avg[:, 1])*m.centerlines.point_data['perimeter'][merged_usable_planes.point_data['centerline_id'][cind[:, 1]].astype(int)])/((1/dist_avg[:, 0])+(1/dist_avg[:, 1]))

        m.surf, _ = cc.smooth_mesh_data_local(m.surf, array='CSA', func=np.mean, iterations = 2)
        m.surf, _ = cc.smooth_mesh_data_local(m.surf, array='perimeter', func=np.mean, iterations = 2)
        m.surf.save(mapped_file)
        m.centerlines.save(cent_file)
    #else:
    surf=pv.read(mapped_file) #for some reason have to read in again. boolean array dimension error. Fix this.
    m = Mesher(
            surf,
            include_aneurysms=False,
            )
    m.centerlines=pv.read(cent_file)
    planes = pv.read(planes_files)
    #Select flowrate regions assuming biilateral
    flowrate = sss #Superior Saggital Sinus at Peak systolic
    m.surf.point_data['flowrate'] = np.zeros(m.surf.n_points)
    m.centerlines.point_data['flowrate']=np.zeros(m.centerlines.n_points)
    #Superior Saggital Sinus branch:
    #Define SSS flowrate
    m.centerlines.point_data['flowrate'][m.centerlines.point_data['Superior_Saggital']==1]=flowrate
    
    main_branch_seg_l=1
    main_branch_seg_r=1 
    flowrate_l=flowrate*split_flow[0]
    flowrate_r=flowrate*split_flow[1]
    m.centerlines.point_data['flowrate'][m.centerlines.point_data['main_branch_l']==main_branch_seg_l]=flowrate_l
    m.centerlines.point_data['flowrate'][m.centerlines.point_data['main_branch_r']==main_branch_seg_r]=flowrate_r

    if ss[0] != 'False':
        m.centerlines.point_data['flowrate'][m.centerlines.point_data['Straight_Sinus']==1]=float(ss[0])
        if ss[1]=='l': 
            flowrate_l += float(ss[0])
            main_branch_seg_l +=1
            m.centerlines.point_data['flowrate'][m.centerlines.point_data['main_branch_l']==main_branch_seg_l]=flowrate_l
        elif ss[1]=='r':
            flowrate_r += float(ss[0])
            main_branch_seg_r +=1
            m.centerlines.point_data['flowrate'][m.centerlines.point_data['main_branch_r']==main_branch_seg_r]=flowrate_r
        elif ss[1]=='j':
            flowrate_l += float(ss[0])*split_flow[0]
            m.centerlines.point_data['flowrate'][m.centerlines.point_data['main_branch_l']==main_branch_seg_l]=flowrate_l
            flowrate_r += float(ss[0])*split_flow[1]
            m.centerlines.point_data['flowrate'][m.centerlines.point_data['main_branch_r']==main_branch_seg_r]=flowrate_r
    
    #Going left
    if lab[0] !='False':
        m.centerlines.point_data['flowrate'][m.centerlines.point_data['Labbe_l']==1]=float(lab[0])
        flowrate_l += float(lab[0]) #this is the current flowrate of the main branch
        main_branch_seg_l +=1
        m.centerlines.point_data['flowrate'][m.centerlines.point_data['main_branch_l']==main_branch_seg_l]=flowrate_l
    if syl[0] !='False':
        m.centerlines.point_data['flowrate'][m.centerlines.point_data['Sylvian_Vein_l']==1]=float(syl[0])
        flowrate_l += float(syl[0]) #this is the current flowrate of the main branch
        main_branch_seg_l +=1
        m.centerlines.point_data['flowrate'][m.centerlines.point_data['main_branch_l']==main_branch_seg_l]=flowrate_l
    if emissary[0] !='False':
        m.centerlines.point_data['flowrate'][m.centerlines.point_data['Emissary_Vein_l']==1]=flowrate_l*float(emissary[0])
        flowrate_l -=flowrate_l*float(emissary[0]) #this is the current flowrate of the main branch
        main_branch_seg_l +=1
        m.centerlines.point_data['flowrate'][m.centerlines.point_data['main_branch_l']==main_branch_seg_l]=flowrate_l
    if condylar[0] !='False':
        m.centerlines.point_data['flowrate'][m.centerlines.point_data['Condylar_Vein_l']==1]=flowrate_l*float(condylar[0])
        flowrate_l -=flowrate_l*float(condylar[0]) #this is the current flowrate of the main branch
        main_branch_seg_l +=1
        m.centerlines.point_data['flowrate'][m.centerlines.point_data['main_branch_l']==main_branch_seg_l]=flowrate_l
    if fen[0] != 'False':
        #get average CSA ratio between branches:
        fen1_CSA = np.mean(m.centerlines.point_data['CSA'][m.centerlines.point_data['fen1_l']==1])
        fen2_CSA = np.mean(m.centerlines.point_data['CSA'][m.centerlines.point_data['fen2_l']==1])
        ratio1 = fen1_CSA/(fen1_CSA+fen2_CSA)
        ratio2 = fen2_CSA/(fen1_CSA+fen2_CSA)
        m.centerlines.point_data['flowrate'][m.centerlines.point_data['fen1_l']==1]=ratio1*m.centerlines.point_data['flowrate'][m.centerlines.point_data['fen1_l']==1]
        m.centerlines.point_data['flowrate'][m.centerlines.point_data['fen2_l']==1]=ratio2*m.centerlines.point_data['flowrate'][m.centerlines.point_data['fen2_l']==1]

    #Going right
    if lab[1] !='False':
        m.centerlines.point_data['flowrate'][m.centerlines.point_data['Labbe_r']==1]=float(lab[1])
        flowrate_r += float(lab[1]) #this is the current flowrate of the main branch
        main_branch_seg_r +=1
        m.centerlines.point_data['flowrate'][m.centerlines.point_data['main_branch_r']==main_branch_seg_r]=flowrate_r
    if syl[1] !='False':
        m.centerlines.point_data['flowrate'][m.centerlines.point_data['Sylvian_Vein_r']==1]=float(syl[1])
        flowrate_r += float(syl[1]) #this is the current flowrate of the main branch
        main_branch_seg_r +=1
        m.centerlines.point_data['flowrate'][m.centerlines.point_data['main_branch_r']==main_branch_seg_r]=flowrate_r
    if emissary[1] !='False':
        m.centerlines.point_data['flowrate'][m.centerlines.point_data['Emissary_Vein_r']==1]=flowrate_r*float(emissary[1])
        flowrate_r -=flowrate_r*float(emissary[1]) #this is the current flowrate of the main branch
        main_branch_seg_r +=1
        m.centerlines.point_data['flowrate'][m.centerlines.point_data['main_branch_r']==main_branch_seg_r]=flowrate_r
    if condylar[1] !='False':
        m.centerlines.point_data['flowrate'][m.centerlines.point_data['Condylar_Vein_r']==1]=flowrate_r*float(condylar[1])
        flowrate_r -=flowrate_r*float(condylar[1]) #this is the current flowrate of the main branch
        main_branch_seg_r +=1
        m.centerlines.point_data['flowrate'][m.centerlines.point_data['main_branch_r']==main_branch_seg_r]=flowrate_r
    if fen[1] != 'False':
        #get average CSA ratio between branches:
        fen1_CSA = np.mean(m.centerlines.point_data['CSA'][m.centerlines.point_data['fen1_r']==1])
        fen2_CSA = np.mean(m.centerlines.point_data['CSA'][m.centerlines.point_data['fen2_r']==1])
        ratio1 = fen1_CSA/(fen1_CSA+fen2_CSA)
        ratio2 = fen2_CSA/(fen1_CSA+fen2_CSA)
        m.centerlines.point_data['flowrate'][m.centerlines.point_data['fen1_r']==1]=ratio1*m.centerlines.point_data['flowrate'][m.centerlines.point_data['fen1_r']==1]
        m.centerlines.point_data['flowrate'][m.centerlines.point_data['fen2_r']==1]=ratio2*m.centerlines.point_data['flowrate'][m.centerlines.point_data['fen2_r']==1]
    
    #get the dP value at every point on the entire centerline
    U_c = m.centerlines.point_data['flowrate']/m.centerlines.point_data['CSA']
    rho = 1057
    m.centerlines.point_data['dP']=0.5*rho*(3/2*U_c)**2/133.322 #dP based on max centerline velocity in mmHg calculated in every branch
    
    '''
    usable_CL=m.centerlines.points[m.centerlines.point_data['unusable']==0]
    usable_CL_flowrate=m.centerlines.point_data['flowrate'][m.centerlines.point_data['unusable']==0]
    tree3 = KDTree(usable_CL) #only include the usable centerlines
    _, idx_c3 = tree3.query(m.surf.points)
    m.surf.point_data['flowrate']=usable_CL_flowrate[idx_c3]
    '''
    usable_ids = np.asarray(np.where(m.centerlines.point_data['unusable']==0))[0]
    usable_planes = pv.MultiBlock()
    for i in usable_ids:
        usable_planes.append(planes[i])
    merged_usable_planes=usable_planes.combine()      
    planes_points = merged_usable_planes.points
    usable_CL_flowrate=m.centerlines.point_data['flowrate'][m.centerlines.point_data['unusable']==0]

    tree3 = KDTree(planes_points) #only include usable planes
    _, idx_p3 = tree3.query(m.surf.points)

    m.surf.point_data['flowrate']=m.centerlines.point_data['flowrate'][merged_usable_planes.point_data['centerline_id'][idx_p3].astype(int)]

    #smooth data
    m.surf, _ = cc.smooth_mesh_data_local(m.surf, array='flowrate', func=np.mean, iterations = 5)
    m.surf, _ = cc.smooth_mesh_data_local(m.surf, array='CSA', func=np.mean, iterations = 5)
    m.surf, _ = cc.smooth_mesh_data_local(m.surf, array='perimeter', func=np.mean, iterations = 5)

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
    m.surf.point_data['taylor_len']=np.sqrt(15)*(Re**(1/4))*(((nu**3)*L/(U**3))**(1/4))
    m.surf.point_data['ds_max(y+=1)']=nu/Uf # 
    m.surf.point_data['dP']=0.5*rho*(3/2*U)**2/133.322 #dP based on max centerline velocity in mmHg calculated in every branch  
    m.centerlines.save(cent_file)
    m.surf.save(mapped_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bilateral Mapping for Meshing')
    parser.add_argument('-prep_dir', dest = 'prep_dir', required = True,
                        help='The directory where the prepped files are stored.')
    parser.add_argument('-sss', dest = 'sss', type=float, default = 6.816019219,
                        help='The inlet flowrate for the Superior Saggital Sinus at peak systole.')
    parser.add_argument('-ss', dest='ss', type=str, nargs="*", default= ['False', 'r'],
                        help='The inlet flowrate for the Straight Sinus at peak systole, and whether it is to the left or right of the SSS.')
    parser.add_argument('-split', dest='split', nargs="*", type=float, default=[0.5, 0.5],
                        help='The flow split at the SSS. Left Right.')    
    parser.add_argument('-lab', dest='lab', nargs="*", type=str, default=['False', 'False'],
                        help='The inlet flowrate(s) for the Labbe vein(s) at peak systole. Left Right.')
    parser.add_argument('-fen', dest='fen', nargs="*", type=str, default=['False','False'],
                        help='Whether there is a fenestration (on either side). Left Right.')
    parser.add_argument('-syl', dest='syl', nargs="*", type=str, default=['False', 'False'],
                        help='The inlet flowrate(s) for the Sylvian vein(s) at peak systole. Left Right.')
    parser.add_argument('-emissary', dest='emissary', nargs="*", type=str, default=['False', 'False'],
                        help='The outlet flow split(s) for the Emissary vein(s) at peak systole. Left Right.')
    parser.add_argument('-condylar', dest='condylar', nargs="*", type=str, default=['False', 'False'],
                        help='The outlet flow split(S) for the Condylar vein(s) at peak systole. Left Right.')
    args = parser.parse_args()

    mapped_info(prep_dir=Path(args.prep_dir), sss= args.sss, ss = args.ss, split_flow=args.split, lab = args.lab, fen = args.fen, syl = args.syl, emissary = args.emissary, condylar = args.condylar)
