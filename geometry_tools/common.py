import vtk 
import numpy as np 
import pyvista as pv 
from scipy.spatial import cKDTree as KDTree 
from scipy.interpolate import interp1d
import warnings
        
# from pathlib import Path 
# import h5py 
# import ast 

def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return '%s:%s: %s: %s\n' % (filename, lineno, category.__name__, message)


def fix_vmtk_group_ids(surf):
    """ Fix group IDs.

    Args:
        surf (PolyData) : Surface with array GroupIds (optional: Mask)

    Returns:
        PolyData with fixed GroupIds

    VMTK groups IDs are often buggy, this fixes them
    based on connectivity.
    """
    import pygeodesic.geodesic as geodesic
    print(surf.point_data)
    g_ids = np.unique(surf.point_data['GroupIds'])

    # Break into pieces, find which have broken ids
    masks = [surf.point_data['GroupIds'] == g for g in g_ids]
    groups = [surf.extract_points(m) for m in masks]

    # Find which id has most mutual with sac, overwrite
    if 'Mask' in surf.point_data:
        check_sac = [g.point_data['Mask'].sum()/g.n_points for g in groups]
        surf.point_data['GroupIds'][surf.point_data['Mask'] == 1] = g_ids[np.argmax(check_sac)]

    # Break into pieces, find which have broken ids
    masks = [surf.point_data['GroupIds'] == g for g in g_ids]
    groups = [surf.extract_points(m) for m in masks]
    n_parts = np.array([g.split_bodies().n_blocks for g in groups])

    # Broken groups:
    split_idx = [idx for idx, x in enumerate(n_parts > 1) if x == True]
    split_g_ids = g_ids[split_idx]

    if len(split_idx) > 0:

        tree = KDTree(surf.points)
        surf.point_data['GroupError'] = np.zeros(surf.n_points, dtype=int)

        for idx in split_idx:
            parts = list(groups[idx].split_bodies())
            small_parts = parts[1:]
            error_points = np.concatenate([x.points for x in small_parts], axis=0)

            _, ii = tree.query(error_points)
            surf.point_data['GroupError'][ii] = 1

        target_indices = [idx for idx, x in enumerate(surf.point_data['GroupError'] == 1) if x == True]
        source_indices = [idx for idx, x in enumerate(surf.point_data['GroupError'] == 0) if x == True]

        target_indices = np.array(target_indices)
        source_indices = np.array(source_indices)

        geoalg = geodesic.PyGeodesicAlgorithmExact(surf.points, surf.faces.reshape(-1, 4)[:, 1:])
        distances, best_source = geoalg.geodesicDistances(source_indices, target_indices)

        surf.point_data['GroupIds'][target_indices] = surf.point_data['GroupIds'][source_indices[best_source]]
    return surf 

def vtk_generate_img_stencil(mesh, spacing=0.05, bounds=None):
    """ Resample surf mesh to image.

    This function has memory problems, something isn't freed at end,
    so running it in a loop causes problems.
    """
    if bounds is None:
        bounds = np.array(mesh.bounds)
    else:
        bounds = np.array(bounds)

    bounds_lengths = np.diff(bounds.reshape(3,2)).T[0]

    # Inflate bounds
    inflate_lens = bounds_lengths * 0.05
    bounds[::2] -= inflate_lens
    bounds[1::2] += inflate_lens
    bounds_lengths = np.diff(bounds.reshape(3,2)).T[0]

    spacing = (spacing, spacing, spacing)
    dimensions = (bounds_lengths / spacing).astype(int)
    dims = (dimensions - 1)
    origin = bounds[::2]

    image = pv.UniformGrid() 
    image.dimensions = dimensions 
    image.origin = origin
    image.spacing = spacing
    # image.SetScalarType(vtk.VTK_UNSIGNED_CHAR,image.GetInformation())
    # image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)    # 3
    image.point_data['ImageScalars'] = np.zeros(np.prod(image.dimensions), dtype=bool)

    pol2Stenc = vtk.vtkPolyDataToImageStencil()
    pol2Stenc.SetTolerance(0.5) 
    pol2Stenc.SetInputData(mesh)
    pol2Stenc.SetInformationInput(image)
    pol2Stenc.Update()

    stencil = vtk.vtkImageStencil()
    stencil.SetInputData(image)
    stencil.ReverseStencilOn()
    stencil.SetBackgroundValue(1)
    stencil.SetStencilData(pol2Stenc.GetOutput())
    stencil.Update()

    newImage = pv.wrap(stencil.GetOutput())
    newImage.point_data['ImageScalars'] = newImage.point_data['ImageScalars']#[:,0]
    
    return newImage

def vtk_taubin_smooth(mesh, pass_band=0.1, feature_angle=60.0, iterations=20):
    """ Smooth mesh using Taubin method. 
    
    Note:
    This also exists in bsl.common. 
    """
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(mesh) 
    smoother.SetNumberOfIterations(iterations)
    smoother.BoundarySmoothingOff()
    smoother.FeatureEdgeSmoothingOff() 
    smoother.SetFeatureAngle(feature_angle)
    smoother.SetPassBand(pass_band)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    # smoother.GenerateErrorScalarsOn() s
    smoother.Update()
    return pv.wrap(smoother.GetOutput())


def smooth_mesh_data(data_array, points, radius, func=np.mean, mask=None):
    """ Smooth the data, not the mesh. 

    Radius-based moving average filter.
    Incorporate distance weighting?
    """
    tree = KDTree(points)
    inds = [tree.query_ball_point(pt, radius) for pt in points]
    out_array = np.zeros_like(data_array)

    if mask is None:
        mask = np.ones(len(points))

    for idx in range(len(data_array)):
        value = func(data_array[inds[idx]])
        
        if np.any(mask[inds[idx]] == 1):
            out_array[idx] = value 
        else:
            out_array[idx] = data_array[idx]

    return out_array

def smooth_mesh_data_local(surf, array='GroupIds', 
        func='median', neighbour_pt_ids=None, iterations=1):
    """ Smooth mesh data based on local connectivity.

    Args:
        surf (polydata): Input surface.
        array (str): Name of array to be smoothed.
    Returns:
        surf (polydata): Surface with smoothed array
        neighbour_pt_ids (list of lists): list of neighbouring point ids,
            index by point id.
    """
    if neighbour_pt_ids == None:
        neighbour_pt_ids = get_neighbour_map(surf) 
    
    neighbour_pt_ids = np.array(neighbour_pt_ids).astype(int)

    surf = surf.copy()
    new_array = surf.point_data[array].copy()

    for idx in range(iterations):
        old_array = new_array.copy()
        for pt_id in list(range(surf.n_points)):
            # Uses 2 connexity by default
            neighbours = neighbour_pt_ids[neighbour_pt_ids[pt_id]]
            neighbours = np.unique([item for sublist in neighbours for item in sublist])
            #print(neighbours)
            # neighbours = neighbour_pt_ids[pt_id]

            # Unfortunately, np.median has the undesired "fallback" 
            # that uses the mean when the array is even. This is bad for 
            # data like GroupIds.
            if func == 'median':
                num_neighbours = len(neighbours)
                neighbour_vals = np.sort(old_array[neighbours])
                center_index = int(np.ceil(num_neighbours / 2) - 1)
                new_val = neighbour_vals[center_index]

            else:
                new_val = func(old_array[neighbours])

            new_array[pt_id] = new_val                        

    surf.point_data[array] = new_array

    return surf, neighbour_pt_ids

def smooth_mesh_data_local_alt(surf, array='GroupIds', neighbour_pt_ids=None, iterations=1):
    """ Smooth mesh data based on local connectivity.
        Uses an inverse distance weighted averaging.

    Args:
        surf (polydata): Input surface.
        array (str): Name of array to be smoothed.
    Returns:
        surf (polydata): Surface with smoothed array
        neighbour_pt_ids (list of lists): numpy array of neighbouring point ids
    """
    if neighbour_pt_ids == None:
        neighbour_pt_ids = get_neighbour_map_alt(surf) #return list of numpy arrays
    surf = surf.copy()
    new_array = surf.point_data[array].copy()
    
    for idx in range(iterations):
        old_array = new_array.copy()
        for pt_id, neigh in enumerate(neighbour_pt_ids):
            if neigh.size>0:
                neighbours = neigh.T
                distance = np.linalg.norm(surf.points[neighbours]-surf.points[pt_id], axis = 1)
                new_val = np.average(surf.point_data[array][neighbours], weights = 1/distance)
                new_array[pt_id] = new_val                        

    surf.point_data[array] = new_array

    return surf, neighbour_pt_ids

# def get_neighbour_map_broken(surf):#, n_points):
#     """ Get full list of adjacent neighbour pts.

#     Args:
#         cells (array): Cell connectivity, shape (n_cells, 3).
#         n_pts (int): Number of point ids.
    
#     Returns:
#         neighbour_pt_ids (list): List of lists containing neighbour pt ids.
#         * Trying with numpy array.
#     """
#     # neighbour_pt_ids = [[] for _ in range(surf.n_points)]
#     edges = surf.extract_all_edges()

#     # edges.points does not neccesarily == surf.points!!
#     # create a map between them
#     tree = KDTree(surf.points)
#     _, ii = tree.query(edges.points, k=1)

#     ee = edges.lines.reshape(-1, 3)[:,1:]
    
#     ee = ee[np.argsort(ee[:, 0])]
    
#     diff = np.diff(ee[:,0])
    
#     upper = np.argwhere(diff) + 1
#     upper = np.concatenate([upper.flatten(), [len(ee)]], axis=0)
#     lower = np.roll(upper,1)
#     lower[0] = 0

#     max_connect = np.diff(upper).max()
#     index = np.zeros((len(upper), 2), dtype=int)
#     neighbour_pt_ids = np.empty((surf.n_points, max_connect), dtype=np.int)
#     neighbour_pt_ids.fill(np.nan)

#     index[:, 0] = lower.flatten()
#     index[:, 1] = upper.flatten()
#     # index[-1] = [upper[-1], len(ee)]    

#     for e, i in enumerate(index): #(surf.n_points):
#         sub = ee[i[0]:i[1]]
#         unique = np.unique(sub[:,1])
#         # print(unique)
#         neighbour_pt_ids[ii[e]][:len(unique)] = unique
#         # neighbour_pt_ids[ii[unique]] = e
#         for u in unique:
#             neighbour_pt_ids[ii[u]] = e

#     neighbour_pt_ids = [x[~np.isnan(x)] for x in neighbour_pt_ids]
#     neighbour_pt_ids = np.array([np.unique(x) for x in neighbour_pt_ids], dtype='object')

#     return neighbour_pt_ids

def get_neighbour_map(surf):#, n_points):
    """ Get full list of adjacent neighbour pts.

    Args:
        cells (array): Cell connectivity, shape (n_cells, 3).
        n_pts (int): Number of point ids.
    
    Returns:
        neighbour_pt_ids (list): List of lists containing neighbour pt ids.
        * Trying with numpy array.
    """
    neighbour_pt_ids = [[] for _ in range(surf.n_points)]
    edges = surf.extract_all_edges()

    # edges.points does not neccesarily == surf.points!!
    # create a map between them
    tree = KDTree(surf.points)
    _, ii = tree.query(edges.points, k=1)
    #print(surf.n_points, ii.shape)
	
    ee = edges.lines.reshape(-1, 3)[:,1:]
    
    for e in ee:
        neighbour_pt_ids[ii[e[0]]].append(ii[e[1]])
        neighbour_pt_ids[ii[e[1]]].append(ii[e[0]])

    neighbour_pt_ids = np.array([np.unique(x) for x in neighbour_pt_ids], dtype='object')
    #print(neighbour_pt_ids)
    return neighbour_pt_ids

def get_neighbour_map_alt(surf):
    neighbour_pt_ids = [] #empty list for storing numpy arrays
    
    for ind in range(surf.n_points):
        pts = []
        pcids = point_cell_ids(surf, ind)
        for cell in pcids:
            cell_pts = pv.vtk_id_list_to_array(surf.GetCell(cell).GetPointIds())
            pts.extend([i for i in cell_pts if i != ind])
        neighbour_pt_ids.append(np.array(list(set(pts))))
    return neighbour_pt_ids #list of numpy arrays
        
def point_cell_ids(surf, ind):
    ids = vtk.vtkIdList()
    surf.GetPointCells(ind, ids)
    return [ids.GetId(i) for i in range(ids.GetNumberOfIds())]

def create_edge_size_array(surf, fix_centerline, min_edge_size=0.1, max_edge_size=0.4, sac_size=0.15, misr_min=0.1, misr_max=2.5, name='Size', ref_edge_ratio=0.8):
    """ Create "Size" array incorporating distance to centerlines and curvature.

    This will likely be refined moving forward.
    Optional:
        Based on DistanceToCenterlinesArray, interpolate between 2.5 mm rad as max, 0.5 mm rad min
        OR
        Based on Dan's method (fix_centerline='fix'): map the vertices to the centerline points, then use a proportion of 
        the MISR as the edge length at that point.
    Based on Curvature, interpolate between 0.3 as min, 0.8 as max
    Based on Mask, set to min value where Mask == 1.
    Then take min of each.

    Refinement regions incorporated:
    Assume everywhere within the refinement region will be assigned an edge length that is 0.8x what it would be with regular remeshing
    """
    distance_interp = interp1d([misr_min, misr_max], [min_edge_size, max_edge_size], 
        kind='linear',
        bounds_error=False,
        fill_value=(min_edge_size, max_edge_size),
        )
    curv_interp = interp1d([0.3, 0.8], [max_edge_size, min_edge_size], 
        kind='linear',
        bounds_error=False,
        fill_value=(max_edge_size, min_edge_size),
        )
    
    if (fix_centerline == 'fix') or (fix_centerline == 'network_fix'):
        surf.point_data['SizeMISR'] = distance_interp(surf.point_data['misr'])
        surf.point_data[name] = surf.point_data['SizeMISR']
    else:
        surf = surf.compute_normals()
        surf.point_data['SizeDistanceToCenterlinesArray'] = distance_interp(surf.point_data['DistanceToCenterlinesArray']) #np.ones(surf.n_points) 

        # The perfectly straight flow extensions end up having high curvature 
        # unless they are perturbed slightly
        # Some PT meshes have areas of really high curvature that need extra refinement
        surf_perturb = surf.copy()
        perturbed_vec = np.einsum(
            'ij,i->ij', 
            surf_perturb.point_data['Normals'], 
            np.random.normal(0, 0.0001, surf_perturb.n_points)
            )
        surf_perturb.points = surf_perturb.points + perturbed_vec
        surf_perturb.point_data['Curvature'] = np.abs(surf_perturb.curvature('Minimum'))
        
        surf.point_data['Curvature'] = surf_perturb.point_data['Curvature'] #np.abs(surf.curvature('Minimum'))
        surf.point_data['SizeCurvature'] = curv_interp(surf.point_data['Curvature']) #np.ones(surf.n_points)
        
        surf.point_data[name] = np.minimum(surf.point_data['SizeDistanceToCenterlinesArray'], surf.point_data['SizeCurvature'])

    surf, n_ids = smooth_mesh_data_local(surf, name, np.mean, iterations=2)

    if 'Mask' in surf.point_data:
        # First dilate mask to include nearby regions
        surf.point_data['MaskDilate'] = surf.point_data['Mask'].copy()
        surf, _ = smooth_mesh_data_local(surf, 'MaskDilate', np.max, iterations=6, neighbour_pt_ids=n_ids)

        sac_mask = surf.point_data['MaskDilate'] == 1
        # sac_size_array = sac_size * np.ones(len(sac_mask))
        current_size_array = surf.point_data[name][sac_mask]

        surf.point_data[name][sac_mask] = np.minimum(sac_size, current_size_array)
    else:
        print('No mask defined in create_edge_size_array.')

    if ('RefinementPoints' in surf.point_data) and ('Enlarge_Cells' in surf.point_data):
        print('Refinement points and enlarged cells defined in create_edge_size_array.')
        surf.point_data['Refs'] = surf.point_data['RefinementPoints'].copy()
        surf, _ = smooth_mesh_data_local(surf, 'Refs', np.max, iterations=6, neighbour_pt_ids=n_ids)

        ref_reg = surf.point_data['Refs'] == 1
        current_size_array = surf.point_data[name][ref_reg]

        surf.point_data[name][ref_reg] = current_size_array*ref_edge_ratio

        surf.point_data['nds'] = surf.point_data['Enlarge_Cells'].copy()
        nd_reg = surf.point_data['nds'] == 1
        current_size_array = surf.point_data[name][nd_reg]

        surf.point_data[name][nd_reg] = current_size_array*1.5

    elif 'RefinementPoints' in surf.point_data:
        print('Refinement points defined in create_edge_size_array.')
        surf.point_data['Refs'] = surf.point_data['RefinementPoints'].copy()
        surf, _ = smooth_mesh_data_local(surf, 'Refs', np.max, iterations=6, neighbour_pt_ids=n_ids)

        ref_reg = surf.point_data['Refs'] == 1
        current_size_array = surf.point_data[name][ref_reg]

        surf.point_data[name][ref_reg] = current_size_array*ref_edge_ratio
    elif 'Enlarge_Cells' in surf.point_data:
        print('Enlarged cells defined in create_edge_size_array.')
        surf.point_data['nds'] = surf.point_data['Enlarge_Cells'].copy()
        nd_reg = surf.point_data['nds'] == 1
        current_size_array = surf.point_data[name][nd_reg]

        surf.point_data[name][nd_reg] = current_size_array*1.5
    else:
        print('No non-dominant and/or refinement region defined in create_edge_size_array.')

    surf, n_ids = smooth_mesh_data_local(surf, name, np.mean, iterations=3)

    return surf


class RefinementSelection():
    """ Interactively create a refinement region by selecting cells on the mesh
    and then obtaining the points they are associated with and assigning them a 
    point_array (eg. 'RefinementPoints') which contains a boolean if those cells were
    picked. 
    
    Later, these booleans will be used to assign a target edge length that will 
    remesh the surface, and ultimately the volumetric mesh.
    """ 
    def __init__(self, surf, name = 'RefinementPoints', title='Select Refinement Zone'):
        self.surf = surf
        self.title = title
        self.name = name
        
    def select(self):
        self.surf.cell_data[self.name] = np.zeros(self.surf.n_cells, dtype=bool)
        select = ClickDragSelect(self.surf, title = self.title)
        self.surf.cell_data[self.name]=select.mesh.cell_data['PickedMask']

    def define_surface(self):
        self.surf.point_data[self.name] = np.zeros(self.surf.n_points, dtype=bool)
        cell_ids = self.surf.faces.reshape(-1, 4)[self.surf.cell_data[self.name]==1][:,1:]
        ids = cell_ids.flatten()
        self.surf.point_data[self.name][ids]=1

class RefinementSelection_OLD():
    """ 
    Still used!!
    Interactively create a refinement region by clipping away 
    parts of a surface mesh that are not required to be refined, then storing a boolean at 
    the selected surface points on the original mesh. 
    
    Later, these booleans will be used to assign a target edge length that will 
    remesh the surface, and ultimately the volumetric mesh.
    """ 
    def __init__(self, surf, name = 'RefinementPoints', title='Clip Refinement Zone'):
        self.surf = surf.fill_holes(100)
        self.title = title
        self.name = name
        self.select()
        #self.define_surface()
        
    def select(self):
        #Define the surface we want to refine by clipping the surface
        warnings.formatwarning = warning_on_one_line
        warnings.warn("Holes from clipping must be fillable. May result in inability to close surface!")
        new_surf = ClickDragDelete(self.surf, title=self.title)
        temprefsurf=new_surf.mesh.triangulate()
        if type(temprefsurf) != pv.core.pointset.PolyData:
                temprefsurf = pv.PolyData(temprefsurf.points, temprefsurf.cells)
        self.temprefsurf=temprefsurf.fill_holes(100).clean()
        self.temprefsurf = self.temprefsurf.connectivity(largest=True)

    def define_surface(self):
        self.refsurf=self.surf.select_enclosed_points(self.temprefsurf, tolerance=0.01)
        self.pts = self.surf.extract_points(self.refsurf['SelectedPoints'].view(bool),
                           adjacent_cells=False)
        
        p = pv.Plotter()
        p.add_text('Red points are the selected refinement region', position='upper_left')
        p.add_mesh(self.surf,color = 'blue', style='wireframe', show_edges=True)
        p.add_points(self.pts, color='r')
        p.show()
        self.surf.point_data[self.name]=self.refsurf.point_data['SelectedPoints']

class Label_CLs():
    """ 
    Identify parts of the centerline that shouldn't be used to calculate parameters based 
    on CSA and perimeter.
    Also label the other branch points
    """ 
    def __init__(self, surf, centerline, main_segs, branches, fen, name = 'branch_centerlines'):
        #split up the centerline into branches
        self.surf = surf 
        self.centerline=centerline
        self.main_segs=main_segs
        self.branches=branches
        self.fen=fen
        self.name = name
        self.centerline.point_data[self.name]=np.zeros(self.centerline.n_points)
        self.centerline.cell_data[self.name]=np.zeros(self.centerline.n_cells)
        self.tree=KDTree(self.centerline.points)
        self.iter_branches()
        self.get_main_branch_points()
        self.select()
        self.identify_points()

        p=pv.Plotter()
        p.add_mesh(self.surf, opacity=0.3, color='grey')
        p.add_mesh(self.centerline, scalars=self.name)
        p.show()

    def iter_branches(self):
        for b in self.branches:
            self.get_branch_points(branch=b, title= 'Select '+b+' segment')  
        if self.fen != 'False':
            self.get_branch_points(branch='fen1', title= 'Select first side of fenestration')
            self.get_branch_points(branch='fen2', title= 'Select second side of fenestration')  
        #also want to select any other other "branches" of the centerline we don't want included in the calcs
        select = ClickDragSelect(self.centerline, title='Select extraneous branches to exclude')
        self.centerline.cell_data[self.name] = select.mesh.cell_data['PickedMask']   

    def get_main_branch_points(self):
        #Note: if there is a fenestration, include it all in the same branch segment
        self.centerline.cell_data['main_branch'] = np.zeros(self.centerline.n_cells, dtype=int)
        self.centerline.point_data['main_branch'] = np.zeros(self.centerline.n_points, dtype=int)
        #identify the cells
        if self.main_segs == 1: #only one branch
            self.centerline.cell_data['main_branch']=1
        else:
            for seg in range(self.main_segs):
                select = ClickDragSelect(self.centerline, title = 'Select main branch segment #'+str(seg+1))
                self.centerline.cell_data['main_branch'] += int(seg+1)*select.mesh.cell_data['PickedMask']
        #identify the points
        self.centerline = self.centerline.cell_data_to_point_data()

    def get_branch_points(self, branch='Labbe', title='Select Labbe segment'):
        self.centerline.cell_data[branch] = np.zeros(self.centerline.n_cells, dtype=bool)
        self.centerline.point_data[branch] = np.zeros(self.centerline.n_points, dtype=bool)
        #identify the cells
        select = ClickDragSelect(self.centerline, title = title)
        self.centerline.cell_data[branch]=select.mesh.cell_data['PickedMask']
        #identify the points when main branch points identified

    def select(self):
        #select a junction point on the main branch for each small branch
        #create sphere at point based on center (point) and hydraulic diameter
        for b in self.branches:
            pl = pv.Plotter()
            pl.add_mesh(self.centerline)
            if b != 'Straight_Sinus':
                #if both of these veins are in the list, then you'll need to select planes
                if ('Trolard_Vein' not in self.branches) and ('Sylvian_Vein' not in self.branches):
                    pl.enable_point_picking(callback = self.point_cb, show_message="Press P to select junction point for " + b)
                    pl.show()
                else:
                    print('Trolard vein and Sylvian vein present! Need to select planes to delete manually!!')
            else:
                print('Straight sinus present! Need to select planes to delete manually!!')
        '''
        warnings.formatwarning = warning_on_one_line
        warnings.warn("Holes from clipping must be fillable. May result in inability to close surface!")
        new_surf = ClickDragDelete(self.surf, title='Clip off small branches')
        temprefsurf=new_surf.mesh.triangulate()
        if type(temprefsurf) != pv.core.pointset.PolyData:
                temprefsurf = pv.PolyData(temprefsurf.points, temprefsurf.cells)
        self.temprefsurf=temprefsurf.fill_holes(100).clean()
        self.temprefsurf = self.temprefsurf.connectivity(largest=True)
        '''
    def point_cb(self, pt):
        _, pt_id = self.tree.query(pt)
        Dh = 4*self.centerline.point_data['CSA'][pt_id]/self.centerline.point_data['perimeter'][pt_id]
        sphere = pv.Sphere(radius=1.2*Dh/2, center=pt)
        temp=self.centerline.select_enclosed_points(sphere, tolerance=0.01)
        self.centerline.point_data[self.name] += temp.point_data['SelectedPoints']

    def identify_points(self):
        #reset all of these back to what they should be
        '''
        if 'Straight_Sinus' in self.branches:
            #if there is a straight sinus in a unilateral model, then there is a torcula, and if there is a torcula,
            #the centerlines there are wonky.
            #self.centerline.point_data[self.name][self.centerline.point_data['main_branch']>1]=0
            self.centerline.point_data[self.name][self.centerline.point_data['Straight_Sinus']==1]=0
        else:'''
        self.centerline.point_data[self.name][self.centerline.point_data['main_branch']!=0]=0
        #make sure that the fenestration points also aren't affected by the sphere of unusable points
        if self.fen != 'False':
            self.centerline.point_data[self.name][self.centerline.point_data['fen1']!=0]=0
            self.centerline.point_data[self.name][self.centerline.point_data['fen2']!=0]=0

class SacSelectTool():
    """ Interactively mark points using a probe.

    I think this is obsolete? 
    """ 
    def __init__(self, surf):
        self.surf = surf
        self.surf.point_data['Mask'] = np.zeros(self.surf.n_points)
        self.surf.point_data['TempMask'] = np.zeros(self.surf.n_points)

    def mask(self, center, radius):
        sphere = pv.Sphere(radius=radius, center=center)
        self.surf = self.surf.select_enclosed_points(sphere)
        mask_index = self.surf.point_data['SelectedPoints']
        ids = [x for x in range(self.surf.n_points) if mask_index[x] == True]
        self.selection = self.surf.extract_points(ids)

        self.selection.point_data['vtkOGIds'] = self.selection.point_data['vtkOriginalPointIds'].copy()
        self.selection = self.selection.extract_largest()
        mask = self.selection.point_data['vtkOGIds']
        self.selection = pv.PolyData(self.selection.points, self.selection.cells)
        self.selection.point_data['vtkOGIds'] = mask
        self.selection = self.selection.clean()

        self.surf.point_data['TempMask'] = self.surf.point_data['Mask'].copy()
        self.surf.point_data['TempMask'][self.selection.point_data['vtkOGIds']] = 1
            
    def select(self):

        def sphere_cb(xyz, probe):
            # Select enclosed points
            self.sphere = pv.Sphere(radius=probe.GetRadius(), center=xyz)
            # self.probe = probe
            self.p.add_mesh(self.sphere, color='r', opacity=0.4, name='probe')
            self.center = probe.GetCenter()
            self.radius = probe.GetRadius()
            self.mask(self.center, self.radius)
            self.p.add_mesh(self.surf, scalars='TempMask', name='surf')

        def choose_cb():
            self.surf.point_data['Mask'] = self.surf.point_data['TempMask']
            
        self.p = pv.Plotter()
        self.p.add_mesh(self.surf, scalars='TempMask', opacity=1.0, name='surf')
        self.p.add_sphere_widget(
            callback=sphere_cb, 
            center=np.mean(self.surf.points, axis=0), 
            radius=4.0, 
            pass_widget=True,
            theta_resolution=8,
            phi_resolution=8,
            style='wireframe',
            )
        self.p.add_key_event('space', choose_cb)
        self.p.show()

        if np.all(self.surf.point_data['Mask'] == 0):
            self.surf.point_data['Mask'] = self.surf.point_data['TempMask']

    

class SelectGeodesic():
    def __init__(self, mesh, scalars='Mask'):
        mesh = mesh.clean()
        mesh = mesh.compute_normals(auto_orient_normals=True)
        self.mesh = mesh
        self.scalars = scalars
        if self.scalars not in mesh.point_data:
            self.mesh.point_data[self.scalars] = np.zeros(self.mesh.n_points)

        self.current_mask = np.zeros_like(self.mesh.point_data[self.scalars])
        self.current_pts = self.mesh.points 
        
        self.stored_points = []
        self.picked_points = []
        self.picked_ids = []
        self.lines = []
        self.interactive = False
        self.tree = KDTree(self.mesh.points)
        
    def interact(self, title='Isolate aneurysms.'):
        self.interactive = True
        self.p = pv.Plotter() 
        self.mesh = self.mesh.compute_normals()
        self.p.add_mesh(self.mesh, name='mesh', scalars=self.scalars, cmap='Reds', show_edges=True)
        self.p.enable_point_picking(
            show_point=True,
            show_message=False,
            callback=self._cb,
            color='red',
            font_size=12,
            point_size=15,
            tolerance=0.005
            )

        self.p.add_text(title, position='upper_left', font_size=18)
        msg = 'Keys:'
        self.p.add_text(msg, position=(0.05, 175), font_size=12)   
        msg = 'f: select points'
        self.p.add_text(msg, position=(0.05, 150), font_size=12)
        msg = 'u: undo'
        self.p.add_text(msg, position=(0.05, 125), font_size=12)
        msg = 'space: complete loop'
        self.p.add_text(msg, position=(0.05, 100), font_size=12)
        msg = 'a: append mask'
        self.p.add_text(msg, position=(0.05, 75), font_size=12)
        msg = 'x: smooth section'
        self.p.add_text(msg, position=(0.05, 50), font_size=12)
        msg = 'q: quit'
        self.p.add_text(msg, position=(0.05, 25), font_size=12)

        self.p.add_key_event('u', self._undo)
        self.p.add_key_event('space', self._finish)
        self.p.add_key_event('a', self.append)
        self.p.add_key_event('x', self.refill_section)
        self.p.add_key_event('c', self._clear)
        self.p.show()
        self.stored_points.append(np.array(self.picked_points))

        return self._get_points()

    def _cb(self, pt):
        """ CB for picking points. """
        self.picked_points.append(pt)
        self.update_points()
        self.update_geodesic()
        self.display()

    def _undo(self):
        """ Remove last picked point and update display. """
        if len(self.picked_points) > 0:
            self.picked_points.pop()
            self.picked_ids.pop() 
        self.update_points()
        self.update_geodesic()
        self.display()

    def _clear(self):
        """ Clear all current picked points and update display. """
        self.picked_points = []
        self.picked_ids = []
        self.lines = []
        self.update_points()
        self.update_geodesic()
        self.display()

    def _finish(self):
        """ Close current loop and update display."""
        self.picked_points.append(self.picked_points[0])
        self.picked_ids.append(self.picked_ids[0])
        self.update_geodesic()
        self.display()
        self.update_mesh()

    def append(self):
        """ Store current geodesic, start a new geodesic. """
        # Stored existing mask array
        # When calling update_mesh, logical or with existing
        self.current_mask = self.mesh.point_data[self.scalars].copy() 
        self.current_pts = self.mesh.points 

        self.stored_points.append(self.picked_points)

        self.picked_points = []
        self.picked_ids = []
        self.lines = []
        if self.interactive:
            self.display()

    def update_points(self):
        """ Gets ids of picked points. """
        if len(self.picked_points) > 0:
            _, self.picked_ids = self.tree.query(self.picked_points, k=1)
            self.picked_ids = list(self.picked_ids)
        else:
            self.picked_ids = []
        # Clean duplicates 
        _, idx = np.unique(self.picked_ids, return_index=True)
        # self.picked_ids = list(np.array(self.picked_ids)[idx])
        # self.picked_points = list(np.array(self.picked_points)[idx])

    def update_geodesic(self):
        """ Updates geodesic using current picked points. """
        if len(self.picked_ids) > 1:
            pairwise = zip(self.picked_ids, self.picked_ids[1:])
            self.lines = [self.mesh.geodesic(a, b) for a, b in pairwise]
        else:
            self.lines = []
       
        if len(self.lines) > 0:
            lines = pv.PolyData() 
            self.merged = lines.merge(self.lines)
        

    def update_mesh(self):
        """ Split the mesh based on current geodesic. """
        # Split the mesh
        self.mesh = self.mesh.triangulate()
        tree = KDTree(self.mesh.points)
        _, ii = tree.query(self.merged.points, k=1)
        split, rdx = self.mesh.remove_points(ii)
        split.point_data['vtkOGIds'] = rdx
        
        split = split.connectivity()
        region_ids = split.point_data['RegionId']
        regions = np.unique(region_ids)
        r_masks = [region_ids == r_id for r_id in regions]
        split = [split.extract_points(r_m, adjacent_cells=False) for r_m in r_masks]
        split = sorted(split, key=lambda x: x.n_points, reverse=True)

        split_pd = [pv.PolyData(s.points, s.cells) for s in split]
        for s, s_pd in zip(split, split_pd):
            for arr in self.mesh.point_data:
                s_pd.point_data[arr] = s.point_data[arr]
            for arr in self.mesh.cell_data:
                s_pd.cell_data[arr] = s.cell_data[arr]

        # Smaller one mark 1, bigger 
        mask = np.ones(self.mesh.n_points, dtype=bool)
        mask[split[0].point_data['vtkOGIds']] = 0
        temp_mask = self.mesh.point_data[self.scalars]
        temp_mask[mask] = 1
        temp_mask[~mask] = 0

        # DM 11 11 21
        # Commented out the logical or, just used temp_mask
        # Interp old mask onto new
        tree = KDTree(self.current_pts)
        _, ii = tree.query(self.mesh.points,k=1)
        self.mesh.point_data[self.scalars] = self.current_mask[ii]
        new_mask = np.logical_or(temp_mask, self.mesh.point_data[self.scalars])
        self.mesh.point_data[self.scalars] = new_mask

        # self.mesh.point_data[self.scalars] = temp_mask

        if self.interactive:
            # self.p.add_mesh(self.mesh, name='mesh', scalars=self.scalars, cmap='coolwarm')
            self.p.add_mesh(self.mesh, name='mesh', scalars=self.scalars, cmap='Reds', show_edges=True)

    def display(self):
        """ Update display based on current state. """
        
        if len(self.picked_ids) > 1:
            tube = polyline_from_points(self.merged.points).tube(0.01)
            # line = pv.PolyData(self.merged.points, self.merged.cells)
            self.p.add_mesh(tube, name='lines', color='b')
        else:
            self.p.add_mesh(pv.Sphere(center=self.mesh.center), name="lines", opacity=0.0)
    
        if len(self.picked_points) > 0:
            points = pv.wrap(np.array(self.picked_points))

            self.p.add_mesh(points, 
                render_points_as_spheres=True, 
                color='r',
                name='points',
                )
        else:
            self.p.add_mesh(pv.Sphere(center=self.mesh.center), name="points", opacity=0.0)

    def smooth_section(self):
        """ Smoothes section with Laplacian filtering. """
        mask = self.mesh.point_data[self.scalars] == 1
        submesh = self.mesh.extract_points(mask, adjacent_cells=False)
        submesh = pv.PolyData(submesh.points, submesh.cells)
        submesh = submesh.smooth(n_iter=100, boundary_smoothing=False)
        self.mesh.points[mask] = submesh.points

        if self.interactive:
            # self.p.add_mesh(self.mesh, name='mesh', scalars=self.scalars, cmap='coolwarm')
            self.p.add_mesh(self.mesh, name='mesh', scalars=self.scalars, cmap='Reds', show_edges=True)

    def refill_section(self):
        """ Cut a hole and fill it. 
        
        """
        mask = self.mesh.point_data[self.scalars] == 0
        mask_sub = self.mesh.point_data[self.scalars] == 1

        self.mesh = self.mesh.clean()
        self.mesh = self.mesh.fill_holes(15.0)
        self.mesh = self.mesh.clean()

        mesh = self.mesh.extract_points(mask, adjacent_cells=False)
        submesh = self.mesh.extract_points(mask_sub)
        submesh = pv.PolyData(submesh.points, submesh.cells)
        edges = submesh.extract_feature_edges(boundary_edges=True, 
            non_manifold_edges=False, feature_edges=False, manifold_edges=False)
        submesh = pv.wrap(edges.points).delaunay_2d()
        # submesh = submesh.decimate(0.7)

        mesh = mesh.merge(submesh)
        mesh = pv.PolyData(mesh.points, mesh.cells)
        # mesh = mesh.boolean_union(submesh)

        # mesh = pv.PolyData(mesh.points, mesh.cells)
        self.mesh = mesh
        self.mesh.point_data[self.scalars] = np.zeros(self.mesh.n_points)

        self.mesh = self.mesh.clean()
        self.mesh = self.mesh.fill_holes(20.0)
        
        self.tree = KDTree(self.mesh.points)

        if self.interactive:
            # self.p.add_mesh(self.mesh, name='mesh', scalars=self.scalars, cmap='coolwarm')
            self.p.add_mesh(self.mesh, name='mesh', scalars=self.scalars, cmap='Reds', show_edges=True)

    def _get_points(self):
        """ Put the points in a geometry.
        """
        print("TEST", self.stored_points)
        self.stored_points = [x for x in self.stored_points if len(x) > 1]
        points = pv.MultiBlock()
        if len(self.stored_points) > 0:
            for pts in self.stored_points:
                points.append(pv.wrap(np.array(pts)))
            return points
        else:
            return []


    def save_stored_points(self, outfile=None):
        """ Save stored points for later use. """
        # neck_ids = [np.zeros(len(ll), dtype=int) + idx for idx, ll in enumerate(self.stored_points)]
        # neck_ids = [item for sublist in neck_ids for item in sublist]

        # points = pv.wrap(np.concatenate(self.stored_points, axis=0))
        # points.point_data['NeckIds'] = neck_ids

        points = pv.MultiBlock()
        for pts in self.stored_points:
            points.append(pv.wrap(np.array(pts)))

        if outfile is not None:
            points.save(outfile)       
        return points

    def use_stored_points(self, points):
        for pts in points:
            self.picked_points = pts.points
            self.update_points()
            self.update_geodesic()
            self.update_mesh()
            self.append()


class ClickDragDelete:
    """ Click and drag to select, space to delete.

    Clip meshes based on cell picking routines.
    Input surface must be vtkPolyData.
    
    Instructions:
    - Press "r" to toggle between selection/interaction.
    - Press "c" to clear selection.
    - Press "space" to delete cells.
    
    Based on example here:
    https://github.com/pyvista/pyvista/pull/281
    """
    def __init__(self, mesh, title='Clip mesh'):
        self.plotter = pv.Plotter()
        self.plotter.add_text(title, position='upper_left', font_size=18)
        msg = 'r: toggle selection mode'
        self.plotter.add_text(msg, position=(0.05, 75), font_size=12)
        msg = 'c: clear selection'
        self.plotter.add_text(msg, position=(0.05, 50), font_size=12)
        msg = 'space: clip '
        self.plotter.add_text(msg, position=(0.05, 25), font_size=12)
        msg = 'k: flag mesh'
        self.plotter.add_text(msg, position=(0.05, 0.05), font_size=12)

        self.mesh = mesh
        self.clear()

        self.plotter.enable_cell_picking(callback=self, show=False, show_message=False)
        self.plotter.add_key_event('c', callback=self.clear)
        self.plotter.add_key_event('space', callback=self.clip)
        self.plotter.add_key_event('k', callback=self.flag)

        self.flag_inspect = False

        self.plotter.show()

    def display(self):
        if self.mesh.n_points > 0:
            self.plotter.add_mesh(self.mesh,
                scalars='DelMask',
                name='mesh',
                show_scalar_bar=False,
                cmap='Reds')
        else:
            self.plotter.remove_actor('mesh')
        
    def __call__(self, picked_cells):
        self.picked.merge(picked_cells, inplace=True)
        if self.picked.n_cells > 0:
            self.mesh['DelMask'][self.picked.cell_data['orig_extract_id']] = 1
        self.display()

        return
    
    def clear(self):
        self.picked = pv.UnstructuredGrid()
        self.mesh.cell_data['DelMask'] = np.zeros(self.mesh.n_cells, dtype=bool)

        self.plotter.add_mesh(self.mesh, 
            color='w', 
            scalars='DelMask', 
            name='mesh',
            show_scalar_bar=False)
        self.display()

    def clip(self):
        if self.picked.n_points > 0:
            cells = np.invert(self.mesh.cell_data['DelMask'])
            self.mesh = self.mesh.extract_cells(cells)
            self.mesh.cell_data['DelMask'] = np.zeros(self.mesh.n_cells, dtype=bool)
            self.plotter.enable_cell_picking(self.mesh, callback=self, show=False, show_message=False)
            self.clear()
            self.display()
        
    def flag(self):
        print('Meshed flagged for further inspection.')
        self.flag_inspect = True


class ClickDragSelect:
    def __init__(self, mesh, title='Pick Points to Enlarge Mesh'):
        self.plotter = pv.Plotter()
        self.plotter.add_text(title, position='upper_left', font_size=18)
        msg = 'r: toggle selection mode'
        self.plotter.add_text(msg, position=(0.05, 75), font_size=12)
        msg = 'c: clear selection'
        self.plotter.add_text(msg, position=(0.05, 50), font_size=12)

        self.mesh = mesh
        self.clear()

        self.plotter.enable_cell_picking(callback=self, show=False, show_message=False)
        self.plotter.add_key_event('c', callback=self.clear)
        self.plotter.show()

    def display(self):
        if self.mesh.n_points > 0:
            self.plotter.add_mesh(self.mesh,
                scalars='PickedMask',
                name='mesh',
                show_scalar_bar=False,
                cmap='Reds')
        else:
            self.plotter.remove_actor('mesh')
        
    def __call__(self, picked_cells):
        self.picked.merge(picked_cells, inplace=True)
        if self.picked.n_cells > 0:
            self.mesh.cell_data['PickedMask'][self.picked.cell_data['orig_extract_id']] = 1
        self.display()

        return
    
    def clear(self):
        self.picked = pv.UnstructuredGrid()
        self.mesh.cell_data['PickedMask'] = np.zeros(self.mesh.n_cells, dtype=bool)

        self.plotter.add_mesh(self.mesh, 
            color='w', 
            scalars='PickedMask', 
            name='mesh',
            show_scalar_bar=False)
        self.display()


class ClickToDelete():
    """ Click a point, cells that contain it will be deleted.
    
    I don't think this works?
    """
    def __init__(self, mesh):
        self.p = pv.Plotter()
        msg = 'Press f to select point'
        self.p.add_text(msg, position=(0.05, 50), font_size=12)
        msg = 'Press x to delete cells'
        self.p.add_text(msg, position=(0.05, 25), font_size=12)
        msg = 'Press u to udno'
        self.p.add_text(msg, position=(0.05, 0), font_size=12)
        
        self.mesh = mesh
        self.prev_mesh = self.mesh.copy()

        self.p.add_mesh(self.mesh, color='w', name='mesh')
        self.p.enable_point_picking(callback=self, show_message=False)
        self.p.enable_cell_picking()
        self.p.add_key_event('x', callback=self)
        self.p.add_key_event('u', callback=self.undo)

        self.p.show()

    def __call__(self): # picked_cells
        pt = self.p.picked_point_id
        self.prev_mesh = self.mesh.copy()
        self.mesh, _ = self.mesh.remove_points([pt], mode='any')
        self.display()

    def display(self):
        if self.mesh.n_points > 0:
            self.p.add_mesh(self.mesh,
                color='w',
                name='mesh',
            )
        else:
            self.p.remove_actor('mesh')

    def undo(self):
        self.mesh = self.prev_mesh
        self.display()

def get_network_endpoints(network):
    """ Get terminal points of network
    """
    # First, get all unique endpoints in the network
    endpoints = []
    cells = []
    for ndx in range(network.n_cells):
        cell_mask = np.zeros(network.n_cells, dtype=bool)
        cell_mask[ndx] = True
        cell = network.extract_cells(cell_mask)
        endpoints.append(cell.points[0])
        endpoints.append(cell.points[-1])
        cells.append(cell)

    endpoints = np.array(endpoints)

    # Look for near-duplicates
    distances = np.zeros((endpoints.shape[0], endpoints.shape[0]))
    for idx in range(len(endpoints)):
        for jdx in range(idx, len(endpoints)):
            distances[idx, jdx] = np.linalg.norm(endpoints[idx] - endpoints[jdx])
            distances[jdx, idx] = distances[idx, jdx]

    tol = 1e-4
    distances = distances > tol
    np.fill_diagonal(distances, True)

    unique = [np.all(x == True) for x in distances]
    network_endpoints = endpoints[unique]
    return network_endpoints

def split_network_into_cells(network):
    cells = []
    for ndx in range(network.n_cells):
        cell_mask = np.zeros(network.n_cells, dtype=bool)
        cell_mask[ndx] = True
        cell = network.extract_cells(cell_mask)
        cells.append(cell)
    return cells 

def get_mean_radii(centerlines_branched, grouplist):
    """ Get mean radii of branches.

    Used for Chnafa flow splitting method.
    """
    mean_radii = {}

    # Get mean radii
    for node in grouplist:
        # print(node)
        mask = centerlines_branched.cell_data['GroupIds'] == int(node)

        branch_segments = centerlines_branched.extract_cells(mask)
        branch = branch_segments.split_bodies()[0]
        radius = branch.point_data['MaximumInscribedSphereRadius']
        
        # Convert to basic line for faster operations
        branch = lines_from_points(branch.points)
        branch.point_data['MaximumInscribedSphereRadius'] = radius
        branch = branch.ptc()

        branch = branch.compute_cell_sizes()
        lengths = branch.cell_data['Length'] 
        radius = branch.cell_data['MaximumInscribedSphereRadius']

        branch_length = np.sum(lengths)

        branch_resistance = np.sum(lengths / radius**4)

        mean_radius = (branch_length / branch_resistance)**0.25

        mean_radii[node] = mean_radius

    return mean_radii

def lines_from_points(points):
    """Given an array of points, make a line set"""
    poly = pv.PolyData()
    poly.points = points
    cells = np.full((len(points)-1, 3), 2, dtype=np.int_)
    cells[:, 1] = np.arange(0, len(points)-1, dtype=np.int_)
    cells[:, 2] = np.arange(1, len(points), dtype=np.int_)
    poly.lines = cells
    return poly


def polyline_from_points(points):
    """ Convert a list of points to a PolyLine"""
    poly = pv.PolyData()
    poly.points = points
    the_cell = np.arange(0, len(points), dtype=np.int_)
    the_cell = np.insert(the_cell, 0, len(points))
    poly.lines = the_cell
    return poly

def check_mem_usage():
    import psutil
    import os
    p = psutil.Process(os.getpid())
    mem_usage = p.memory_info().rss / 1024 / 1024
    print("{} MB".format(mem_usage))


def get_sac_surface_mask(mesh, sac):
    """ Get ids of surface points of sac on mesh.
    """
    mesh.point_data['vtkOGIds'] = list(range(mesh.n_points))

    sac = sac.fill_holes(20.0)
    sac = sac.compute_normals(auto_orient_normals=True)
    sac_inflate = sac.copy()
    sac_inflate.points = sac.points + 0.1*sac.point_data['Normals']

    mesh = mesh.select_enclosed_points(sac_inflate, check_surface=False)
    mesh['SacMask'] = mesh.point_data['SelectedPoints']
    mesh, _ = smooth_mesh_data_local(mesh, array='SacMask')

    surf = mesh.extract_surface()
    mesh_sac = surf.extract_points(surf.point_data['SacMask'] == 1)

    mesh_sac_ids = mesh_sac.point_data['vtkOGIds'].copy()

    mesh_sac_array = np.zeros(mesh.n_points, dtype=int)
    mesh_sac_array[mesh_sac_ids] = 1

    mesh.point_data['SurfaceSacMask'] = mesh_sac_array.astype(bool)

    return mesh

    
def decimate_edge_length(surf, target_edge_length):
    edges = surf.extract_all_edges()
    mean_el = edges.compute_cell_sizes().cell_data['Length'].mean()
    target_el = target_edge_length
    target_reduction = 1 - (mean_el / target_el)
    surf_d = surf.decimate(target_reduction, volume_preservation=True)
    surf = copy_arrays(surf, surf_d)
    return surf

def copy_arrays(src, dst):
    tree = KDTree(src.points)
    _, ii = tree.query(dst.points, k=1)
    for arr in src.point_data:
        dst.point_data[arr] = src.point_data[arr][ii]

    centers = src.cell_centers()
    tree = KDTree(centers.points)
    _, ii = tree.query(dst.cell_centers().points, k=1)
    for arr in src.cell_data:
        dst.cell_data[arr] = src.cell_data[arr][ii]
        
    return dst

def get_nearest_slice(mesh, origin, normal):
    """ Get nearest slice from surface or mesh.
    Could be used for tubeclipper 2.0.
    Originated from DEM `scraps` file `2022-04-27.py`
    """
    sl = mesh.slice(normal=normal, origin=origin, generate_triangles=False)

    sl = sl.connectivity()
    regions = np.unique(sl.point_data['RegionId'])
    rings = [sl.extract_points(sl.point_data['RegionId'] == x) for x in regions]
    ring_centers = pv.PolyData(np.array([x.center for x in rings]))
    ring_centers.point_data['RegionId'] = regions

    tree = KDTree(ring_centers.points)
    dd, ii  = tree.query(origin)
    closest_ring_id = ring_centers.point_data['RegionId'][ii]

    mask = sl.cell_data['RegionId'] == closest_ring_id
    closest_ring = sl.extract_cells(mask)
    return closest_ring

def get_parent_slice_location_from_sac_zones(surf, key=None):
    """ Get slices of the parent to calc flowrate.

    Used to calc parent flowrate for ICI.
    Definitely hacky, but fine for now.

    Hyper-specific to ICI, changed name to reflect.

    key is the dict key to a surface array.
    """
    if key is None:
        sac_zone_keys = [x for x in surf.point_data if 'sac_zone_' in x]

        if len(sac_zone_keys) > 1:
            print("Multiple zones present!")
        
        key = sac_zone_keys[0]

    mask = surf.point_data[key] == 1
    regions = surf.extract_points(mask)
    r_split = regions.split_bodies()
    g_ids = [np.median(x.point_data['GroupIds']) for x in r_split]
    min_g_id_index = np.argmin(g_ids)
    min_g_id = g_ids[min_g_id_index]
    parent_zone = r_split[min_g_id_index]
    edges = parent_zone.extract_feature_edges(45)
    edges = edges.split_bodies()
    edges = [pv.PolyData(e.points, lines=e.cells) for e in edges]
    caps = [e.delaunay_2d() for e in edges]
    caps = [c.compute_normals() for c in caps]
    caps = [c.compute_cell_sizes() for c in caps]
    caps = [c.ptc() for c in caps]
    origins = [c.center for c in caps]
    origins = pv.wrap(np.array(origins))
    normals = [np.average(c.cell_data['Normals'], axis=0, weights=c.cell_data['Area']) for c in caps]
    normals = np.array(normals)
    origins.point_data['Normals'] = normals
    
    # slices = [get_nearest_slice(surf, o, n) for o, n in zip(origins.points, normals)]
    return origins

def get_normal_component(surf, array='u', normals='Normals',):
    """ Get normal component of vector "array" wrt "normals".

    Creates array named array + '_normal' on surf.
    """
    surf.point_data[f'{array}_normal'] = np.einsum(
        'ij,ij->i', 
        surf.point_data[normals], 
        surf.point_data[array],
        )
    return surf

class Flow_Extender():
    def __init__(self, surf = None, centerlines = None, inlet_points=None, outlet_points=None,length = 2):
        self.surf = surf
        #self.surf_og=surf
        self.centerlines = centerlines
        self.length=length
        self.inlet_points=inlet_points
        self.outlet_points = outlet_points
        self.accept = True
    
    '''
    Runs the three functions that are required to create a Flow Extension on the outlet only
    '''
    def add_outlet_flow_ext(self):
        self.get_boundary_pts()
        self.get_normal_radius_effective()
        self.extrude()
        return self

    def get_boundary_pts(self):
        edges = self.surf.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False)
        edges = edges.connectivity()
        self.edges=edges
        regions = np.unique(edges.point_data['RegionId'])
        self.regions = regions
        masks = [edges.point_data['RegionId'] == r for r in regions]
        self.profiles = pv.MultiBlock([edges.extract_points(m) for m in masks])

    def get_normal_radius_effective(self):
        self.radii = np.empty(len(self.profiles))
        self.lengths = np.empty(len(self.profiles))
        self.prof_surf=pv.MultiBlock()
        for idx, prof in enumerate(self.profiles):
            prof_surf = prof.delaunay_2d()
            self.prof_surf.append(prof_surf)
            area = prof_surf.area
            radius_eff = np.sqrt(area/np.pi)
            self.radii[idx]=radius_eff
            self.lengths[idx]=self.length*radius_eff*2

        #associate lengths with inlet or outlet points
        centers = np.array([x.points.mean(axis=0) for x in self.profiles])
        centers_m = pv.wrap(centers)
        tree = KDTree(centers)
        inlet_ids = [tree.query(i)[1] for i in self.inlet_points]
        outlet_ids = list(set(range(centers_m.n_points)) - set(inlet_ids))
        self.lengths_out=self.lengths[outlet_ids]
        self.lengths_in=self.lengths[inlet_ids]
        self.radii_in=self.radii[inlet_ids]

        self.inlet_points = [centers[i] for i in inlet_ids]
        self.outlet_points = [centers[i] for i in outlet_ids]

        #get normals for profiles using centerlines
        self.tree = KDTree(self.centerlines.points)
        _, in_ids = self.tree.query(self.inlet_points)
        _, out_ids = self.tree.query(self.outlet_points)
        self.in_normals = self.centerlines.point_data['FrenetTangent'][in_ids]
        self.out_normals = -self.centerlines.point_data['FrenetTangent'][out_ids]

    def extrude(self):
        for id, pt in enumerate(self.outlet_points):
            #check that z is negative (should always be for outlets, but the Frenet Tangent isn't always oriented properly)
            if self.out_normals[id][2]>0:
                #look for closest neighbour centerline pt
                _, pidx = self.tree.query(pt)
                p2 = self.centerlines.points[pidx]
                #if the vector between the two points still has a positive Z, do nothing, otherwise invert the normal vector
                #otherwise the Frenet Tangent is inverted
                if (pt-p2)[2]<0:
                    self.out_normals[id]=-self.out_normals[id]
            center=pt+self.out_normals[id]*self.lengths_out[id]
            plane = pv.Plane(center=center, direction=self.out_normals[id], i_size = 30, j_size=30)
            self.prof_surf[id] = self.prof_surf[id].extrude(self.out_normals[id]*self.lengths_out[id]*1.5, capping=False)
            self.prof_surf[id] = self.prof_surf[id].triangulate()
            #self.prof_surf[id] = self.prof_surf[id].subdivide(2)
            
            clipped=self.prof_surf[id].clip_surface(plane, invert=False)
            
            self.outlet_points[id] = center
            self.surf=self.surf.merge(clipped, merge_points=True)  
            self.surf=self.surf.clean(tolerance=0.0001)
  
        self.surf = self.surf.fill_holes(1)
        self.surf=self.surf.smooth(n_iter=5)
        self.surf = self.surf.fill_holes(1)
        self.surf=self.surf.clean(tolerance=0.0001)
        edges = self.surf.extract_feature_edges(
                boundary_edges=True, 
                feature_edges=False, 
                manifold_edges=False)

        def _reject():
            self.accept = False

        pl2=pv.Plotter()
        pl2.add_title(title = "Inspect for unfilled holes")
        pl2.add_mesh(edges, color='red')
        pl2.add_mesh(self.surf, opacity=0.5)
        pl2.add_text('r+q: reject and redo', position=(0.05, 25), font_size=12)
        pl2.add_text('q: accept', position=(0.05, 50), font_size=12)
        pl2.add_key_event('r',_reject)
        pl2.add_axes()
        pl2.show()
