""" Volume meshing and solver input files.

`Mesher` extends `geometry_tools.surface.Surfer` with the steps that come after the surface is
prepared:

- `Mesher.surface_preparation` adds flow extensions, builds the `Size` array from local radius and
  curvature, and remeshes the surface (used by `surface_process.py`, mainly for aneurysms).
- `Mesher.generate_volume_mesh` builds the tetrahedral mesh with a boundary layer, using
  vmtkMeshGenerator and the surface's `Size` array.
- `Mesher.generate_h5_file`, `Mesher.generate_info_file` and `Mesher.generate_xml_gz_file` write the
  mesh, boundary conditions and FEniCS mesh that the lab's solver reads.
- `Mesher.generate_flow_rates` splits the inlet flow among the outlets.

A `Mesher` can be built from a surface, or from an existing volume mesh with a `CellEntityIds`
array, in which case the wall (entity 1) is extracted as the surface. VMTK's TetGen meshing can be
unreliable; check every mesh with `meshquality.py`.
"""
import pyvista as pv
import numpy as np
from geometry_tools import vmtk_wrapper as vmtk
from geometry_tools.surface import Surfer

from pathlib import Path 
from geometry_tools import common as cc 
from scipy.spatial import cKDTree as KDTree 
import h5py 

class Mesher(Surfer):
    """ Basic meshing tools based on VMTK. 

    Attributes:
        surf : surface PolyData or None 
        mesh : volumetric UnstructuredGrid or None
        inlet_points : array of inlet points
        outlet_points : array of outlet points
        centerlines : PolyData centerlines of surf
        centerlines_branched : PolyData centerlines of surf with branch IDs
    """

    def __init__(self, surf=None, mesh=None, inlet_points=None, outlet_points=None, aneurysm_points=None, include_aneurysms=False):
        """ Init the mesher instance.

        Either surf or mesh must be given. If mesh, surf will be extracted 
        based on CellEntityIds.
        """
        self.mesh = mesh
        super().__init__(surf=surf, 
            inlet_points=inlet_points, 
            outlet_points=outlet_points, 
            aneurysm_points=aneurysm_points,
            include_aneurysms=include_aneurysms)

        if (self.mesh is not None) and (self.surf is None):
            # Extract surface using entity ids
            surf_mask = self.mesh.cell_data['CellEntityIds'] == 1
            surf = self.mesh.extract_cells(surf_mask)
            surf_pt_ids = surf.point_data['vtkOriginalPointIds']
            surf = pv.PolyData(surf.points, surf.cells)
            surf.point_data['vtkOriginalPointIds'] = surf_pt_ids
            self.surf = surf.compute_normals()

    def set_inlets_outlets(self):
        """ Interactively choose inlet point.

        All other open boundaries will be considered outlets. To update these points 
        (for example, after adding flow extensions), see self.update_inlet_outlets. 
        To save these inlet-outlet points, see self.save_inlet_outlet_points()
        """
        centers_m, inlet_ids, outlet_ids = Surfer.set_inlets_outlets(self)
        if self.mesh is not None:
            self._set_inlet_outlet_entity_ids(centers_m, inlet_ids, outlet_ids)

    def update_inlets_outlets(self):
        """ Update inlet-outlet points based on distance metric.
        """
        centers_m, inlet_ids, outlet_ids = Surfer.update_inlets_outlets(self)
        if self.mesh is not None:
            self._set_inlet_outlet_entity_ids(centers_m, inlet_ids, outlet_ids)

    def _set_inlet_outlet_entity_ids(self, centers_m, inlet_ids, outlet_ids):    
        """ Internal method for keeping track of CellEntityIds.

        Args:
            centers_m : PolyData containing points centered on open profiles
            inlet_ids : list index of inlet ids
            outlet_ids : list of index of outlet ids
        
        Sets:
            inlet_entity_ids : list of inlet entity ids
            outlet_entity_ids : list of outlet entity ids

        """
        entity_ids = np.unique(self.mesh.cell_data['CellEntityIds'])
        valid_entity_ids = sorted(set(entity_ids) - set([0, 1]))
        entity_masks = [self.mesh.cell_data['CellEntityIds'] == i for i in valid_entity_ids]
        caps = [self.mesh.extract_cells(em) for em in entity_masks]
        cap_centers = [c.points.mean(axis=0) for c in caps]

        self.caps = dict(zip(valid_entity_ids, caps))

        tree = KDTree(cap_centers)
        
        inlet_points = centers_m.points[inlet_ids]
        outlet_points = centers_m.points[outlet_ids]

        inlet_temp_ids = tree.query(inlet_points)[1]
        outlet_temp_ids = tree.query(outlet_points)[1]

        self.inlet_entity_ids = [valid_entity_ids[x] for x in inlet_temp_ids]
        self.outlet_entity_ids = [valid_entity_ids[x] for x in outlet_temp_ids]

        if hasattr(self, 'centerlines_branched'): 
            # Match inlet/outlet points with a group id
            c_branch = self.centerlines_branched.copy()
            c_branch = c_branch.ctp()

            tree = KDTree(c_branch.points)
            ii = tree.query(cap_centers)[1]

            group_ids = [c_branch.point_data['GroupIds'][i] for i in ii]

            self.inlet_group_ids = [group_ids[x] for x in inlet_temp_ids]
            self.outlet_group_ids = [group_ids[x] for x in outlet_temp_ids]
            
    def surface_preparation(self, proj_dir, multi_inlets, ref, ND, fix_centerline, neck_points=None, min_edge_size=0.1, max_edge_size=0.4, sac_size=0.15, misr_min=0.1, misr_max=2.5):
        """ Refine surface, add flow extensions. 

        Remeshes surface, clips endpoints normal to centerlines,
        adds flow extensions, creates size array based on local radius
        and curvature, then does a final remshing and cleaning.
        """
        # VMTK uses a weird dtype sometimes, but recreating the surface 
        # from the points and cells fixes this issue. Need to copy 
        # array "Mask" though.
        surf_og = pv.PolyData()
        surf_og.copy_structure(self.surf)

        if neck_points is not None: 
            surf_og.point_data['Mask'] = self.surf.point_data['Mask']
        elif (ND == 'nd') and (ref == 'refine'):
            surf_og.point_data['RefinementPoints'] = self.surf.point_data['RefinementPoints']
            surf_og.point_data['Enlarge_Cells'] = self.surf.point_data['Enlarge_Cells']
        elif ND == 'nd':
            surf_og.point_data['Enlarge_Cells'] = self.surf.point_data['Enlarge_Cells']
        elif ref == 'refine':
            surf_og.point_data['RefinementPoints'] = self.surf.point_data['RefinementPoints']

        # surf_og.point_data['GroupIds'] = self.surf.point_data['GroupIds']

        surf = pv.PolyData(self.surf.points, self.surf.faces)
        if neck_points is not None: 
            surf.point_data['Mask'] = surf_og.point_data['Mask'].copy()
        elif (ref == 'refine') and (ND == 'nd'):
            surf.point_data['RefinementPoints'] = surf_og.point_data['RefinementPoints']
            surf.point_data['Enlarge_Cells'] = surf_og.point_data['Enlarge_Cells']        
        elif ND == 'nd':
            surf.point_data['Enlarge_Cells'] = surf_og.point_data['Enlarge_Cells']
        elif ref == 'refine':
            surf.point_data['RefinementPoints'] = surf_og.point_data['RefinementPoints']
        surf = surf.clean()
        
        # Use centerlines without aneurysm
        centerlines1 = self.centerlines
        centerlines1.save(proj_dir / ('network_centerlines0.vtp'))
        
        surf, centerlines1 = vmtk.distance_to_centerlines(surf, centerlines1)
        surf = surf.clean()

        #AH's implementation of Dan's solution to the off-centre centrelines. There is no use for this atm
        if (fix_centerline == 'fix'):
            surf.point_data['centerline_map']=np.array(surf.n_points)
            surf.point_data['misr']=np.array(surf.n_points)
            tree = KDTree(centerlines1.points)
            pt, idx_c = tree.query(surf.points)
            #this assigns the closest centerline point id to each point on the surf
            surf.point_data['centerline_map']=idx_c 
            #this assigns the misr associated with the centerline to each point
            surf.point_data['misr']=centerlines1.point_data['MaximumInscribedSphereRadius'][idx_c] 
            #fix around the ends
            _, idx_d = tree.query(surf.points[surf.point_data['misr']==0], k=10)
            max_neigh_misr = np.amax(centerlines1.point_data['MaximumInscribedSphereRadius'][idx_d], axis=1)
            surf.point_data['misr'][surf.point_data['misr']==0]=max_neigh_misr

            
        # surf = cc.create_edge_size_array(surf, max_size=max_size, min_size=min_size, sac_size=sac_size, misr_min=misr_min, misr_max=misr_max)
        surf = cc.create_edge_size_array(surf, fix_centerline, min_edge_size=min_edge_size, max_edge_size=max_edge_size, sac_size=sac_size, 
            misr_min=misr_min, misr_max=misr_max, name='Size', ref_edge_ratio=0.5)
        surf = vmtk.surface_remeshing(surf, element_size_mode='edgelengtharray', edgearray='Size')
        #print(surf.point_data)
        
        # DM 22 02 22
        # NOTE
        # Smoothing here is a BAD idea because you get "ringing" artifacts at the boundaries. 
        # You should smooth earlier in the pipeline to avoid this.
        # Leaving this line here, commented out, to remind you next time you think 
        # it's a good idea.
        # if smooth:
        #     surf = cc.vtk_taubin_smooth(surf, pass_band=0.05, iterations=100)

        #moved the flow extensions to surface_prep!!
        surf = surf.clean()

        self.surf = surf
        self.update_inlets_outlets()
        if multi_inlets == 'multi':    
            #self.generate_centerlines_multi(proj_dir)
            centerlines1 = self.centerlines
            #centerlines1.save(proj_dir / ('network_centerlines.vtp'))
        else:
            self.generate_centerlines(include_aneurysms=False)
            centerlines1 = self.centerlines

        if neck_points is not None:
            s = cc.SelectGeodesic(surf, scalars='Mask')
            print('neck', neck_points)
            s.use_stored_points(neck_points)
            surf = s.mesh
        else:
            surf = surf.interpolate(surf_og, n_points=1, pass_point_data=True) #radius=0.5, strategy='null_value', null_value=0)
        #surf.save(proj_dir/'after_interp0.vtp')
        
        #print(surf.point_data)
        surf, centerlines = vmtk.distance_to_centerlines(surf, centerlines1)
        surf = surf.clean()

        if (fix_centerline == 'fix') or (fix_centerline == 'network_fix'):
            surf.point_data['centerline_map']=np.array(surf.n_points)
            surf.point_data['misr']=np.array(surf.n_points)
            tree = KDTree(centerlines1.points)
            pt, idx_c = tree.query(surf.points)
            #this assigns the closest centerline point id to each point on the surf
            surf.point_data['centerline_map']=idx_c 
            #this assigns the misr associated with the centerline to each point
            surf.point_data['misr']=centerlines1.point_data['MaximumInscribedSphereRadius'][idx_c] 
            #fix around the ends
            _, idx_d = tree.query(surf.points[surf.point_data['misr']==0], k=10)
            max_neigh_misr = np.amax(centerlines1.point_data['MaximumInscribedSphereRadius'][idx_d], axis=1)
            surf.point_data['misr'][surf.point_data['misr']==0]=max_neigh_misr

        surf = cc.create_edge_size_array(surf, fix_centerline, min_edge_size=min_edge_size, max_edge_size=max_edge_size, sac_size=sac_size, 
            misr_min=misr_min, misr_max=misr_max, name='Size', ref_edge_ratio=0.5)
            
        surf, _ = cc.smooth_mesh_data_local(surf, 'Size', np.mean, 
            iterations=5, 
            )

        surf_rm = vmtk.surface_remeshing(surf, element_size_mode='edgelengtharray', edgearray='Size')

        surf_rm = surf_rm.clean()
        surf_rm = surf_rm.triangulate()
        surf_rm = surf_rm.clean()
        # surf_rm = surf_rm.interpolate(surf_og, radius=0.5)
        # surf_rm = surf_rm.interpolate(surf, radius=0.5)

        if neck_points is not None:
            s = cc.SelectGeodesic(surf_rm, scalars='Mask')
            s.use_stored_points(neck_points)
            surf_rm = s.mesh 
        #Interpolation here seems to add little artifacts to the mesh where the is Size=0
        #in the point array. Still needed for the refinement points and mask points. 
        #Must re-calculate the size array for when the meshgenerator needs it, though...
        else:
            surf_rm = surf_rm.interpolate(surf_og, n_points=1, pass_point_data=True) #radius=0.5, strategy='null_value', null_value=0)
            surf_rm = surf_rm.clean()
        #surf_rm.save(proj_dir/'after_interp1.vtp')
        surf_rm = surf_rm.interpolate(surf, radius=0.7, pass_point_data=True)
        #surf_rm.save(proj_dir/'after_interp2.vtp')    
        surf_rm = cc.create_edge_size_array(surf_rm, fix_centerline, min_edge_size=min_edge_size, max_edge_size=max_edge_size, sac_size=sac_size, 
            misr_min=misr_min, misr_max=misr_max, name='Size', ref_edge_ratio=0.5)
            
        surf_rm, _ = cc.smooth_mesh_data_local(surf_rm, 'Size', np.mean, 
            iterations=5, 
            )

        surf_rm = surf_rm.clean()
        # print('*'*50)
        # print(surf.point_data)
        # print('*'*50)
        # print(surf_og.point_data)
        self.surf = surf_rm 
        self.centerlines = centerlines

    def select_refinement_regions(self):
        """ Interactively choose refinement region.

        See docs of geometry_tools.common.SacSelectTool for more info.
        This is not used!!
        """
        select = cc.SacSelectTool(self.surf)
        select.select()
        self.surf = select.surf 

    def mark_refinement_regions(self):
        """ Automatically mark refinement region based on GroupIds.
        To mark manually, use select_refinement_regions.
        """
        self.surf.point_data['Mask'] = np.zeros(self.surf.n_points)
        for an_id in self.aneurysm_group_ids:
            mask = self.surf.point_data['GroupIds'] == an_id
            self.surf.point_data['Mask'][mask] = 1

    def refine_picked_regions(self):
        """ Update point array "Size" of surf using chosen refinement regions.
        """
        surf = self.surf 
        centerlines = self.centerlines_aneurysm

        surf, centerlines = vmtk.distance_to_centerlines(surf, centerlines)
        surf = cc.create_edge_size_array(surf, max_size=0.4, min_size=0.18,)
        surf.point_data['Size'][surf.point_data['Mask'] > 0.99] = 0.18
        # surf.point_data['Size'] = cc.smooth_mesh_data(surf.point_data['Size'], surf.points, 1.0, func=np.mean)
        surf = vmtk.surface_array_smoothing(surf, array_name='Size', iterations=5)
        surf.set_active_scalars('Mask')
        
        self.surf = surf
        self.centerlines_aneurysm = centerlines

    def generate_flow_rates_legacy(self):
        """ Generate flow rates using Christophe's old code.
        """
        from geometry_tools.legacy import networks, network_boundary_conditions
        
        network = networks.Network()
        centerlinesBranches = networks.SetNetworkStructure(self.centerlines, network, print,
            isConnectivityNeeded=True)

        flowSplitting = network_boundary_conditions.FlowSplitting()
        flowSplitting.ComputeAlphas(network, lambda x: None)
        flowSplitting.ComputeBetas(network, lambda x: None)
        flowSplitting.CheckTotalFlowRate(network, lambda x: None)

        betas = {}
        inlet = {}

        for element in network.elements:
            if element.IsAnOutlet():
                networkPoint = element.GetOutPointsx1()[0]
                betas[element.vtkGroupIdList[0]] = element.GetBeta()

            if  element.IsAnInlet():
                inlet[element.vtkGroupIdList[0]] = element.GetInletRadius()

        print('betas \n', betas)

        new_centerlines = pv.wrap(centerlinesBranches)

        for b in betas.keys():
            mask = new_centerlines.cell_data['GroupIds'] == b 
            branch_segments = new_centerlines.extract_cells(mask)
            branch = branch_segments.split_bodies()[0]
            radius = branch.point_data['MaximumInscribedSphereRadius']

    def generate_flow_rates(self):
        """ Get flow rates based on Christophe method.

        Gives equivalent values as legacy call, just rewritten.
        The centerlines should be passed before calling centerline branch ids.
        NOTE: does not work for multiple inlets!!
        """ 
        self.get_group_adjacency()
        if self.include_aneurysms==True:
            nodes = [x for x in self.G_no_aneurysm.nodes] 
            print(nodes)
        else:
            nodes = [x for x in self.G.nodes] 
            print(nodes)

        centerlines_branched = vmtk.centerline_branches_ids(self.centerlines)
        self.centerlines_branched = centerlines_branched
        self.update_inlets_outlets()
        mean_radii = cc.get_mean_radii(centerlines_branched, nodes)
        beta_values = {}
        beta_values['0'] = 1.

        outlet_flow_divisions = {}

        for node in nodes:
            children = [x for x in self.G_no_aneurysm.successors(node)]
            if len(children) > 0:
                node_radii = mean_radii[node]
                children_radii = np.array([mean_radii[x] for x in children])
                children_sum = np.sum(children_radii**2)
                children_ratio = children_radii**2 / children_sum

                for cdx, ch in enumerate(children):
                    beta_values[ch] = children_ratio[cdx]*beta_values[node]
        
            else:
                outlet_flow_divisions[node] = beta_values[node]

        # print('\n' + 'Outlet_flow_divisions', outlet_flow_divisions)

        # Replace GroupIds keys with EntityIds 
        translate_out = dict(zip(self.outlet_group_ids, self.outlet_entity_ids))

        out_flow_keys = outlet_flow_divisions.keys()
        new_out_flow_keys = [translate_out[int(x)] for x in out_flow_keys]
        out_flow_div_values = [outlet_flow_divisions[x] for x in out_flow_keys]

        # print('*****dan*******'*5)
        #print(out_flow_div_values)
        # print('*****dan*******'*5)

        self.outlet_flow_divisions = dict(zip(new_out_flow_keys, out_flow_div_values))

    def generate_volume_mesh(self, SkipRemeshing=0):
        """ Build the volume mesh from `self.surf` and store it in `self.mesh`.

        Uses vmtkMeshGenerator with the surface's `Size` array as the target edge length and a
        4-sublayer boundary layer (see `geometry_tools.vmtk_wrapper.volume_meshing`).

        Args:
            SkipRemeshing: 1 if `self.surf` has already been remeshed to the `Size` array, so
                vmtk meshes it as-is. 0 lets vmtk remesh the surface first.
        """
        mesh = vmtk.volume_meshing(self.surf, SkipRemeshing=SkipRemeshing)
        self.mesh = mesh


    def generate_h5_file(self, outfile):
        """ Generate lab-specific h5 mesh file for our solver.

        File layout:
            - Mesh
                - coordinates
                - topology
                - ID_N
                    - cellIds, coordinates, pointIds, topology
                - Wall
                    - cellIds, coordinates, normal, pointIds, topology

        NOTE unexpected behaviour observed 2022-02-28:
        Is the wall being appended to the mesh?
        NOTE I don't observe this behaviour, cellentityid=0 is all tetrahedrons for me. 
        I will comment this out - AH
        """
        case_name = outfile.stem

        # Quads only
        #mesh_quad = vmtk.assert_all_quads(self.mesh)
        mesh_volume = self.mesh.cell_data['CellEntityIds'] == 0
        mesh_quad = self.mesh.extract_cells(mesh_volume)

        # Get surf
        surf_mask = self.mesh.cell_data['CellEntityIds'] == 1
        surf = self.mesh.extract_cells(surf_mask)
        surf_pt_ids = surf.point_data['vtkOriginalPointIds']
        surf = pv.PolyData(surf.points, surf.cells)
        surf.point_data['vtkOriginalPointIds'] = surf_pt_ids
        surf = surf.compute_normals()

        # Wall points
        wall_coordinates = surf.points 
        wall_pointIds = surf.point_data['vtkOriginalPointIds']
        wall_topology = surf.faces.reshape(-1,4)[:,1:]
        wall_cellIds = wall_pointIds[wall_topology]
        wall_normals = surf.point_data['Normals']
        
        # Extract caps        
        entity_ids = np.unique(self.mesh.cell_data['CellEntityIds'])
        valid_entity_ids = sorted(set(entity_ids) - set([0, 1]))
        entity_masks = [self.mesh.cell_data['CellEntityIds'] == i for i in valid_entity_ids]
        caps = [self.mesh.extract_cells(em) for em in entity_masks]

        # Make sure points are duplicate
        '''
        tree = KDTree(mesh_quad.points)
        inds = [tree.query(c.points, k=1)[1] == 0 for c in caps]
        dists = np.all([np.all(tree.query(c.points, k=1)[0] == 0) for c in caps])
        assert dists, 'Cap points are not in quad mesh'
        '''
        # Get relevant cap quantities
        caps_coordinates = [c.points for c in caps]
        caps_pointIds = [c.point_data['vtkOriginalPointIds'] for c in caps]
        caps_topology = [c.cells.reshape(-1,4)[:,1:] for c in caps]
        caps_cellIds = [c.point_data['vtkOriginalPointIds'][c.cells.reshape(-1,4)[:,1:]] for c in caps]

        # Now create dataset
        f = h5py.File(outfile, "w")
        
        f.create_dataset('Mesh/coordinates', 
            data=mesh_quad.points, 
            compression="gzip", 
            compression_opts=9
            )
        f.create_dataset('Mesh/topology', 
            data=mesh_quad.cells.reshape(-1,5)[:,1:], 
            compression="gzip", 
            compression_opts=9
            )
        
        f.create_dataset('Mesh/Wall/coordinates', 
            data=wall_coordinates, 
            compression="gzip", 
            compression_opts=9
            )
        f.create_dataset('Mesh/Wall/pointIds', 
            data=wall_pointIds, 
            compression="gzip", 
            compression_opts=9
            )
        f.create_dataset('Mesh/Wall/topology', 
            data=wall_topology, 
            compression="gzip", 
            compression_opts=9
            )
        f.create_dataset('Mesh/Wall/cellIds', 
            data=wall_cellIds, 
            compression="gzip", 
            compression_opts=9
            )
        f.create_dataset('Mesh/Wall/normal', 
            data=wall_normals, 
            compression="gzip", 
            compression_opts=9
            )
        
        for cdx in range(len(caps)):
            f.create_dataset('Mesh/ID_{}/coordinates'.format(cdx + 1), 
                data=caps_coordinates[cdx], 
                compression="gzip", 
                compression_opts=9
                )
            f.create_dataset('Mesh/ID_{}/pointIds'.format(cdx + 1), 
                data=caps_pointIds[cdx], 
                compression="gzip", 
                compression_opts=9
                )
            f.create_dataset('Mesh/ID_{}/topology'.format(cdx + 1), 
                data=caps_topology[cdx], 
                compression="gzip", 
                compression_opts=9
                )
            f.create_dataset('Mesh/ID_{}/cellIds'.format(cdx + 1), 
                data=caps_cellIds[cdx], 
                compression="gzip", 
                compression_opts=9
                )

        f.close()

    def generate_info_file(self, outfile, fcoeffsfile, multi_inlets, inlet_vel, inlet_flowrates, waveform='FC_MCA_10'):
        """ Generate lab-specific info file.

        Formats a bunch of values into strings, including 
        area, radius, flow divisions, etc of inlet/outlet

        Outflow divisions for PT are based on a Di^2/sum(Di^2) approximation
        """

        #Print FC_VENOUS file
        if waveform == 'FC_VENOUS':
            fcoeffsfile = open(fcoeffsfile, "w")
            fcoeffsfile.write("# Qavg_mls= 6.7 period_ms= 915\n")
            fcoeffsfile.write('1 0\n-0.146846 0.079431\n-0.129927 0.015974')
            fcoeffsfile.close()
        # # Get branch center, normal, rad, area
        center_points = vmtk.branch_center_normal_rad_area(self.mesh)
        
        #only used for the pt outlet divisions
        outlet_area = 0
        for cdx, entity_id in enumerate(self.caps):
            if entity_id in self.outlet_entity_ids:
                outlet_area += center_points[entity_id].point_data['Area'][0]
        # # Match GroupIds from self.centerlines_branched to CellEntityIds
        # # Or match GroupIds with inlet points, inlets points is matched with entity ids

        # # Extract caps   
        # entity_ids = np.unique(self.mesh.cell_data['CellEntityIds'])
        # valid_entity_ids = sorted(set(entity_ids) - set([0, 1]))
        # entity_masks = [self.mesh.cell_data['CellEntityIds'] == i for i in valid_entity_ids]
        # caps = [self.mesh.extract_cells(em) for em in entity_masks]

        info_data = {}

        # entity_id_group_id_matcher = dict(zip(self.outlet_entity_ids, self.outlet_group_ids))
        # entity_id_group_id_matcher = dict(zip(self.outlet_group_ids, self.outlet_entity_ids))
        inlet_FR_id=0
        for cdx, entity_id in enumerate(self.caps):
            #     entity_id = valid_entity_ids[cdx]
            center = center_points[entity_id]

            c = center.points[0]
            normal = center.point_data['Normal'][0]
            rad = center.point_data['Radius'][0]
            area_ = center.point_data['Area'][0]

            c = (','.join("{:.12f}".format(x) for x in c))
            c = '(' + c + ')'
            normal = (','.join("{:.12f}".format(x) for x in normal))
            normal = '(' + normal + ')'
            rad = "{:.12f}".format(rad)
            area = "{:.12f}".format(area_)

            dataline = [str(cdx + 1), str(None), c, normal, rad, area]

            if entity_id in self.inlet_entity_ids:
                if inlet_vel != False:
                    inflowrate = inlet_vel * area_
                else:
                    inflowrate = inlet_flowrates[inlet_FR_id]
                    inlet_FR_id+=1
                dataline[1] = waveform
                dataline.append("{:.12f}".format(inflowrate))

            elif (entity_id in self.outlet_entity_ids) and multi_inlets=='single':
                # FIX! Need to match GroupIds with CellEntityIds
                if len(self.outlet_flow_divisions) != 0: #ie. if there is more than one outlet and this is an arterial mesh
                    flow_division = self.outlet_flow_divisions[entity_id]
                    dataline.append("{:.12f}".format(flow_division))
                elif len(self.outlet_entity_ids) == 1: #if only one outlet
                    flow_division = 1.00
                    dataline.append("{:.12f}".format(flow_division))
                else: #if single inlet, multiple outlets
                    flow_division = area_/outlet_area
                    dataline.append("{:.12f}".format(flow_division))
            elif (entity_id in self.outlet_entity_ids) and multi_inlets!='single':
                #WARNING: Not the greatest!
                flow_division = area_/outlet_area
                dataline.append("{:.12f}".format(flow_division))
            else:
                print('Check inlets outlets')

            info_data[entity_id] = dataline
        if waveform=='FC_VENOUS':
            print('Double-check the flowrates in Paraview before running!')
        infofile = open(outfile, "w")
        header = "# id, wave, center, normal, radius, area, FR(inlet)/AR(outlet)"
        infofile.write(header + 2*"\n")
        infofile.write(outfile.stem + "\n")
        infofile.write("\n")
        infofile.write("<INLETS>" + "\n")

        for e_id in self.inlet_entity_ids:
            dataline = info_data[e_id]    
            dataline = '   '.join(dataline)
            infofile.write(dataline + "\n")

        infofile.write("\n")
        infofile.write("<OUTLETS>" + "\n")
        for e_id in self.outlet_entity_ids:
            dataline = info_data[e_id]    
            dataline = '   '.join(dataline)
            infofile.write(dataline + "\n")

        infofile.close()

    def generate_xml_gz_file(self, xml_gz_file):
        """ Generate xml file using VMTK.
        """
        xml_file = xml_gz_file.parents[0] / xml_gz_file.stem

        # mesh = vmtk.assert_all_quads(self.mesh)

        vmtk.write_mesh(self.mesh, xml_file)
      
if __name__ == "__main__":
    print('See example scripts directory')
