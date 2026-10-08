from networkx.algorithms.dag import ancestors
from networkx.algorithms.distance_measures import center
from networkx.algorithms.lowest_common_ancestors import lowest_common_ancestor
import pyvista as pv 
import numpy as np
from geometry_tools import vmtk_wrapper as vmtk
# from pathlib import Path 
from geometry_tools import common as cc 
from scipy.spatial import cKDTree as KDTree 
# import h5py 
from tubeclipper import TubeClipper
import networkx as nx 

class Surfer():
    """ Basic mesh manip tools based on VMTK.
    """

    def __init__(self, surf=None, inlet_points=None, outlet_points=None, aneurysm_points=None, inlet_normals=None, outlet_normals=None, include_aneurysms=False):
        """ Init the meshproto instance.

        Surf must be given. 
        """

        self.surf = surf 
        self.inlet_points = inlet_points
        self.outlet_points = outlet_points
        self.aneurysm_points = aneurysm_points
        self.include_aneurysms=include_aneurysms

        if self.inlet_points is not None:
            self.update_inlets_outlets()
            self.inlet_points = np.array(inlet_points)
            self.outlet_points = np.array(outlet_points)
        
    def _pick_points(self, mesh=None, text='', render_points_as_spheres=False):
        """ Internal method for picking points for choosing inlet(s).

        Input mesh should be, for example, a PolyData containing 
        inlets and outlets. Returns points in a list.
        """

        def _point_picker_cb(mm, pid):
            # point = mm.points[pid]
            # print('Picked point', point)
            #self._picked_id = pid
            self._picked_ids.append(pid)
            p.add_point_labels([mm.points[pid],],['{}'.format(pid)], )

        def _reset_picked_cb():
            print('reset')
            self._picked_id = None
            self._picked_ids = []

        self._picked_ids = []
        self._picked_id = None

        p = pv.Plotter()
        p.add_mesh(self.surf, color='w', opacity=1.0, pickable=False)
        p.add_text(text, position='upper_left')
        p.enable_point_picking(callback=_point_picker_cb, show_message=False, 
                            color='r', point_size=10, 
                            use_mesh=True, show_point=True, 
                            render_points_as_spheres=render_points_as_spheres)
        if mesh is not None:
            p.add_mesh(mesh, color='b', point_size=30)

        p.add_text('p: pick point', position=(0.05, 0.05), font_size=12)
        p.add_key_event('c', _reset_picked_cb)
        p.show()
        return self._picked_ids #[self._picked_id]

    def copy_structure(self):
        surf_new = pv.PolyData()
        surf_new.copy_structure(self.surf)
        self.surf = surf_new
        
    def copy_arrays(self, src, dst):
        # tree = KDTree(src.points)
        # _, ii = tree.query(dst.points, k=1)
        # for arr in src.point_data:
        #     dst.point_data[arr] = src.point_data[arr][ii]

        # centers = src.cell_centers()
        # tree = KDTree(centers.points)
        # _, ii = tree.query(dst.cell_centers().points, k=1)
        # for arr in src.cell_data:
        #     dst.cell_data[arr] = src.cell_data[arr][ii]
            
        # return dst
        return cc.copy_arrays(src, dst)

    def decimate_surface(self, target_edge_length):
        # edges = self.surf.extract_all_edges()
        # mean_el = edges.compute_cell_sizes().cell_data['Length'].mean()
        # target_el = target_edge_length
        # target_reduction = 1 - (mean_el / target_el)
        # surf_d = self.surf.decimate(target_reduction, volume_preservation=True)
        # self.surf = self.copy_arrays(self.surf, surf_d)
        self.surf = cc.decimate_edge_length(self.surf, target_edge_length)

    def clip_endpoints_with_spheres(self, factor=1.4, outlet_clip=None):
        
        if outlet_clip is None:
            endpoints = np.concatenate([self.inlet_points, self.outlet_points], axis=0)
            endlets = pv.wrap(endpoints)

            tree = KDTree(self.centerlines_aneurysm.points)
            _, ii = tree.query(endlets.points, k=3)
            ii = np.array(ii)[:,-1]
            
            misr = self.centerlines_aneurysm.point_data['MaximumInscribedSphereRadius'][ii]
            
            endlets.point_data['MaximumInscribedSphereRadius'] = misr
            outlet_clip = endlets.glyph(geom=pv.Sphere(1.0), factor=factor)

        self.surf = self.surf.clip_surface(outlet_clip, invert=False)
        self.surf = self.surf.extract_largest()
        return outlet_clip

    def clip_endpoints_with_tubeclipper(self, endpoints_pv=None, include_aneurysms=True):
        """ Can introduce buggy mesh! Beware!
        """
        if endpoints_pv is None:
            endpoints = np.concatenate([self.inlet_points, self.outlet_points], axis=0)
            
            if include_aneurysms==True:
                tree = KDTree(self.centerlines_aneurysm.points)
            else:
                tree = KDTree(self.centerlines.points)
                
            _, ii_inlets = tree.query(self.inlet_points, k=3)
            _, ii_outlets = tree.query(self.outlet_points, k=3)

            ii_inlets_n = np.array(ii_inlets)[:,-1]
            ii_outlets_n = np.array(ii_outlets)[:,-1]

            ii_inlets = np.array(ii_inlets)[:,0]
            ii_outlets = np.array(ii_outlets)[:,0]
            
            if include_aneurysms==True:
                # Get points at those points
                in_points = self.centerlines_aneurysm.points[ii_inlets]
                out_points = self.centerlines_aneurysm.points[ii_outlets]
            
                # Get normal at those points
                in_normals = self.centerlines_aneurysm.point_data['FrenetTangent'][ii_inlets_n]
                out_normals = -self.centerlines_aneurysm.point_data['FrenetTangent'][ii_outlets_n]
            else:
                # Get points at those points
                in_points = self.centerlines.points[ii_inlets]
                out_points = self.centerlines.points[ii_outlets]
            
                # Get normal at those points
                in_normals = self.centerlines.point_data['FrenetTangent'][ii_inlets_n]
                out_normals = -self.centerlines.point_data['FrenetTangent'][ii_outlets_n]
                
            points = np.concatenate([in_points, out_points], axis=0)
            normals = np.concatenate([in_normals, out_normals], axis=0)

            endpoints_pv = pv.wrap(points)
            endpoints_pv.point_data['Normals'] = normals

        surf_closed = self.surf.fill_holes(30.0).clean()
        select = endpoints_pv.select_enclosed_points(surf_closed, tolerance=0.00001, check_surface=False)

        # for origin, normal in zip(endpoints_pv.points, endpoints_pv.point_data['Normals']):
        #     t = TubeClipper(self.surf)
        #     t.clip(origin, normal)
        #     self.surf = t.far_side

        for sdx in range(select.n_points):
            check = select.point_data['SelectedPoints'][sdx]
            if check == True:
                origin = select.points[sdx]
                normal = select.point_data['Normals'][sdx]
                t = TubeClipper(self.surf)
                t.clip(origin, normal)
                self.surf = t.far_side
                
        old_surf = self.surf.copy()
        # self.surf = self.surf.triangulate()
        # self.surf = pv.PolyData(self.surf.points, self.surf.cells)

        # self.surf = self.surf.interpolate(old_surf)
        self.copy_arrays(old_surf, self.surf)
        
        #pretty sure there is no groupid issue without aneurysms?
        if include_aneurysms == True:
            self.surf = cc.fix_vmtk_group_ids(self.surf)
        return endpoints_pv

    def clip_endpoints_with_vmtk(self):
        surf = vmtk.surface_end_clipper(self.surf, centerlines=self.centerlines)
        return surf

    def set_inlets_outlets(self):
        """ Interactively choose inlet point.

        All other open boundaries will be considered outlets. To update these points 
        (for example, after adding flow extensions), see self.update_inlet_outlets. 
        To save these inlet-outlet points, see self.save_inlet_outlet_points()
        """
        text = "Pick the inlet(s)"
        centers = self.get_open_profiles()
        centers_m = pv.wrap(centers)
        inlet_ids = self._pick_points(centers_m, text=text)
        outlet_ids = list(set(range(centers_m.n_points)) - set(inlet_ids))
        
        self.inlet_ids = inlet_ids

        assert inlet_ids[0] is not None, "No inlet selected."

        self.inlet_points = [centers[i] for i in inlet_ids]
        self.outlet_points = [centers[i] for i in outlet_ids]
        return centers_m, inlet_ids, outlet_ids

    def update_inlets_outlets(self):
        """ Update inlet-outlet points based on distance metric.
        """
        centers = self.get_open_profiles()
        centers_m = pv.wrap(centers)
        tree = KDTree(centers)
        inlet_ids = [tree.query(i)[1] for i in self.inlet_points]
        outlet_ids = list(set(range(centers_m.n_points)) - set(inlet_ids))
        
        self.inlet_points = [centers[i] for i in inlet_ids]
        self.outlet_points = [centers[i] for i in outlet_ids]
        return centers_m, inlet_ids, outlet_ids

    def check_surface_topology(self):
        """ Warn about surface problems that break vmtkNetworkExtraction
        ("can't reconstruct new profile"). Only prints, never modifies the surface.

        Returns True if no problems were found.
        """
        surf = self.surf.triangulate()
        problems = []

        #Repeated triangles make every edge non-manifold
        faces = surf.faces.reshape(-1, 4)[:, 1:]
        n_dup = len(faces) - len(np.unique(np.sort(faces, axis=1), axis=0))
        if n_dup:
            problems.append(f'{n_dup} duplicate faces (see common.remove_duplicate_faces)')

        #Edges shared by more than two triangles
        non_manifold = surf.extract_feature_edges(boundary_edges=False, feature_edges=False,
                                                  manifold_edges=False, non_manifold_edges=True)
        if non_manifold.n_cells:
            problems.append(f'{non_manifold.n_cells} non-manifold edges')

        #Zero-area triangles give degenerate sphere intersections
        areas = surf.compute_cell_sizes(length=False, volume=False).cell_data['Area']
        n_degen = int(np.sum(areas < 1e-12 * areas.max()))
        if n_degen:
            problems.append(f'{n_degen} zero-area triangles')

        #Disconnected pieces (floating slivers left by the clip, etc.)
        n_regions = len(np.unique(surf.connectivity().point_data['RegionId']))
        if n_regions > 1:
            problems.append(f'surface has {n_regions} disconnected regions')

        #Each open boundary should be a simple closed loop: every boundary vertex on exactly two boundary edges
        edges = surf.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False)
        if edges.n_cells:
            _, degree = np.unique(edges.lines.reshape(-1, 3)[:, 1:], return_counts=True)
            n_bad = int(np.sum(degree != 2))
            if n_bad:
                problems.append(f'{n_bad} boundary vertices not on a simple closed loop (jagged/pinched clip edge)')
            n_loops = len(np.unique(edges.connectivity().point_data['RegionId']))
            if self.inlet_points is not None and self.outlet_points is not None:
                n_expected = len(self.inlet_points) + len(self.outlet_points)
                if n_loops != n_expected:
                    problems.append(f'{n_loops} open boundaries but {n_expected} inlets + outlets (holes in the surface?)')
        else:
            problems.append('no open boundaries')

        if problems:
            print('WARNING: surface problems that can cause "can\'t reconstruct new profile" in the network extractor:')
            for p in problems:
                print(f'  - {p}')
        return not problems

    def get_open_profiles(self):
        """ Get centers of open profiles
        """
        edges = self.surf.extract_feature_edges(
            boundary_edges=True, 
            feature_edges=False, 
            manifold_edges=False
            )
        if edges.n_points == 0:
            raise RuntimeError('No open boundaries found on the surface. Check that the clip actually opened the vessel ends '
                               'and that the surface has no duplicate faces (see common.remove_duplicate_faces). '
                               'If a bad *_noext.vtp was saved, delete it so the clipping is redone.')
        edges = edges.connectivity()
        regions = np.unique(edges.point_data['RegionId'])
        masks = [edges.point_data['RegionId'] == r for r in regions]
        profiles = pv.MultiBlock([edges.extract_points(m) for m in masks])

        num_profiles = len(profiles)
        centers = [x.points.mean(axis=0) for x in profiles]
        centers = np.array(centers)

        return centers 

    def select_geodesic(self,title='Isolate aneurysms.', mask_name='Mask'):
        s = cc.SelectGeodesic(self.surf, scalars=mask_name)
        pts = s.interact(title=title)
        self.surf = s.mesh
        return pts


    def generate_centerlines(self, include_aneurysms=True, seed_selector='idlist', endpoints=0):
        """ Generate centerlines using VMTK.

        Watch the centerline smoothing at the end!! May need tweaking for your case.

        Consider moving this into vmtk_wrapper, the nearest ids works well.
        """
        surf_capped = pv.PolyData()
        surf_capped.copy_structure(vmtk.surface_capper(self.surf))
        tree = KDTree(surf_capped.points)
        inlet_ids = [tree.query(i, k=1)[1] for i in self.inlet_points]
        outlet_ids = [tree.query(o, k=1)[1] for o in self.outlet_points]
    
        self.inlet_ids = inlet_ids 
        self.outlet_ids = outlet_ids

        
        
        if include_aneurysms == True:
            self.aneurysm_ids = [tree.query(i, k=1)[1] for i in self.aneurysm_points]
            target_ids = self.outlet_ids + self.aneurysm_ids

            centerlines = vmtk.centerlines(
                surf_capped, 
                seed_selector=seed_selector, 
                src_ids=self.inlet_ids,
                target_ids=target_ids
                )
            centerlines = vmtk.centerlines_smooth(centerlines, iterations=100, sm_factor=0.1)
            self.centerlines_aneurysm = centerlines
            self.centerlines_aneurysm = vmtk.centerline_geometry(self.centerlines_aneurysm)

            ind = list(range(len(self.outlet_ids)))
            cline_temp = self.centerlines_aneurysm.extract_cells(ind)
            centerlines = pv.PolyData(cline_temp.points, lines=cline_temp.cells)
            for arr in cline_temp.point_data:
                centerlines.point_data[arr] = cline_temp.point_data[arr]
            for arr in cline_temp.cell_data:
                centerlines.cell_data[arr] = cline_temp.cell_data[arr]
            self.centerlines = centerlines
            
        else:
            target_ids = self.outlet_ids 

            centerlines = vmtk.centerlines(
                surf_capped, 
                seed_selector=seed_selector, 
                src_ids=self.inlet_ids,
                target_ids=target_ids,
                endpoints=endpoints
                )
            #NOTE: If your vessels are really tortuous, you are going to want to turn the 
            # smoothing wayyy down...    
            #centerlines = vmtk.centerlines_smooth(centerlines, iterations=100, sm_factor=0.1)

            self.centerlines = centerlines
            self.centerlines = vmtk.centerline_geometry(self.centerlines)
        
        return self 

    def generate_centerlines_multi(self, proj_dir, seed_selector='idlist'):
        '''
        """ Generate centerlines using VMTK
        Here, the inlets are iterated over to generate individual sets of centerlines.
        Then, the centerlines are all merged together into one single centerline.
        This is necessary for multiple inlets, and needs quite a lot of monitoring if you have 
        both multiple inlets and many outlets. It can make terrible centerlines if you are not careful.

        The only reason we need this is for tubeclipper to do a good job.
        """
        print("Generating multiple centerlines")
        surf_capped = pv.PolyData()
        surf_capped.copy_structure(vmtk.surface_capper(self.surf))
        tree = KDTree(surf_capped.points)
        inlet_ids = [tree.query(i, k=1)[1] for i in self.inlet_points]
        outlet_ids = [tree.query(o, k=1)[1] for o in self.outlet_points]
    
        self.inlet_ids = inlet_ids 
        self.outlet_ids = outlet_ids
        target_ids = self.outlet_ids 
        centerlines_unmerged_blocks = pv.MultiBlock()

        for idx,i_id in enumerate(inlet_ids):
            
            if idx<1:
                centerlines_unmerged = vmtk.centerlines(
                    surf_capped, 
                    seed_selector=seed_selector, 
                    src_ids=[i_id],
                    target_ids=target_ids,
                    )
                centerlines_unmerged.save(proj_dir/('centerline{}.vtp'.format(idx)))
                centerlines_unmerged_blocks.append(centerlines_unmerged)
            else:
                centerlines_unmerged = vmtk.centerlines(
                    surf_capped, 
                    seed_selector=seed_selector, 
                    src_ids=[i_id],
                    #NOTE: this might create better centerlines if you try switching this to
                    #different outlet targets. Try a few and look at the output first.
                    target_ids=[0],
                    )
                centerlines_unmerged.save(proj_dir/('centerline{}.vtp'.format(idx)))
                centerlines_unmerged_blocks.append(vmtk.surface_append(centerlines_unmerged_blocks[idx-1],centerlines_unmerged))
                centerlines_appended = centerlines_unmerged_blocks[idx]
                #centerlines_appended.save(proj_dir/('appended_cl.vtp'))

        centerlines_branched = vmtk.centerline_branches_ids(centerlines_appended)
        centerlines_merged = vmtk.merge_centerlines(centerlines_branched)
        centerlines_total = vmtk.centerlines_smooth(centerlines_merged, iterations=1, sm_factor=0.01)

        self.centerlines = centerlines_total
        #centerlines_total.save(proj_dir/('merged_centerlines.vtp'))
        self.centerlines = vmtk.centerline_geometry(self.centerlines)
        '''
        """
        Generate centerlines using the network extractor. This won't cause merging issues.
        """
        self.check_surface_topology()
        self.centerlines, _ = vmtk.network_extractor(self.surf, ratio=1.01)
        self.centerlines = vmtk.centerline_geometry(self.centerlines)

        return self         
    # def _project_centerline_attrs(self, centerlines, centerlines_branched):
    #         centerlines_og = centerlines.copy()
    #         centerlines_branched = centerlines_branched.copy()

    #         # Convert cell data to point data
    #         centerlines_branched = centerlines_branched.ctp()

    #         # Create search tree for centerlines with GroupIds
    #         tree = KDTree(centerlines_branched.points)

    #         # Prepare points from original structure to query 
    #         q_points = centerlines_og.points
    #         dd, ii = tree.query(q_points, k=1)

    #         # Project onto centerlines_og
    #         centerlines_og.point_data['GroupIds'] = np.round(centerlines_branched.point_data['GroupIds'][ii]).astype(int)
    #         centerlines_og.point_data['Blanking'] = np.round(centerlines_branched.point_data['Blanking'][ii]).astype(int)

    #         return centerlines_og    
    
    def _project_centerline_attrs(self, centerlines, centerlines_branched):
            centerlines_og = centerlines.copy()
            centerlines_branched = centerlines_branched.copy()

            # Want a map from centerlines_og to centerlines_branched
            # But have to get cell data from the destination 

            group_ids = np.unique(centerlines_branched.cell_data['GroupIds'])
            segments = [centerlines_branched.extract_cells(centerlines_branched.cell_data['GroupIds'] == g) for g in group_ids]

            group_ids_interp = np.zeros(centerlines_branched.n_points, dtype=int)
            blanking_interp = np.zeros(centerlines_branched.n_points, dtype=int)

            tree = KDTree(centerlines_branched.points)

            for g, s in zip(group_ids,segments):
                # This will default rounding to higher GroupIds
                _, ii = tree.query(s.points, k=1)
                group_ids_interp[ii] = g
                blanking_interp[ii] = s['Blanking'][0]

            _, ii = tree.query(centerlines_og.points, k=1)
            centerlines_og.point_data['GroupIds'] = group_ids_interp[ii]
            centerlines_og.point_data['Blanking'] = blanking_interp[ii]

            # Project onto centerlines_og
            # centerlines_og.point_data['GroupIds'] = np.round(centerlines_branched.point_data['GroupIds'][ii]).astype(int)
            # centerlines_og.point_data['Blanking'] = np.round(centerlines_branched.point_data['Blanking'][ii]).astype(int)

            return centerlines_og    
    
    def branch_centerlines(self, project_back=True):

        self.centerlines_aneurysm_branched = vmtk.centerline_branches_ids(self.centerlines_aneurysm)

        if project_back:
            self.centerlines_aneurysm_split = self._project_centerline_attrs(
                self.centerlines_aneurysm, self.centerlines_aneurysm_branched)
        
        # This is the really slow step because of the glyphs.
        self.surf, self.neighbour_pt_ids = vmtk.surface_centerline_projection_MISR(
            self.surf, self.centerlines_aneurysm_branched, sm_iterations=1)
        
        # Check if this line is now redundant after adding new fix function.
        self.update_aneurysm_group_ids()

        # New fix function:
        self.surf = cc.fix_vmtk_group_ids(self.surf)

        return self

    def branch_centerlines_pt(self, project_back=True):
        print("\nBranching centerlines...")
        self.centerlines_branched = vmtk.centerline_branches_ids(self.centerlines)

        if project_back:
            self.centerlines_split = self._project_centerline_attrs(
                self.centerlines, self.centerlines_branched)
        
        self.surf, self.neighbour_pt_ids = vmtk.surface_centerline_projection_MISR(
            self.surf, self.centerlines_branched, sm_iterations=1)
        
        #print(self.surf.point_data)
        return self

    def update_aneurysm_group_ids(self):    
        tree = KDTree(self.surf.points)
        nearest_temp_idx = tree.query(self.aneurysm_points, k=1)[1]
        self.aneurysm_group_ids = [str(x) for x in self.surf.point_data['GroupIds'][nearest_temp_idx]]
        self.get_group_adjacency()
        # self.check_group_id_integrity()

        return self
           
    def extract_sacs(self):
        """ Redux based on new vmtk.surface_centerline_projection_MISR.
        """
        self.sacs = {}

        for g in self.aneurysm_group_ids:
            sac_mask = self.surf.point_data['GroupIds'] == g
            sac = self.surf.extract_points(sac_mask)
            sac_pd = pv.PolyData(sac.points, sac.cells)
            for arr in sac.point_data:
                sac_pd.point_data[arr] = sac.point_data[arr]
            for arr in sac.cell_data:
                sac_pd.cell_data[arr] = sac.cell_data[arr]
            self.sacs[g] = sac_pd

                
    def clip_boundaries(self, method='select', refine = False):
        """ Interactively clip branches then fix clip to normal 

        Args:
            method (str): 'select' uses common.ClickDragDelete tool,
                          'box' uses vmtk.clipper tool.
        """
        self.surf.clean()
        if method == 'select':
            cb = cc.ClickDragDelete(self.surf, title='Clip boundaries')
            surf = cb.mesh 
            surf = surf.triangulate()
            #print(surf.point_data)

            if type(surf) != pv.core.pointset.PolyData:
                surf = pv.PolyData(surf.points, surf.cells)

        else:   
            surf = self.surf.fill_holes(10.0)
            surf = vmtk.clipper(surf)

        surf = surf.connectivity(largest=True)
        surf = surf.clean() 

        self.surf = surf    
        #return cb.flag_inspect 

    def save_inlet_outlet_points(self, points_file, centerlines='centerlines', include_aneurysms=True, include_normals=False):
        """ Save inlet_points and outlet_points to a single h5 file.

        File keys will be "inlets" and "outlets"
        """ 
        points = pv.MultiBlock()
        points['inlets'] = pv.wrap(np.array(self.inlet_points))
        points['outlets'] = pv.wrap(np.array(self.outlet_points))
        if include_aneurysms==True:
            points['aneurysms'] = pv.wrap(np.array(self.aneurysm_points))

        if (hasattr(self, centerlines)) and (include_normals == True):
            tree = KDTree(self.centerlines.points)
            _, ii = tree.query(self.inlet_points)
            points['inlets'].point_data['normals'] = self.centerlines.point_data['FrenetTangent'][ii]
            _, ii = tree.query(self.outlet_points)
            points['outlets'].point_data['normals'] = self.centerlines.point_data['FrenetTangent'][ii]

        points.save(points_file)


    def pick_aneurysm(self, text='Pick aneurysm'):
        """ Pick a point on the dome.
        """
        def _point_picker_cb(mm, pid):
            # point = mm.points[pid]
            # print('Picked point', point)
            self._picked_ids.append(pid)
            picked_points = pv.wrap(mm.points[self._picked_ids])
            p.add_mesh(picked_points, name='picked_pts', color='r', point_size=10,
                render_points_as_spheres=True)

        def _reset_picked_cb():
            self._picked_ids = []
            p.remove_actor('picked_pts')

        self._picked_ids = []

        p = pv.Plotter()
        p.add_mesh(self.surf, color='w', opacity=1.0)

        p.add_text(text, position='upper_left')
        p.add_text('p: pick points', position=(0.05, 25), font_size=12)
        p.add_text('u: reset all picks', position=(0.05, 0.05), font_size=12)
        p.enable_point_picking(callback=_point_picker_cb, show_message=False, 
                            color='r', point_size=30, 
                            use_mesh=True, show_point=False, 
                            render_points_as_spheres=True,
                            tolerance=0.005)
        p.add_key_event('u', _reset_picked_cb)
        p.show()
        # print('picked', self._picked_ids)

        self.aneurysm_points = self.surf.points[self._picked_ids]

    def get_group_adjacency(self):
        """ Get adjlist and networkx DiGraph of centerlines.
        
        The centerlines should be passed before calling centerline branch ids.
        If getting errors, try recomputing centerlines -- there might be a bug
        where self.centerlines is overridden with centerlines_branched. 
        """
        if self.include_aneurysms==True:
            if hasattr(self, 'centerlines'):
                self.edges, self.G = vmtk.extract_group_adjacency(self.centerlines_aneurysm)
            if hasattr(self, 'centerlines'):
                _, self.G_no_aneurysm = vmtk.extract_group_adjacency(self.centerlines)
        else:
            if hasattr(self, 'centerlines'):
                self.edges, self.G = vmtk.extract_group_adjacency(self.centerlines)

    def get_bifurcation_ref_systems_vectors(self):
        # self.ref = vmtk.bifurcation_ref_systems(self.centerlines_branched)
        # self.bif_vec = vmtk.get_bifurcation_vectors(self.centerlines_branched, self.ref)

        # if hasattr(self, 'centerlines_aneurysm_branched'):
        if self.include_aneurysms==True:
            self.ref_aneurysm = vmtk.bifurcation_ref_systems(self.centerlines_aneurysm_branched)
            self.bif_vec_aneurysm = vmtk.get_bifurcation_vectors(self.centerlines_aneurysm_branched, self.ref_aneurysm)
        else: #May need this for inclusions
            self.ref = vmtk.bifurcation_ref_systems(self.centerlines_branched)
            self.bif_vec = vmtk.get_bifurcation_vectors(self.centerlines_branched, self.ref)

        # for b in self.bif_vec_aneurysm:

    def get_mean_segments(self):
        """ Get the "average" segment for each each group id in centerline
        """
        # Get group adjacency
        # edges, G = vmtk.extract_group_adjacency(self.centerlines_aneurysm)
        centerlines_split = self.centerlines_aneurysm_split

        # Get valid (non-blanked) ids
        mask = np.invert(centerlines_split.point_data['Blanking'].astype(bool))
        all_ids = np.unique(centerlines_split.point_data['GroupIds'])
        valid_ids = np.unique(centerlines_split.point_data['GroupIds'][mask])
        print('valid:', len(valid_ids), valid_ids)
        valid_ids = [x for x in valid_ids if str(x) in self.G.nodes]
        #print('valid2:', len(valid_ids), valid_ids)
        blanking_ids = np.unique(centerlines_split.point_data['GroupIds'][~mask])
        print("blankingids: ", len(blanking_ids), blanking_ids)
        # Split each centerline into segments based on groupIds
        segments = {key : [] for key in valid_ids}
        centerlines_multi = centerlines_split.split_bodies()

        # Create a new variable like "GroupIds" called "GroupIdsNonBlanking"
        # Anywhere a a group is blanked, relabel it with the successor's group id.
        for cline in centerlines_multi:
            cline.point_data['GroupIdsNonBlanking'] = cline.point_data['GroupIds'].copy()
            groups = np.unique([x for x in cline.point_data['GroupIds'] if x in all_ids])
            blanking_groups = [x for x in groups if x in blanking_ids]
            # blanking_successors = [list(self.G.successors(blanking_groups[i] - 1)) for i in range(len(blanking_groups))]
            blanking_masks = [cline.point_data['GroupIds'] == x for x in blanking_groups]
            blanking_diffs = [np.diff(x.astype(int)) for x in blanking_masks]
            blanking_successor_idx = [np.argmin(x) + 1 for x in blanking_diffs]
            blanking_successors = [cline.point_data['GroupIds'][x] for x in blanking_successor_idx]
            reassignments = blanking_successors #[[x for x in suc if x in groups][0] for suc in blanking_successors]

            masks = [cline.point_data['GroupIdsNonBlanking'] == x for x in blanking_groups]
            
            for msk, r in zip(masks, reassignments):
                cline.point_data['GroupIdsNonBlanking'][msk] = r

        # For each valid group id, extract lines associated with that group ID.
        # The items in segments are lists because multiple lines may have the same 
        # point id (to be averaged later)
        for cline in centerlines_multi:
            groups = np.unique([x for x in cline.point_data['GroupIdsNonBlanking'] if x in valid_ids])
            # print(groups)
            for gid in groups:
                mask = cline.point_data['GroupIdsNonBlanking'] == gid
                line_points = cline.extract_points(mask, adjacent_cells=False, include_cells=False)
                segments[gid].append(line_points)

        # Get mean of each groupIDs segment
        # Have to interpolate arrays onto mean segment
        mean_segments = {key : None for key in valid_ids}
        for key in valid_ids:
            line_groups = segments[key]

            points = [p.points for p in line_groups]
            
            n_points = np.max([p.n_points for p in line_groups])
            splines = [pv.Spline(p, n_points) for p in points]
            splines = [s.interpolate(l, strategy='closest_point') for s, l in zip(splines, line_groups)]

            # NOTE this is a hacky way of doing it. find diverging point
            # or use the bif vectors.
            if n_points > 20:
                back_off = 20
                mean_spline = np.mean([s.points[:-back_off] for s in splines], axis=0)
            else:
                mean_spline = np.mean([s.points for s in splines], axis=0)
            spline = pv.Spline(mean_spline, n_points)
            mean_segments[key] = spline


        # # Prepend last point in parent segment to child
        for key in valid_ids:
            parent = [int(x) for x in self.G.predecessors(str(key))]
            if len(parent) != 0:
                parent = parent[0]
            else:
                parent = None

            if parent is not None:
                # ref_group = parent + 1 
            
                # Prepend point
                pt = mean_segments[parent].points[-1].reshape(1,-1)
                segment = mean_segments[key].points
                points = np.concatenate([pt, segment], axis=0)
                spline = pv.Spline(points, segment.shape[0] + 1)
                spline.point_data['GroupIds'] = key
                mean_segments[key] = spline

        centerlines_multi_merge = centerlines_multi.combine()
        tree = KDTree(centerlines_multi_merge.points)

        for key in mean_segments.keys():
            # Transfer relevant arrays to new spline
            ndx = tree.query(mean_segments[key].points, k=1)[1]

            for arr in centerlines_multi_merge.point_data.keys():
                mean_arr = centerlines_multi_merge.point_data[arr][ndx] #p.mean([s.point_data[arr] for s in splines], axis=0)
                mean_segments[key].point_data[arr] = mean_arr
           
            mean_segments[key].point_data['OriginalGroupIds'] = mean_segments[key].point_data['GroupIds'].copy()
            mean_segments[key].point_data['GroupIds'] = key
            mean_segments[key] = mean_segments[key].compute_arc_length()
            mean_segments[key]['arc_length_inv'] = mean_segments[key]['arc_length'][::-1]

        keys = list(mean_segments.keys())
        items = [mean_segments[x] for x in keys]
        keys = [str(x) for x in keys]

        self.mean_segments = dict(zip(keys, items))
        self.mean_segments = pv.MultiBlock(self.mean_segments)
        print(self.mean_segments)

    def get_branch_endpoints(self):
        """ Get endpoints by splitting up the surface.
        """
        self.clipping_points = {}

        groups = np.unique(self.surf.point_data['GroupIds'])
        #print(groups)
        masks = [self.surf.point_data['GroupIds'] == int(g) for g in groups]

        keys = [int(g) for g in groups]
        #print(keys)
        parts = [self.surf.extract_points(m) for m in masks]
        parts = [p.extract_largest() for p in parts]
        parts = [p.triangulate() for p in parts]
        parts = [pv.PolyData(p.points, faces=p.cells).clean() for p in parts]
        parts = dict(zip(keys, parts))

        for g in keys:
            line = self.mean_segments[g]
            line = line.clean(tolerance=0.05)
            part = parts[g]
            select = line.select_enclosed_points(part.fill_holes(20), tolerance=1e-5, check_surface=False)
            s_mask = select.point_data['SelectedPoints'] == 1

            if np.sum(s_mask) > 2:
                sub = pv.PolyData(select.points[s_mask])
                sub.point_data['FrenetTangent'] = select.point_data['FrenetTangent'][s_mask]
                sub.point_data['MaximumInscribedSphereRadius'] = select.point_data['MaximumInscribedSphereRadius'][s_mask]
                sub.point_data['arc_length'] = select.point_data['arc_length'][s_mask]
                sub.point_data['arc_length'] = sub.point_data['arc_length'] - sub.point_data['arc_length'][0]

                # Just not starting at endpoint
                new_start_index = 1
                new_end_index = len(sub.points) - 2

                # Extract slices of the part based on the centerline and normal.
                # If n_points == n_cells, the slice forms a loop
                # otherwise it's just a line.
                slice_condition = False
                new_start_index -= 1
                while not slice_condition:
                    new_start_index += 1
                    if new_start_index < sub.n_points:
                        slice1 = part.slice(normal=sub.point_data['FrenetTangent'][new_start_index],origin=sub.points[new_start_index])
                    else:
                        break

                    if slice1.n_points == 0:
                        slice_condition = False
                    else:
                        slice_condition = slice1.n_points == slice1.n_cells

                    if new_start_index > len(sub.points):
                        print('Broken loop!')

                # Then for the other end
                slice_condition = False
                new_end_index += 1
                while not slice_condition:
                    new_end_index -= 1
                    if new_end_index > 0:
                        slice2 = part.slice(normal=sub.point_data['FrenetTangent'][new_end_index],origin=sub.points[new_end_index])
                    else:
                        break

                    if slice2.n_points == 0:
                        slice_condition = False
                    else:
                        slice_condition = slice2.n_points == slice2.n_cells

                if len(sub.points[new_start_index:new_end_index]) > 2:
                    sub_new = pv.wrap(sub.points[new_start_index:new_end_index])
                    sub_new.point_data['Normals'] = sub.point_data['FrenetTangent'][new_start_index:new_end_index]
                    sub_new.point_data['arc_length'] = sub.point_data['arc_length'][new_start_index:new_end_index]

                    self.clipping_points[g] = {}
                    self.clipping_points[g]['start'] = pv.wrap(sub_new.points[0])
                    self.clipping_points[g]['start'].point_data['Normal'] = sub_new.point_data['Normals'][0].reshape(1,3)

                    self.clipping_points[g]['end'] = pv.wrap(sub_new.points[-1])
                    self.clipping_points[g]['end'].point_data['Normal'] = -sub_new.point_data['Normals'][-1].reshape(1,3) 


    def get_branch_endpoints_OLD(self):
        """ 
        This works and is robust, but super slow. Depreceate. 
        Might be better when groupids is screwy though.
        """
        self.clipping_points = {}
        cpos = None
        
        tree = KDTree(self.centerlines_aneurysm.points)

        # Remove short branches from search
        self.GG = self.G.copy()

        surf_d = cc.decimate_edge_length(self.surf, 0.8)
        surf_d = cc.copy_arrays(self.surf, surf_d)

        for g in self.GG.nodes:
            # print(g)
            self.clipping_points[g] = {}
            self.M = self.GG.copy()
            self.M.remove_edges_from(nx.selfloop_edges(self.M))

            # For each aneurysm, get children, parents and siblings
            children = sorted(self.M.successors(g))
            parent = sorted(self.M.predecessors(g)) 
            
            G = self.M.to_undirected()
            G.remove_node(g)
            S = [G.subgraph(c).copy() for c in nx.connected_components(G)]
            S = [list(x.nodes) for x in S]

            descendants = [x for x in S if len(set(children) - set(x)) != len(children)]
            other_fam = [x for x in S if x not in descendants]

            # Flatten list of lists
            descendants = [val for sublist in descendants for val in sublist]
            other_fam = [val for sublist in other_fam for val in sublist]

            if len(parent) > 0:
                siblings = sorted(self.M.successors(parent[0]))

                # Remove the branch from the list of siblings
                siblings = [x for x in siblings if x != g]
            else:
                siblings = []

            s_line = self.mean_segments[str(g)]

            if g not in self.aneurysm_group_ids: # Don't define an endpoint for aneurysms
                # March backward along branch until clipping condition with children
                t = TubeClipper(surf_d)
                idx = len(s_line.points) - 1

                # Make sure cross section doesn't intersect children
                if len(descendants) != 0:
                    check = True
                    while (check == True) and (idx > 2):
                        _, ii = tree.query(s_line.points[idx], k=1)
                        origin = self.centerlines_aneurysm.points[ii]
                        normal = self.centerlines_aneurysm.point_data['FrenetTangent'][ii]
                        t.clip(origin, -normal)
                        clipped = t.clipped
                        mask = clipped.point_data['Side'] == 1
                        clipped = clipped.extract_points(mask)
                        # clipped = clipped.clean()

                        check = np.any([int(s) in clipped.point_data['GroupIds'] for s in descendants])
                        # clipped.plot(color='w')#scalars='GroupIds')
                        # p = pv.Plotter()
                        # p.camera_position = cpos
                        # p.add_mesh(clipped, color='r')
                        # p.add_mesh(self.surf, color='w', opacity=0.1)
                        # p.show()
                        # cpos = p.camera_position

                        print('ch', g, check, idx)
                        if check == True:
                            idx -= 2

                else:
                    _, ii = tree.query(s_line.points[-1], k=1)
                    origin = self.centerlines_aneurysm.points[ii]
                    normal = self.centerlines_aneurysm.point_data['FrenetTangent'][ii]

                if (len(descendants) == 0) or (check == False):
                    self.clipping_points[g]['end'] = pv.wrap(origin)
                    self.clipping_points[g]['end'].point_data['Normal'] = -normal.reshape(1,3)
                # else:
                    # self.clipping_points[g]['end'] = None

            # Now march forward along branch from beginning
            t = TubeClipper(surf_d)

            # Get index of first item after blanking ends 
            # diff = np.diff(s_line.point_data['Blanking'])
            # BLANKING is janky don't use
            idx = 0 #np.argmax(diff == -1) 

            # Make sure it doesn't intersect with parent or siblings
            fam = other_fam #siblings + parent

            if len(fam) != 0:
                check2 = True

                while (check2 == True) and (idx < len(s_line.points)):
                    _, ii = tree.query(s_line.points[idx], k=1)
                    origin = self.centerlines_aneurysm.points[ii]
                    normal = self.centerlines_aneurysm.point_data['FrenetTangent'][ii]
                    t.clip(origin, normal)
                    clipped = t.clipped
                    mask = clipped.point_data['Side'] == 1
                    clipped = clipped.extract_points(mask)

                    check2 = np.any([int(s) in clipped.point_data['GroupIds'] for s in fam])
                    
                    # clipped.plot(color='b')#scalars='GroupIds')
                    # p = pv.Plotter()
                    # p.camera_position = cpos
                    # p.add_mesh(clipped, color='r')
                    # p.add_mesh(self.surf, color='w', opacity=0.1)
                    # p.show()
                    # cpos = p.camera_position
                    
                    print('ch2', g, check, idx)
                    if check2 == True:
                        idx += 2

            else:
                _, ii = tree.query(s_line.points[0], k=1)
                origin = self.centerlines_aneurysm.points[ii]
                normal = self.centerlines_aneurysm.point_data['FrenetTangent'][ii]

            if len(fam) == 0 or check2 == False:
                self.clipping_points[g]['start'] = pv.wrap(origin)
                self.clipping_points[g]['start'].point_data['Normal'] = normal.reshape(1,3)
            # else:
                # self.clipping_points[g]['start'] = None 

        self.valid_clipping_points = {}
        for g in self.clipping_points.keys():
            if 'start' in self.clipping_points[g].keys():
                if 'end' in self.clipping_points[g].keys():
                    self.valid_clipping_points[str(g)] = self.clipping_points[g]

        self.clipping_points = self.valid_clipping_points
        
        # del_keys = []
        # for g in self.clipping_points.keys():
        #     start = self.clipping_points[g]['start']
        #     end = self.clipping_points[g]['end']
        #     if start is None or end is None:
        #         del_keys.append(g)
        
        # for k in del_keys:
        #     del self.clipping_points[k]
            

    # def get_branch_endpoints_old(self, min_branch_length=100):
    #     """ 
    #     Want to get start/end points where, if you clipped the plane
    #     normal to the centerline, it would contain none of the children, 
    #     siblings, or parents.

    #     This is a very slow function, but robust.
    #     """
    #     self.clipping_points = {}
        
    #     tree = KDTree(self.centerlines_aneurysm.points)

    #     # Remove short branches from search
    #     self.GG = self.G.copy()

    #     # Iterate each group
    #     for g in self.GG.nodes:
    #         self.clipping_points[g] = {}

    #         # print('------- g', g)
    #         self.M = self.GG.copy()

    #         # Remove any self loops
    #         self.M.remove_edges_from(nx.selfloop_edges(self.M))

    #         # For each aneurysm, get children, parents and siblings
    #         children = sorted(self.M.successors(g))
    #         parent = sorted(self.M.predecessors(g))#[0]
            
    #         if len(parent) > 0:
    #             siblings = sorted(self.M.successors(parent[0]))

    #             # Remove the branch from the list of siblings
    #             siblings = [x for x in siblings if x != g]
    #         else:
    #             siblings = []

    #         s_line = self.mean_segments[g]

    #         if g not in self.aneurysm_group_ids: # Don't define an endpoint for aneurysms
    #             # March backward along branch until clipping condition with children
    #             t = TubeClipper(self.surf)
    #             idx = len(s_line.points) - 1

    #             # Make sure cross section doesn't intersect children
    #             if len(children) != 0:
    #                 check = True
    #                 while (check == True) and (idx > 2):
    #                     _, ii = tree.query(s_line.points[idx], k=1)
    #                     origin = self.centerlines_aneurysm.points[ii]
    #                     normal = self.centerlines_aneurysm.point_data['FrenetTangent'][ii]
    #                     t.clip(origin, -normal)
    #                     clipped = t.clipped
    #                     mask = clipped.point_data['Side'] == 1
    #                     clipped = clipped.extract_points(mask)

    #                     check = np.any([s in clipped.point_data['GroupIds'] for s in children])
                        
    #                     # print('ch', check)
    #                     if check == True:
    #                         idx -= 2
                
    #             else:
    #                 _, ii = tree.query(s_line.points[-1], k=1)
    #                 origin = self.centerlines_aneurysm.points[ii]
    #                 normal = self.centerlines_aneurysm.point_data['FrenetTangent'][ii]

    #             if (len(children) == 0) or (check == False):
    #                 self.clipping_points[g]['end'] = pv.wrap(origin)
    #                 self.clipping_points[g]['end'].point_data['Normal'] = -normal.reshape(1,3)
    #             else:
    #                 self.clipping_points[g]['end'] = None
    #         else:
    #             self.clipping_points[g]['end'] = None

    #         # Now march forward along branch from beginning
    #         t = TubeClipper(self.surf)

    #         # Get index of first item after blanking ends 
    #         # diff = np.diff(s_line.point_data['Blanking'])
    #         # BLANKING is janky don't use
    #         idx = 0 #np.argmax(diff == -1) 

    #         # Make sure it doesn't intersect with parent or siblings
    #         fam = siblings + parent
    #         if len(fam) != 0:
    #             check = True
    #             while (check == True) and (idx < len(s_line.points)):
    #                 _, ii = tree.query(s_line.points[idx], k=1)
    #                 origin = self.centerlines_aneurysm.points[ii]
    #                 normal = self.centerlines_aneurysm.point_data['FrenetTangent'][ii]
    #                 t.clip(origin, normal)
    #                 clipped = t.clipped
    #                 mask = clipped.point_data['Side'] == 1
    #                 clipped = clipped.extract_points(mask)

    #                 check = np.any([s in clipped.point_data['GroupIds'] for s in fam])
                    
    #                 # clipped.plot(scalars='GroupIds')
    #                 # print('ch2', check)
    #                 if check == True:
    #                     idx += 2

    #         else:
    #             _, ii = tree.query(s_line.points[0], k=1)
    #             origin = self.centerlines_aneurysm.points[ii]
    #             normal = self.centerlines_aneurysm.point_data['FrenetTangent'][ii]

    #         if len(fam) == 0 or check == False:
    #             self.clipping_points[g]['start'] = pv.wrap(origin)
    #             self.clipping_points[g]['start'].point_data['Normal'] = normal.reshape(1,3)
    #         else:
    #             self.clipping_points[g]['start'] = None 


    def mark_distance_from_sacs(self, n_spheres, min_branch_length=100):
        """ Get point n_spheres away from sacs.
    
        Be careful with index between mean_segments and digraphs!!

        """
        self.sac_zones = {}

        # Before looping, collapse short branch nodes to parent
        # We are exluding them from the search
        self.GG = self.G.copy()
        group_ids = list(self.mean_segments.keys())
        segment_lengths = [self.mean_segments[g].n_points for g in group_ids]

        for idx in range(len(group_ids)):
            g = group_ids[idx]
            l = segment_lengths[idx]

            if (l < min_branch_length) and (g not in self.aneurysm_group_ids):
                parent = sorted(self.GG.predecessors(g))
                if len(parent) > 0:
                    self.GG = nx.contracted_nodes(self.GG, parent[0], g)
                else:
                    self.GG.remove_node(g)

        for an_id in self.aneurysm_group_ids:
            self.sac_zones[an_id] = {}

            # For each an_id, collapse other aneurysm_ids
            # to exclude them from the search
            other_an_ids = [x for x in self.aneurysm_group_ids if x != an_id]
            self.M = self.GG.copy()
            for g in other_an_ids:
                parent = sorted(self.GG.predecessors(g))[0]
                self.M = nx.contracted_nodes(self.M, parent, g)

            # Remove any self loops
            self.M.remove_edges_from(nx.selfloop_edges(self.M))

            # For each aneurysm, get parents and siblings
            parent = sorted(self.M.predecessors(an_id))[0]
            siblings = sorted(self.M.successors(parent))

            # Remove the aneurysm from the list of siblings
            siblings = [x for x in siblings if x != an_id]
            print('sibs', siblings)

            for s in siblings:
                if s in self.clipping_points.keys():
                    self.sac_zones[an_id][s] = {}

                    # Get start and end points
                    start = self.clipping_points[s]['start']
                    end = self.clipping_points[s]['end']
                    pts = np.concatenate([start.points, end.points], axis=0)
                    tree = KDTree(self.mean_segments[str(s)].points)
                    _, ii = tree.query(pts, k=1)
                    idx = ii[0]
                    arc_vals = self.mean_segments[str(s)].point_data['arc_length'][ii]

                    s_line = self.mean_segments[str(s)]

                    n = 1
                    cpos = None
                    while (n <= n_spheres) and idx < ii[1]:
                        arc_val = s_line.point_data['arc_length'][idx]
                        MISR = s_line.point_data['MaximumInscribedSphereRadius'][idx]
                        next_val = arc_val + MISR
                        idx = np.argmin((s_line.point_data['arc_length'] - next_val)**2)
                        # print('Step', s, n)
                        # p = pv.Plotter()
                        # p.camera_position = cpos
                        # p.add_mesh(self.surf, color='w', opacity=0.1)
                        # p.add_mesh(pv.wrap(s_line.points[idx]), color='r')
                        # p.show()
                        # cpos = p.camera_position
                        n += 1
                    
                    if n < n_spheres:
                        idx = ii[1]

                    origin = pv.wrap(s_line.points[idx])
                    origin.point_data['Normal'] = -s_line.point_data['FrenetTangent'][idx].reshape(1,3)

                    self.sac_zones[an_id][s][n_spheres] = origin
                    self.sac_zones[an_id][s][n_spheres].relation = 'sibling'

            # Then parent
            self.sac_zones[an_id][parent] = {}
            
            # Get start and end points
            start = self.clipping_points[parent]['start']
            end = self.clipping_points[parent]['end']
            pts = np.concatenate([start.points, end.points], axis=0)
            tree = KDTree(self.mean_segments[str(parent)].points)
            _, ii = tree.query(pts, k=1)
            idx = ii[1]
            arc_vals = self.mean_segments[str(parent)].point_data['arc_length'][ii]

            s_line = self.mean_segments[str(parent)]

            n = 1
            while (n <= n_spheres) and idx > ii[0]:
                arc_val = s_line.point_data['arc_length'][idx]
                MISR = s_line.point_data['MaximumInscribedSphereRadius'][idx]
                next_val = arc_val - MISR
                idx = np.argmin((s_line.point_data['arc_length'] - next_val)**2)
                n += 1
            
            if n < n_spheres:
                idx = ii[0]

            origin = pv.wrap(s_line.points[idx])
            origin.point_data['Normal'] = s_line.point_data['FrenetTangent'][idx].reshape(1,3)

            self.sac_zones[an_id][parent][n_spheres] = origin
            self.sac_zones[an_id][parent][n_spheres].relation = 'parent'

    def mark_near_vessel_regions(self, mesh, n_spheres):
        """
        Be careful with index between mean_segments and digraphs!!
        """
        for an_id in self.aneurysm_group_ids:
            # First mask n_spheres away
            far = TubeClipper(mesh)
            near = TubeClipper(mesh)

            for s in self.sac_zones[an_id].keys():
                pt_far = self.sac_zones[an_id][s][n_spheres]

                far.clip(pt_far.points[0], pt_far.point_data['Normal'][0])

                if pt_far.relation == 'parent':
                    pt_near = self.clipping_points[s]['end']
                else:
                    pt_near = self.clipping_points[s]['start']
                
                near.clip(pt_near.points[0], -pt_near.point_data['Normal'][0])

            near = near.clipped
            far = far.clipped
            near.point_data['Side'] = ~near.point_data['Side'] 
            region = near.point_data['Side'] * far.point_data['Side'] 

            mesh.point_data['sac_zone_{:02d}'.format(int(an_id))] = region

        zone_arr_names = [x for x in mesh.point_data if 'sac_zone_' in x]
        zone_arrs = [mesh.point_data[a] for a in zone_arr_names]
        mesh.point_data['sac_zones'] = np.sum(zone_arrs, axis=0)

        # for idx, an_id in enumerate(self.aneurysm_group_ids):
            # mesh.point_data['sac_zones'][mesh.point_data['GroupIds'] == an_id] = idx + 3

    def get_plc_points(self, n_spheres):
        """ Get PLC points based on mark_near_vessel_regions.
        """
        self.plc_points = {}

        for an_id in self.aneurysm_group_ids:
            plc_pts = pv.MultiBlock()
            for relative in self.sac_zones[an_id].keys():
                pt_far = self.sac_zones[an_id][relative][n_spheres]

                if pt_far.relation == 'parent':
                    pt_near = self.clipping_points[relative]['end']
                else:
                    pt_near = self.clipping_points[relative]['start']

                plc_pts[relative] = pt_near #.append(pt_near)

            self.plc_pts = plc_pts # = pv.MultiBlock(plc_pts)

    def get_mutual_ancestors(self):
        """ For landmarking a given bifurcation. 
        """
        centers = self.get_open_profiles()
        centers_m = pv.wrap(centers)
        outlet_ids = self._pick_points(centers_m, text='Pick relevant outlets')
        outlet_points = [centers[i] for i in outlet_ids]
        mean_segments = pv.MultiBlock(self.mean_segments).combine()
        # mean_segments = self.mean_segments.combine()
        tree = KDTree(mean_segments.points)
        _, ii = tree.query(outlet_points)
        g_ids = mean_segments.point_data['GroupIds'][ii].astype(int)
        lowest_common = nx.lowest_common_ancestor(self.G, g_ids[0], g_ids[1])
        ancestors = nx.ancestors(self.G, lowest_common)
        fam = list(ancestors) + [lowest_common]
        artery = self.mean_segments[[str(x) for x in fam]]
        return artery, fam

if __name__ == "__main__":
    print('See example scripts directory')
