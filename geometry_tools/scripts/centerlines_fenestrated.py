"""
This file creates vmtk centerlines for a surface mesh with one or more fenestrations, the same way
centerlines_fixed.py does for non-fenestrated vessels: vmtknetworkextraction gives the layout of the
vessel network, and vmtkcenterlines is then run along every network segment to get accurate centerlines.

vmtkcenterlines finds the cheapest path between two seeds, so when both sides of a fenestration start and
end at the same two junctions it traces the wider side twice and the narrower side is lost. To avoid
this, each segment is traced through extra seeds taken from its own network line. Segments that are part
of a loop always get a seed halfway along, and any traced piece that strays onto a different network
segment is split in half again until it follows its own segment.

The number of loops in the network is checked against the number of holes through the surface (its
genus), which is the number of fenestrations. Whether vmtknetworkextraction finds both sides of a
fenestration depends on the ratio, so if the given ratio doesn't match, a few others are tried. Extra
loops are network extraction artifacts (common in flat vessels), so loop sides whose centerline just
repeats another centerline are dropped.

Call this using:

centerlines_fenestrated.py prep_dir case_name [edge_length] [iterations] [ratio] [resample_step_length]

or with no arguments to use the cfen settings in config.py. prep_dir is the directory with the surface
file (should end in "cl.vtp"). The output files have the same names as the ones from centerlines_fixed.py
so map_info_direct.py can be run straight after. Every centerline cell is one network segment, with cell
data 'SegmentId' and 'LoopId' (0 = not part of a loop, otherwise which fenestration it belongs to).
"""

import numpy as np
import pyvista as pv
from vmtk import vmtkscripts
from geometry_tools import vmtk_wrapper as vmtk
from scipy.spatial import cKDTree as KDTree
from pathlib import Path
import sys

RADIUS = 'MaximumInscribedSphereRadius'
TRACE_STEP = 0.2 #Point spacing used while tracing and checking segments, before the final resampling
RATIOS_TO_TRY = [1.01, 1.05, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6] #Network extraction ratios tried when the given one misses a fenestration

def split_lines(lines):
    '''Point ids of every cell in a pyvista lines array'''
    cells, i = [], 0
    while i < len(lines):
        n = lines[i]
        cells.append(lines[i+1:i+1+n])
        i += n+1
    return cells

def polylines(point_lists, radius_lists):
    '''One PolyData with a polyline cell for each list of points'''
    lines, start = [], 0
    for pts in point_lists:
        lines += [len(pts)] + list(range(start, start+len(pts)))
        start += len(pts)
    lines_pd = pv.PolyData(np.vstack(point_lists), lines=np.array(lines))
    lines_pd.point_data[RADIUS] = np.concatenate(radius_lists)
    return lines_pd

def arc_length(points):
    return np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]

def point_at(points, frac):
    '''Point a fraction of the way along a polyline, by arc length'''
    s = arc_length(points)
    return np.array([np.interp(frac*s[-1], s, points[:, k]) for k in range(3)])

def surface_genus(surf):
    '''
    Number of holes through the surface (= number of fenestrations) and number of openings,
    from the Euler characteristic of the surface: V - E + F = 2 - 2*genus - openings
    '''
    n_edges = surf.extract_all_edges().n_cells
    openings = surf.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False, non_manifold_edges=False)
    n_open = len(np.unique(openings.connectivity().point_data['RegionId'])) if openings.n_points else 0
    chi = surf.n_points - n_edges + surf.n_cells
    return (2 - n_open - chi)//2, n_open

def network_graph(network, tol):
    '''
    Split the network into segments and merge the segment end points into graph nodes. The network's
    Topology array can't be used for this since it gives the same node number to different points.
    Returns the segments, the pair of nodes at the ends of each segment, and the node positions
    '''
    segments = [(network.points[ids], network.point_data[RADIUS][ids]) for ids in split_lines(network.lines)]
    ends = np.array([[pts[0], pts[-1]] for pts, _ in segments]).reshape(-1, 3)
    node_of_end = -np.ones(len(ends), dtype=int)
    nodes = []
    for i, p in enumerate(ends):
        if node_of_end[i] >= 0:
            continue
        close = (np.linalg.norm(ends - p, axis=1) < tol) & (node_of_end < 0)
        node_of_end[close] = len(nodes)
        nodes.append(ends[close].mean(axis=0))
    return segments, node_of_end.reshape(-1, 2), np.array(nodes)

def node_components(edges, keep, n_nodes):
    '''Label every node with the connected piece of the graph it is in, using only the edges where keep is True'''
    label = np.arange(n_nodes)
    changed = True
    while changed:
        changed = False
        for u, v in edges[keep]:
            m = min(label[u], label[v])
            if label[u] != m or label[v] != m:
                label[u] = label[v] = m
                changed = True
    return label

def find_loops(edges, active, n_nodes):
    '''
    LoopId for every edge (0 if the edge isn't part of a loop, otherwise which loop it is in) and the number
    of independent loops in the graph. Edges where active is False are left out
    '''
    in_loop = np.zeros(len(edges), dtype=bool)
    for i, (u, v) in enumerate(edges):
        if active[i]:
            keep = active.copy()
            keep[i] = False
            label = node_components(edges, keep, n_nodes)
            in_loop[i] = label[u] == label[v] #Still connected without this edge, so it is part of a loop
    #Loop edges that share nodes belong to the same loop
    label = node_components(edges, in_loop, n_nodes)
    loop_labels = sorted(set(label[edges[in_loop, 0]]))
    loop_ids = np.array([loop_labels.index(label[u])+1 if in_loop[i] else 0 for i, (u, v) in enumerate(edges)])

    #Independent loops = edges - nodes + connected pieces
    used_nodes = np.unique(edges[active])
    label = node_components(edges, active, n_nodes)
    n_loops = active.sum() - len(used_nodes) + len(np.unique(label[used_nodes]))
    return loop_ids, n_loops

class SegmentTracer():
    '''
    Runs vmtkcenterlines between two points inside the vessel. The Delaunay tessellation and Voronoi
    diagram are computed once and reused for every segment, rather than on every call like
    vmtk_wrapper.centerlines does
    '''
    def __init__(self, surf):
        #Same small perturbation as vmtk_wrapper.centerlines, to avoid the vmtkcenterlines bug with perfectly straight flow extensions
        #Copy first, compute_normals shares its points with the input so this would move the caller's surface too
        surf = surf.copy().compute_normals()
        noise = np.random.RandomState(0).normal(0, 0.005, surf.n_points)
        surf.points = surf.points + surf.point_data['Normals']*noise[:, None]
        self.surf = surf

        print("Computing the Delaunay tessellation and Voronoi diagram...")
        dv = vmtkscripts.vmtkDelaunayVoronoi()
        dv.Surface = surf
        dv.Execute()
        self.delaunay, self.voronoi, self.pole_ids = dv.DelaunayTessellation, dv.VoronoiDiagram, dv.PoleIds
        #Pole of a surface point = centre of the largest inscribed sphere touching it, so these sit along the middle of the lumen
        poles = [self.pole_ids.GetId(i) for i in range(surf.n_points)]
        self.pole_tree = KDTree(pv.wrap(self.voronoi).points[poles])

    def seed(self, point):
        '''Surface point whose pole is closest to point, so the path starts/ends at that spot in the lumen and not at the wall'''
        return int(self.pole_tree.query(point)[1])

    def trace(self, start, end):
        '''Centerline points and radii from start to end, or None if vmtk couldn't find a path'''
        cl = vmtkscripts.vmtkCenterlines()
        cl.Surface = self.surf
        cl.SeedSelectorName = 'idlist'
        cl.SourceIds = [self.seed(start)]
        cl.TargetIds = [self.seed(end)]
        cl.Resampling = 1
        cl.ResamplingStepLength = TRACE_STEP
        cl.AppendEndPoints = 0
        cl.StopFastMarchingOnReachingTarget = 1
        cl.DelaunayTessellation = self.delaunay
        cl.VoronoiDiagram = self.voronoi
        cl.PoleIds = self.pole_ids
        cl.Execute()
        path = pv.wrap(cl.Centerlines)
        if path.n_points < 2:
            return None
        pts, r = np.array(path.points), np.array(path.point_data[RADIUS])
        if np.linalg.norm(pts[0]-start) > np.linalg.norm(pts[-1]-start): #vmtk traces from the target back to the source
            pts, r = pts[::-1], r[::-1]
        return pts, r

class SegmentCenterlines():
    '''Traces every network segment with vmtkcenterlines and checks each piece stays on its own segment'''
    def __init__(self, tracer, segments, edges, nodes, max_splits, stray_fraction):
        self.tracer = tracer
        self.segments = segments
        self.edges = edges
        self.nodes = nodes
        self.max_splits = max_splits
        self.stray_fraction = stray_fraction

        #Lookups to tell which network segment a point is on
        self.net_tree = KDTree(np.vstack([pts for pts, _ in segments]))
        self.net_seg = np.concatenate([np.full(len(pts), i) for i, (pts, _) in enumerate(segments)])
        self.seg_trees = [KDTree(pts) for pts, _ in segments]
        #Network radius is 0 at the openings, so don't let the allowed distance go below half the typical radius
        self.seg_radius = [np.maximum(r, 0.5*np.median(r)) for _, r in segments]

    def stray(self, path, seg):
        '''Fraction of path points that sit on a different network segment instead of their own'''
        _, nearest = self.net_tree.query(path)
        d_own, i_own = self.seg_trees[seg].query(path)
        off = (self.net_seg[nearest] != seg) & (d_own > self.seg_radius[seg][i_own])
        return off.mean()

    def trace_piece(self, seg, a, b, depth):
        '''Trace the part of segment seg between fractions a and b of its length, splitting it in half if the path strays'''
        pts, r = self.segments[seg]
        path = self.tracer.trace(point_at(pts, a), point_at(pts, b))
        if path is not None and self.stray(path[0], seg) <= self.stray_fraction:
            return [path], False
        if depth == self.max_splits:
            #Out of splits, use the network line for this piece so the segment is still complete
            s = arc_length(pts)/arc_length(pts)[-1]
            keep = (s >= a) & (s <= b)
            piece = np.vstack([point_at(pts, a), pts[keep], point_at(pts, b)])
            return [(piece, np.interp(np.r_[a, s[keep], b], s, r))], True
        m = (a+b)/2
        first, fb1 = self.trace_piece(seg, a, m, depth+1)
        second, fb2 = self.trace_piece(seg, m, b, depth+1)
        return first + second, fb1 or fb2

    def trace_segment(self, seg, n_pieces):
        '''Centerline for one network segment, starting with n_pieces equal pieces'''
        pieces, used_fallback = [], False
        for k in range(n_pieces):
            p, fb = self.trace_piece(seg, k/n_pieces, (k+1)/n_pieces, depth=0)
            pieces += p
            used_fallback = used_fallback or fb
        #Join the pieces, each one starts where the last one ended so drop the repeated point
        pts = np.vstack([pieces[0][0]] + [p[1:] for p, _ in pieces[1:]])
        r = np.concatenate([pieces[0][1]] + [r[1:] for _, r in pieces[1:]])
        #Make sure the ends are exactly on the graph nodes so segments meet at the junctions
        u, v = self.edges[seg]
        pts[0], pts[-1] = self.nodes[u], self.nodes[v]
        return pts, r, len(pieces), used_fallback

def extract_network(surf, ratios, genus):
    '''
    Run vmtknetworkextraction with each ratio in turn until the network has one loop per fenestration.
    Network extraction is quick, but whether it finds both sides of a fenestration depends on the ratio.
    If none match, the network with too many loops is preferred (the extras can be removed after tracing)
    '''
    best = None
    for ratio in ratios:
        raw_network, _ = vmtk.network_extractor(surf, ratio=ratio)
        network = vmtk.resample_cl(raw_network, length=TRACE_STEP)
        segments, edges, nodes = network_graph(network, tol=1e-4*surf.length)
        _, n_loops = find_loops(edges, np.ones(len(edges), dtype=bool), len(nodes))
        print(f"Network with ratio {ratio}: {len(nodes)} nodes, {len(edges)} segments and {n_loops} loop(s)")
        score = (n_loops < genus, abs(n_loops - genus))
        if best is None or score < best[0]:
            best = (score, ratio, raw_network, segments, edges, nodes)
        if n_loops == genus:
            break
    return best[1:]

def coverage(path, radius, others):
    '''Fraction of path points that are inside another centerline (within the local radius of it)'''
    if not others:
        return 0.0
    d, _ = KDTree(np.vstack(others)).query(path)
    return (d < radius).mean()

def remove_duplicate_loops(traced, edges, active, n_nodes, genus, min_coverage=0.8):
    '''
    While the network has more loops than the surface has holes, drop the loop side whose centerline is
    most covered by the other centerlines, since it is a repeat of a vessel that has already been traced
    '''
    loop_ids, n_loops = find_loops(edges, active, n_nodes)
    while n_loops > genus:
        cover = {}
        for i in np.where((loop_ids > 0) & active)[0]:
            others = [traced[j][0] for j in np.where(active)[0] if j != i]
            cover[i] = coverage(traced[i][0], traced[i][1], others)
        worst = max(cover, key=cover.get)
        if cover[worst] < min_coverage:
            break
        print(f"Dropping segment {worst}: {cover[worst]*100:.0f}% of it repeats other centerlines")
        active[worst] = False
        loop_ids, n_loops = find_loops(edges, active, n_nodes)
    return loop_ids, n_loops

def make_cl(prep_dir, case_name, edge_length, iters, r, resample_step_length, max_splits=3, stray_fraction=0.2):
    out_dir = prep_dir.parent
    surf_file = sorted(prep_dir.glob('*cl.vtp'))[0]
    remeshed_file = out_dir/(case_name+'_remeshed.vtp')
    network_file = out_dir/(case_name+'_network.vtp')
    graphed_cl_file = out_dir/(case_name+'_cl_centerline_graph_vmtk.vtp')
    resampled_file = out_dir/(case_name+'_centerline_resampled.vtp')

    if not remeshed_file.exists():
        surf = pv.read(surf_file)
        surf = vmtk.surface_remeshing(surf, edgelength=edge_length, iterations=iters)
        surf.save(remeshed_file)
    else:
        surf = pv.read(remeshed_file) #Surface taking existing remeshed file
    surf = surf.triangulate().clean() #Cleaning up the surface just in case

    genus, n_open = surface_genus(surf)
    print(f"Surface has {n_open} openings and {genus} fenestration(s)")

    #Network layout of the vessels, trying the other ratios only if r doesn't find every fenestration
    ratios = [r] + [x for x in RATIOS_TO_TRY if x != r]
    r, network, segments, edges, nodes = extract_network(surf, ratios, genus)
    network.save(network_file)
    active = np.ones(len(edges), dtype=bool)
    loop_ids, n_loops = find_loops(edges, active, len(nodes))
    print(f"Using the network from ratio {r}")

    #Trace each segment, loop segments start out split in half so each side of the fenestration is followed
    tracer = SegmentTracer(surf)
    traced = SegmentCenterlines(tracer, segments, edges, nodes, max_splits, stray_fraction)
    results = []
    for seg in range(len(edges)):
        pts, rad, n_pieces, used_fallback = traced.trace_segment(seg, n_pieces=2 if loop_ids[seg] else 1)
        results.append((pts, rad))
        note = " - WARNING: part of it could not be traced and uses the network line instead" if used_fallback else ""
        print(f"Segment {seg} (nodes {edges[seg][0]}-{edges[seg][1]}, loop {loop_ids[seg]}): {n_pieces} piece(s){note}")

    if n_loops > genus:
        loop_ids, n_loops = remove_duplicate_loops(results, edges, active, len(nodes), genus)
    if n_loops != genus:
        reason = ("The extra loops did not repeat other centerlines closely enough to be removed" if n_loops > genus
                  else "A side of a fenestration was missed with every ratio tried")
        print(f"WARNING: the centerlines have {n_loops} loop(s) but the surface has {genus} fenestration(s). "
              f"{reason}. Check {network_file.name} and the centerlines in ParaView")

    keep = np.where(active)[0]
    centerline = polylines([results[i][0] for i in keep], [results[i][1] for i in keep])
    centerline.cell_data['SegmentId'] = keep
    centerline.cell_data['LoopId'] = loop_ids[keep]
    centerline = vmtk.resample_cl(centerline, length=resample_step_length)
    centerline = vmtk.centerline_geometry(centerline)
    centerline.save(graphed_cl_file)
    centerline.save(resampled_file)

    print(f"\nSaved {len(keep)} centerline segments to {graphed_cl_file}")
    for loop in range(1, loop_ids.max()+1):
        print(f"  Fenestration {loop}: segments {list(keep[loop_ids[keep] == loop])}")
    return centerline

def find_case_name(directory):
    '''If no case name is given, take the first stl file that can be found in the parent folder of the prep directory'''
    stl_files = Path(directory).glob("*.stl")
    return next(stl_files).stem

if __name__ == "__main__":
    if len(sys.argv) == 1:
        import config
        prep_dir = Path(config.cfen_prep_dir)
        print(f"Using the settings outlined in the config.py file. Prep Directory: {prep_dir}")
        case_name = config.cfen_case_name
        if case_name == "" or case_name == None:
            case_name = find_case_name(prep_dir.parent)
        edge_length = config.cfen_edge_length
        iters = config.cfen_iters
        ratio = config.cfen_ratio
        resample_step_length = config.cfen_resample_step_length
        max_splits = config.cfen_max_splits
        stray_fraction = config.cfen_stray_fraction
    else:
        if sys.argv[1] == "info":
            print(f'Usage of centerlines_fenestrated.py file:\npython centerlines_fenestrated.py [path/to/prep/dir] [case_name] [optional: edge_length] [optional: number_of_iterations] [optional: ratio] [optional: resample_step_length]')
            sys.exit()
        prep_dir = Path(sys.argv[1])
        case_name = sys.argv[2]
        edge_length = float(sys.argv[3]) if len(sys.argv) > 3 else 0.4
        iters = int(sys.argv[4]) if len(sys.argv) > 4 else 10
        ratio = float(sys.argv[5]) if len(sys.argv) > 5 else 1.01
        resample_step_length = float(sys.argv[6]) if len(sys.argv) > 6 else 1.5
        max_splits = 3
        stray_fraction = 0.2

    make_cl(prep_dir=prep_dir, case_name=case_name, edge_length=edge_length, iters=iters, r=ratio,
            resample_step_length=resample_step_length, max_splits=max_splits, stray_fraction=stray_fraction)
