#surface_prep (sp)
sp_surf_file = ""
sp_proj_dir = ""
sp_surf_type = ""

#centerlines_fixed (cf)
cf_prep_dir = sp_proj_dir #Surface_prep project directory should have clipped files
cf_case_name = ""
cf_edge_length = 0.4
cf_iters = 10
cf_ratio = 1.01
cf_resample_step_length = 1.5

#centerlines_single (cs)

#map_info_bilateral (mib)
mib_prep_dir = sp_proj_dir
mib_sss = 6.816019219
mib_ss = False
mib_split = [0.5, 0.5]
mib_lab = ['False', 'False']
mib_fen = ['False', 'False']
mib_syl = ['False', 'False']
mib_emissary = ['False', 'False']
mib_condylar = ['False', 'False']