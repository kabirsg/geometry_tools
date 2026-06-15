###############################
###     1. SURFACE PREP     ###
###############################
#surface_prep (sp)
sp_surf_file = ""
sp_proj_dir = ""
sp_surf_type = "pt"

#######################################
###     2. CENTERLINE GENERATION    ###
#######################################
#centerlines_fixed (cf)
cf_prep_dir = sp_proj_dir #Surface_prep project directory should have clipped files
cf_case_name = ""
cf_edge_length = 0.4 #Edge length of the remeshed surface - Default is 0.4
cf_iters = 10 #Number of iterations for surface remeshing - Default is 10
cf_ratio = 1.01 #Ratio between the sphere step and the local maximum radius
cf_resample_step_length = 1.5 #Primary parameter for density - Default is 1.5 -> places a centerline point every _ [units of the file] along the centerline

#centerlines_single (cs)
cs_proj_dir = sp_proj_dir
cs_case_name = ""

###########################
###     3. MAP INFO     ###
###########################
#map_info_direct (mid)
#Note: For flow rate parameters (except sss), if the vessel is not present, input "False" (with quotes)
mid_prep_dir = sp_proj_dir #Surface_prep project directory should have the clipped files
mid_sss = 6.816019210 #Superior Sagittal Sinus flow rate - required
mid_ss = "False" #Sigmoid Sinus flow rate
mid_lab = "False" #Labbe Flow rate
mid_syl = "False" #Sylvian vein flow rate
mid_emissary = "False" #Emissary vein flow rate
mid_condylar = "False" #Condylar vein flow rate
mid_fen = "False" #Boolean ("True"/"False") indicating presence of a fenestration stenosis


#map_info_bilateral (mib)
mib_prep_dir = sp_proj_dir #Directory storing prepped files
mib_sss = 6.816019219 #Superior Saggital Sinus peak systolic flow rate
mib_split = [0.5, 0.5] #The flow split at the Superior Saggital Sinus
mib_ss = ['False', 'r'] #Straight Sinus peak systole flow rate, and whether it is on the left ('l') or right ('r') side
mib_lab = ['False', 'False'] #Labbe vein(s) peak systolic flow rate
mib_syl = ['False', 'False'] #Sylvian vein(s) peak systolic flow rate
mib_emissary = ['False', 'False'] #Emissary vein(s) peak systolic flow rate
mib_condylar = ['False', 'False'] #Condylar vein(s) peak systolic flow rate
mib_fen = ['False', 'False'] #Boolean ('True'/'False') indicating presence of fenestration stenosis

###################################################
###     4. CHECK TAYLOR LENGTHS IN PARAVIEW     ###
###################################################


###########################
###     5. MAKE MESH    ###
###########################
#make_mesh (mm)
mm_proj_dir = sp_proj_dir #Directory storing project files
mm_proj_name = cf_case_name #Case name - default using the same as from centerline generation
mm_min_el = 0.2 #Minium mesh edge length - based on Taylor lengths
mm_max_el = 0.7 #Maximum mesh edge length - based on Taylor lengths
mm_multi_inlet = 'single' #'single' or 'Multi' inlets
mm_ref = 'False' #Refinement level

###############################
###     6. MESH QUALITY     ###
###############################
#meshquality (mq)
mq_file_name = "" #Mesh file path (.vtu file generation by make_mesh)