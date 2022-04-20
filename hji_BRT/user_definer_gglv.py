# %%

print("\n\nExecution Script for computing the BRT of Hybrid gLV systems: Glucose")
print("Utilizing Mo Chen's Optimized DP Toolbox")
print("willsharpless@berkeley.edu or ucsd.edu")
print("Created April 14, 2021")

import numpy as np
import os, math, time
from datetime import date, datetime
print("Executed",date.today().strftime("%B %d, %Y"))
import matplotlib.pyplot as plt

# import plotly.graph_objects as go
# from plotly.subplots import make_subplots

# from tvtk.api import tvtk
# import vtk
# from skimage import measure

# Mo's Scripts
# from Grid.GridProcessing import Grid
# from Shapes.ShapesFunctions import *
# from dynamics.GLV_3D import *
# from plot_options import *
# from Plots.plotting_utilities import *
# from solver import HJSolver

# %%

### Import a little data
if os.getcwd().split('/')[1] == "Users": #local
        home = "/Users/willsharpless/"; prepath = "/Documents/"
        od_path = home + prepath + "Thesis/pairwise_second/od_readings/"
else: # arkin server
        home = "/usr2/people/willsharpless"; prepath = "/" 
        od_path = home + prepath + "od_readings/"

to_arkin = home + prepath + "Julia/arkin/"

conds = np.load(od_path + "od_cond_order.npy")
concs = [0.31, 1., 3.1, 10.]

# %%

### General Specifications

""" USER INTERFACES
- Define grid
- Generate initial values for grid using shape functions
- Time length for computations
- Initialize plotting option
- Call HJSolver function
"""

max_abu = 0.5
target_set_radius = 0.05
dims = 3
qdims = 4 #np.size(p_full_all)[2]

### Look-back length and time step (hours)
t_step = 4; small_number = 1e-5; lookback_length = t_step
tau_1 = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

### Controller/Disturbance Specs
u_lim, d_lim = 5e-2, 2.5e-2 #0.0, 0.0 #5e-2, 2.5e-2 #if max_abu is ~0.25 corresponds to 20%/10% input/stochasticity
uMin, uMax = [0.0 for i in range(dims)], [0.0 for i in range(dims)]
dMin, dMax = [-d_lim,-d_lim,-d_lim,-d_lim], [d_lim, d_lim, d_lim, d_lim]

### Define hybrid switching schedules
n_sw = 3
switch_order = 5*np.ones((qdims**n_sw,n_sw), dtype=np.int8)
for i in range(qdims):
    for ii in range(qdims):
        for iii in range(qdims):
           switch_order[i*4**2 + ii*4 + iii,:] = np.array([i, ii, iii])

ctrl_spec_sets = [[0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]]
# %%

grid_res = 50 # pts/dims
g = Grid(np.array([0.0 for i in range(dims)]), 
         np.array([max_abu for i in range(dims)]), 
         dims, # dims
         np.array([grid_res for i in range(dims)]), # pts_each_dim
         [2]) # pDim ??

def volume(mg_X, mg_Y, mg_Z, my_V):
    data = (my_V <= 0).astype(np.int8)
    dX, dY, dZ = [np.diff(it, axis=a)[0,0,0] for it, a in zip((mg_X,mg_Y,mg_Z),(0,1,2))]
    
    ## giving me the biggest headache
    # data[data == 0] = -1
    # grid = vtk.vtkImageData(spacing=(dX, dY, dZ), origin=(mg_X.min(), mg_Y.min(), mg_Z.min()))
    # grid = vtk.vtkImageData()
    # grid.SetOrigin(mg_X.min(), mg_Y.min(), mg_Z.min())
    # grid.SetSpacing(dX, dY, dZ)
    # # grid.point_data.scalars = data.T.ravel() # It wants fortran order???
    # # grid.point_data.scalars.name = 'scalars'
    # grid.SetDimensions(data.shape)
    # iso = vtk.vtkImageMarchingCubes()
    # iso.SetInputData(grid)
    # mass = vtk.vtkMassProperties()
    # mass.SetInputData(iso)
    # return mass.volume

    coarse_volume = data.sum() * dX * dY * dZ
    return coarse_volume

# %%

### Iterate and compute BRT and volume of V=0 isosurface for all control species sets and concentrations

"""
Assign one of the following strings to `compMethod` to specify the characteristics of computation
"none" -> compute Backward Reachable Set
"minVWithV0" -> compute Backward Reachable Tube
"maxVWithVInit" -> compute max V over time
"minVWithVInit" compute min V over time
"""

# for cs in ctrl_spec_sets: # eg. [0, 1, 2] = all, [2] = only actuate 3rd species
## each takes about ~11 hours fyi

#     for s in cs:
#             uMin[s], uMax[s] = -u_lim, u_lim

#     ### Import gLV Data
#     rs, As, xds, comms, target_sets = [],[],[],[],[]
#     for c in range(len(conds)):
#             r = np.load(to_arkin + "r_A/r_" + conds[c].split()[0] + ".npy")
#             A = np.load(to_arkin + "r_A/A_" + conds[c].split()[0] + ".npy")
#             xd = -1*np.linalg.inv(A)@r
#             comm = gLV_3D(r, A.tolist(), xd.tolist(), 
#                                     uMin = uMin, uMax = uMax,
#                                     dMin = dMin, dMax = dMax)
#             target_set = CylinderShape(g, [], xd, target_set_radius)
            
#             rs.append(r); As.append(A); xds.append(xd); comms.append(comm); target_sets.append(target_set)

#     for i_xd in range(qdims):
#         print("\nDriving to xd of",conds[i_xd],"----------------------------------------------------------")
#         volumes = np.zeros(qdims**3)
#         for i_sw, sw in enumerate(switch_order):
#         # for i_sw in [0,21,42,63]:
#         # for i_sw in [0]:
#             sw = switch_order[i_sw, :]
#             start = time.time()
#             print("\n Computing BRT of Switch Order: " + " - > ".join(conds[sw][::-1]), "...") #ITERATING FORWARD ON BACKWARDS RECURSION
#             # title = conds[c] + ", Lookback = " + str(3*lookback_length) +" hours, |u|,|d| < " + str((u_lim, d_lim)) + ", controllable species " + str(controllable_species)
#             po = PlotOptions("3d_plot", [0,1,2], [0], plotting=False)
            
#             # result = HJSolver(comms[c], g, target_sets[c], tau_12, "minVWithV0", po)
#             # mg_X, mg_Y, mg_Z, my_V, fig = plot_isosurface(g, result, po)
#             # est_volume = volume(mg_X, mg_Y, mg_Z, my_V)

#             for i_c, c in enumerate(sw):
#                 if i_c == 0: #initialize
#                     result = HJSolver(comms[c], g, target_sets[i_xd], tau_1, "minVWithV0", po)
#                 else: #recurse
#                     result = HJSolver(comms[c], g, result, tau_1, "minVWithV0", po)

#             mg_X, mg_Y, mg_Z, my_V, fig = plot_isosurface(g, result, po)
#             volumes[i_sw] = volume(mg_X, mg_Y, mg_Z, my_V)
#             end = time.time()

#             print("   Volume of V=0 Isosurface", volumes[i_sw])
#             print("   done in", np.round((end - start)/60,1), "minutes")
#             np.save("hybrid_BRT_glucose_"+str(cs)+"_gLV_volumes"+conds[i_xd]+".npy", volumes)
#     #         # fig.show()
#     #         # break

#         print("\nwrote volumes to " + "hybrid_BRT_glucose_gLV_volumes"+conds[i_xd]+".npy\n")
#     print("\n Finished with control set " + str(cs))

# print("\n\nHe Terminado")
# print(datetime.now().strftime("%H:%M:%S"))
# print(date.today().strftime("%B %d, %Y"))

# %%

### Find largest BRT's
cs_data = {}

for cs in ctrl_spec_sets:
    
    print("\n\nWith control species set ", cs)

    best_so = np.ones((qdims, n_sw), dtype=np.int8)
    best_v = np.ones((qdims), dtype=np.float)
    static_so = np.repeat(np.array([[0, 1, 2, 3]], dtype=np.int8).T, n_sw,axis=1)
    static_v = np.ones((qdims), dtype=np.float)

    for i_xd in range(qdims):
        print("\nFor xd of", conds[i_xd])

        # sort by volume
        if cs == [0, 1, 2]:
            volumes = np.load(os.getcwd() + "/BRTs/hybrid_BRT_glucose_gLV_volumes"+conds[i_xd]+".npy")
        else:
            volumes = np.load(os.getcwd() + "/BRTs/hybrid_BRT_glucose_"+str(cs)+"_gLV_volumes"+conds[i_xd]+".npy")

        top_idx = np.argsort(volumes)[::-1]
        ranked_so = switch_order[top_idx,:]
        ranked_v = volumes[top_idx]

        # remove unrealistic, glucose-decreasing switch orders
        mask = np.zeros(len(ranked_v), dtype=bool)
        for i_s, so in enumerate(ranked_so):
            if np.all(np.sort(so[::-1]) == so[::-1]):
                mask[i_s] = True
        ranked_so = ranked_so[mask]
        ranked_v = ranked_v[mask]

        for rank in range(5):
            sw = ranked_so[rank,:]; vol = ranked_v[rank]
            print("   #",rank," Largest BRT results from " + " - > ".join(conds[sw][::-1]), " with V=0 volume", np.round(vol, 4))

        static_rank = np.where((ranked_so == np.repeat(i_xd,n_sw)).all(axis=1))[0][0]
        sw = ranked_so[static_rank,:]; 
        vol = ranked_v[static_rank]
        print("\n Static Schedule", " - > ".join(conds[sw]) ,"ranked #",static_rank+1,"/64 with V=0 volume", np.round(vol, 4))

        best_so[i_xd,:] = ranked_so[0,:] #store best schedule
        best_v[i_xd] = ranked_v[0] #store best schedule volume
        static_v[i_xd] = vol #store static volume

    cs_data[str(cs)] = (best_so, best_v, static_v)

np.save("hybrid_BRT_glucose_bvs_cs_data.npy", cs_data)
        # break

# %%

cs_data_ar = np.load("hybrid_BRT_glucose_bvs_cs_data.npy", allow_pickle=True)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

fig = plt.figure()
plt.title("Static vs Hybrid Glucose Environment")

for (csi, cs) in enumerate(ctrl_spec_set):

    if cs==[0, 1] or cs==[0, 2] or cs==[1, 2]:
        continue

    best_so, best_v, static_v = cs_data_ar [str(cs)]
    ratio_v = best_v/static_v
    
    if cs==[2]:
        plt.plot(concs, static_v, colors[csi], linestyle='solid', label="[2],[0,1],[0,2],[1,2] static")
        plt.plot(concs, best_v, colors[csi], linestyle='dashed', label="[2],[0,1],[0,2],[1,2] hybrid")
    else:
        plt.plot(concs, static_v, colors[csi], linestyle='solid', label=str(cs) + " static")
        plt.plot(concs, best_v, colors[csi], linestyle='dashed', label=str(cs) + " hybrid")
    
    # plt.plot(concs, ratio_v, label=str(cs))

plt.ylabel("Controllable State Volume (Abundance^3)")
# plt.ylabel("Best Hybrid/Static Volume Ratio")

plt.xlabel("Glucose (mM)")
plt.xscale('log')
plt.yscale('log')
plt.legend()

plt.savefig("/Users/willsharpless/Documents/Thesis/Write/Paper/figs/BRT_comp.png")

plt.show()
# %% 

### Recompute BRT of static and best switching schedules
# 
# volumes = np.zeros(2*qdims)
# conds_sh = np.array([c.split()[0] for c in conds])
# vol_mag = np.round(np.divide(best_v, static_v)).astype(int).astype(str)

# subplot_titles = []
# for i_s, schedules in enumerate([static_so, best_so]):
#     for i_xd, sw in enumerate(schedules):
#         name = "->".join(conds_sh[sw][::-1])
#         if i_s == 0:
#             subplot_titles.append(name)
#         else:
#             if i_xd == 0:
#                 subplot_titles.append(name + ", " + vol_mag[i_xd] +"x")
#             else:
#                 subplot_titles.append(name + ", ~" + vol_mag[i_xd] +"x")

# combined_fig = make_subplots(rows=2, cols=qdims, 
#                             subplot_titles=tuple(subplot_titles),
#                             specs=[[{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}],
#                                    [{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}]])

# for i_s, schedules in enumerate([static_so, best_so]):
#     for i_xd, sw in enumerate(schedules):

#         start = time.time()
#         print("\n Computing BRT of Switch Order: " + " - > ".join(conds_sh[sw][::-1]), "...") #ITERATING FORWARD ON BACKWARDS RECURSION
#         title = "Driving to x_e of " + conds_sh[i_xd] + ": " + " -> ".join(conds_sh[sw][::-1])
#         po = PlotOptions("3d_plot", [0,1,2], [0], plotting=False, title=title)

#         for i_c, c in enumerate(sw):
#             if i_c == 0: #initialize
#                 result = HJSolver(comms[c], g, target_sets[i_xd], tau_1, "minVWithV0", po)
#             else: #recurse
#                 result = HJSolver(comms[c], g, result, tau_1, "minVWithV0", po)

#         mg_X, mg_Y, mg_Z, my_V, fig = plot_isosurface(g, result, po)
#         volumes[i_xd + i_s*qdims] = volume(mg_X, mg_Y, mg_Z, my_V)
#         end = time.time()

#         print("   Volume of V=0 Isosurface", volumes[i_xd + i_s*qdims])
#         print("   done in", np.round((end - start)/60,1), "minutes")
#         # figs.append(fig)
#         # fig.write_image("BRT_test.png")
#         # break
    
#         name = "xe_"+ str(conds_sh[i_xd]) + "_schedule_" + "->".join(conds_sh[sw][::-1])
#         # fig.write_image(name+".jpeg")

#         # might not work
#         combined_fig.add_trace(go.Isosurface(
#                                 x=mg_X.flatten(),
#                                 y=mg_Y.flatten(),
#                                 z=mg_Z.flatten(),
#                                 value=my_V.flatten(),
#                                 opacity=0.7,
#                                 colorscale='blues',
#                                 # showscale=False,
#                                 isomin=-0.1,
#                                 surface_count=20,
#                                 isomax=0 #,
#                                 # caps=dict(x_show=True, y_show=True, z_show=True)
#                                ),
#                             row=i_s+1, col=i_xd+1)
        
#         # combined_fig.update_layout(scene = dict(
#         #             xaxis_title='RA',
#         #             yaxis_title='PK',
#         #             zaxis_title='PA'),
#         #             # height=700, width=700),
#         #             scene_camera = dict(eye=dict(x=-2, y=-2, z=0.1)),
#         #             margin=dict(l=0,r=0,b=0,t=0))

#         combined_fig.update_scenes(xaxis_title_text='RA',  
#                   yaxis_title_text='PA',  
#                   zaxis_title_text='PK')

#     # break

# combined_fig.show()
# combined_fig.write_image("allBRT.jpeg")

# %%

# # ### Combine into one plot
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots

# combined_fig = make_subplots(rows=2, cols=qdims)

# # figs = []
# # for i in range(8):
# #     figs.append(go.FigureWidget())

# for i_f, f in enumerate(figs):
#     combined_fig.add_trace(data=go.Isosurface(
#                                 x=mg_X.flatten(),
#                                 y=mg_Y.flatten(),
#                                 z=mg_Z.flatten(),
#                                 value=my_V.flatten(),
#                                 opacity=0.7,
#                                 colorscale='oranges',
#                                 isomin=0,
#                                 surface_count=1,
#                                 isomax=0,
#                                 caps=dict(x_show=False, y_show=False)
#                                ),
#                             row=int(math.floor(i_f/qdims)) + 1, col=(i_f)%4 + 1)

