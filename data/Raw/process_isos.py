
# %%

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.stats import linregress

orgs = ['RA', 'SK', 'MP', 'PK', 'BM', 'PA', 'FG']
conds = ["0p31mM", "1mM", "3p1mM", "10mM", "25C", "27p5C", "30C", "32p5C"]
colors = list(mcolors.TABLEAU_COLORS.items())
n_reps = 3
# n_tp = 491 #???
n_conds = int(len(conds))

os.chdir("/Users/willsharpless/Documents/Thesis/Data/Isogenic_curves/")

# %% Import excel files

dfs = {}
print("Reading...")
for file in os.listdir(os.getcwd()):
    if file.split('_')[0] != "iso" or file.split('.')[1] != "xlsx":
        continue
    cond = file.split('.')[0].split('_')[1]
    print(file)
    dfs[cond] = pd.read_excel(file)
    

# %% Plot if curious (fig plotting in julia)

ix_dict = {}

def plot_and_pull_ix(cond):
    df = dfs[cond]
    t_ar = df['Time [s]'].values

    #extracting a 7 tp for fitting
    t_fit = np.arange(0,49,8)
    ix_fit = np.zeros(len(t_fit),dtype=int)
    for tc, t in enumerate(t_fit):
        if t == 0:
            ix_fit[tc] = 3
            continue
        d_tfit = np.abs(t_ar/3600 - t)
        ix_fit[tc] = np.argmin(d_tfit)

    fig, ax = plt.subplots()
    for i_o, org in enumerate(orgs):
        c = colors[i_o][0]

        for rep_tag in ['','.1','.2']:
            org_tc = org + rep_tag
            od_ar = df[org_tc].values
            # ax.plot(t_ar, od_ar, color=c)

            t_ar_fit, od_ar_fit = np.zeros(len(t_fit)), np.zeros(len(t_fit))
            for i in range(len(t_fit)):
                ix = ix_fit[i]
                t_ar_fit[i], od_ar_fit[i] = t_ar[ix]/3600, od_ar[ix]
            ax.plot(t_ar_fit, od_ar_fit, '-o', color=c)
            ax.set_title(cond)
    return ix_fit, ax
    
fig, axs = plt.subplots(2,4)
for i_c, cond in enumerate(conds):
    ix_fit, ax = plot_and_pull_ix(cond)
    ix_dict[cond] = ix_fit
    print(int(np.floor(i_c/4)),i_c%4)
    axs[int(np.floor(i_c/4)),i_c%4] = ax[0]

plt.show()

# %% Transform into appropriate shape & export

def pull_cd(df, org, ix_fit):
    org_tc = [org + rep_tag for rep_tag in ['','.1','.2']]
    cond_data = df[org_tc].values.transpose()
    cond_data_fit = cond_data[:,ix_fit]
    return cond_data_fit

for org in orgs:
    org_arr = np.zeros((n_reps, 7, n_conds))

    for ic, cond in enumerate(conds):
        cd = pull_cd(dfs[cond], org, ix_dict[cond])
        org_arr[:,:,ic] = cd

    np.save("iso_"+org, org_arr)

# %% Test

a = np.load("iso_RA.npy")
a.shape
