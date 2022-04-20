#!/usr/bin/env python
# coding: utf-8

# # Matching Zotu's to Consensus Sequences - Glucose Experiment
# willsharpless@berkeley.edu

# In[1]:

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from Bio import SeqIO
from Bio import pairwise2
from Bio.Align import substitution_matrices
from sklearn import manifold
from sklearn.cluster import KMeans
import operator

import scipy as scp
import networkx as nx

from matplotlib.backends.backend_pdf import PdfPages
from cycler import cycler
from PIL import Image

strict_binning = True
log_scale = False
pdf_filename = 'Glucose_sb_WAS0421.pdf'

# %%
tally_df = pd.read_csv('/Users/willsharpless/Documents/thesis/APA001/APA001.tsv', sep='\t')

tally_df = tally_df.rename(columns = {'index':'Well'})

# In[2]: Pairing Zotus to Consensus seqs

## Extract zotu's and consensus sequences
cons_path = "/Users/willsharpless/Documents/Thesis/Sequencing/ag_syncom_27F_1492R.fasta"
zotu_path = "/Users/willsharpless/Documents/Thesis/APA001/APA001.fna"

cons_seqs = SeqIO.parse(cons_path,"fasta")
zotu_seqs = SeqIO.parse(zotu_path,"fasta")

zotu_align_df = pd.DataFrame(columns = ["Zotu", "Best_Aligns", "Identity", "Score", "NextBest_Aligns", "Next_Identity", "Next_Score", "Tally_tot"])

## Iterate thru zotu's
for zotu in zotu_seqs:

    ## Align with all consensus sequences
    prf_dict = {}
    cons_seqs = SeqIO.parse(cons_path,"fasta") # stupid biopython only allows one iteration?
    for cons in cons_seqs:
        # Scoring like usearch: https://drive5.com/usearch/manual/aln_params.html
        a = pairwise2.align.globalms(zotu.seq, cons.seq, 1, -2, -10, -1)
        Score = a[0].score
        al = pairwise2.format_alignment(*a[0]).split("\n")[1]
        Identity = (len(al) - len(al.split(".")) + 1)/len(al)*100
        prf_dict[cons.id[:2]] = [Identity, Score]
        # print(prf_dict[cons.id[:2]])

    # print(prf_dict)
    ## Take top 2 alignments and add to df
    bestie = max(prf_dict.items(), key=operator.itemgetter(1))
    del prf_dict[bestie[0]]
    nextbestie = max(prf_dict.items(), key=operator.itemgetter(1))
    tally_tot = tally_df.loc[tally_df.Zotu == zotu.id]["count"].sum()
    row = [zotu.id, bestie[0], round(bestie[1][0],2), bestie[1][1], nextbestie[0], round(nextbestie[1][0],2), nextbestie[1][1], tally_tot]
    zotu_align_df.loc[len(zotu_align_df.index)] = row

zotu_align_df.head(10)

# %% Throwing out bad Zotu's

# 10 base difference from Sanger max (97.4% = 363/373 bases)
zotu_bin = zotu_align_df.loc[zotu_align_df.Identity >= 97.4]

if strict_binning == True:
    # 5 base difference from Sanger max (98.927% = 369/373 bases)
    zotu_bin = zotu_align_df.loc[zotu_align_df.Identity >= 98.928]

    # note, need to add FG which has no high quality Sanger
    zotu_bin = zotu_bin.append(zotu_align_df.loc[zotu_align_df.Best_Aligns == 'FG'])

# 2 base differentiation from next best (2/373 = 0.53%)
zotu_bin = zotu_bin.loc[zotu_align_df.Identity - zotu_align_df.Next_Identity >= 0.53]

# %% Projecting bins onto tally copy

# Order vital! Corresponds to # abbreviations 1 to 7
org_idx = ['RA', 'SK', 'MP', 'PK', 'BM', 'PA', 'FG']

# Tally-Binned Dataframe preallocation
tb_df = tally_df[:0]
tb_df = tb_df.drop(columns=['Zotu','count'])
tb_df = tb_df.rename(columns = {'primer_name':'Plate'})

plates = tally_df.primer_name.unique()
wells = tally_df.Well.unique()

plate_col, well_col = [],[]

for p in range(len(plates)):
    for w in range(len(wells)):
        plate_col.append(plates[p])
        well_col.append(wells[w])

tb_df.Plate = plate_col
tb_df.Well = well_col
tb_df.tot = 0
tb_df[org_idx] = 0

## Count corresponding Zotu's for each well
for i,row in tb_df.iterrows():

    tmp = tally_df.loc[(tally_df.primer_name == row.Plate) & (tally_df.Well == row.Well)]

    for j, trow in tmp.iterrows():
        
        # add total
        tb_df.at[i, 'tot'] = trow.tot

        # Match Zotu and add to correct org
        if trow.Zotu in zotu_bin.Zotu.unique():
            org = zotu_bin.loc[zotu_bin.Zotu == trow.Zotu, 'Best_Aligns'].item()
            tb_df.at[i, org] += trow['count']

for i,row in tb_df.iterrows():
    tb_df.at[i,'Plate'] = int(row.Plate[11:])
    tb_df.at[i, 'Well'] = int(row.Well[-2:])

tb_df.head()

# %% Import Extra Time Point 3 reads (muted in og sequencing)

extra_3_1_df = pd.read_pickle("extra_tp3_glucose_df.pkl")

tb_df = pd.concat([tb_df, extra_3_1_df]).groupby(['Plate','Well']).sum().reset_index()

# %% Make relative abundance df and pair names

ra_df = tb_df.copy()

def rel_abu(row):
    if row[2] > 0: return row[3:]/row[2]
    else: return row[3:]

ra_df.iloc[:,3:] = ra_df.apply(rel_abu, axis = 1, result_type='expand')

ra_df.head()

# min > 1e-4:
# ra_df[ra_df > 0.0000001].min(axis=0)

cycle_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = cycler('color', cycle_colors[:len(org_idx)])

tspan = np.linspace(24, 144, 6)
glabs = ["0.31 mM", "1 mM", "3.1 mM", "10 mM"]

pair_name = []
pair_name.append("Full Community:")

for x in range(7):
    for y in range(x,7):
        pair_name.append(org_idx[x] + ' - ' + org_idx[y] + ':')

# %% Plot

#region
# import itertools
# from cycler import cycler

# cycle_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# colors = cycler('color', cycle_colors[:len(org_idx)])

# tspan = np.linspace(24, 144, 6)
# glabs = ["0.31 mM", "1 mM", "3.1 mM", "10 mM"]
# # c = np.random.random((7, 3))
# # cmap = plt.get_cmap('gnuplot')
# # colors = itertools.cycle([cmap(i) for i in np.linspace(0, 1, len(org_idx))])
# # c = cm.rainbow(np.linspace(0, 1, len(org_idx)))

# fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(10,5), sharey='row')
# # ax.rc('axes', prop_cycle=colors)

# j=0
# for col in ax:
#     j+=1
#     for i in range(4):
#         tmp = ra_df.loc[(ra_df.Plate%4 == j%4) & (ra_df.Well ==9+i)].sort_values('Plate')
#         # k=13
#         # tmp = ra_df.loc[(ra_df.Plate%4 == j%4) & (ra_df.Well == k)].sort_values('Plate')
        
#         # plt.figure()
#         # plt.rc('axes', prop_cycle=colors)
#         col.set_prop_cycle(colors)
#         col.plot(tspan, tmp.iloc[:,3:])
#         col.set_title(glabs[j-1])
#         if j==3:
#             col.legend(org_idx)

# plt.suptitle("Full Comm");
#endregion

# %% Plot, Combine, Export as PDF

#region
# im_list = []

# def get_concat_v(im1, im2):
#     dst = Image.new('RGB', (im1.width, im1.height + im2.height))
#     dst.paste(im1, (0, 0), mask=im1.split()[3])
#     dst.paste(im2, (0, im1.height), mask=im2.split()[3])
#     return dst

# ## LOOP THRU PAIRS
# for k in range(len(pair_name)):

#     ## Individual Plots - One Plot
#     fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(10,5), sharey=True, sharex=True, dpi=400)
#     fig.patch.set_facecolor('white')
#     # ax.set_prop_cycle(colors)
#     for i in range(3):
#         for j in range(4):
#             tmp = ra_df.loc[(ra_df.Plate%4 == (j+1)%4) & (ra_df.Well == 10 + 3*k + i)].sort_values('Plate')
            
#             # plt.subplot(3, 4, 4*i + j + 1)
#             # plt.set_prop_cycle(colors)
#             ax[i,j].plot(tspan, tmp.iloc[:,3:])
#             ax[i,j].set_ylim(1e-4,1)

#             if log_scale == True:
#                 ax[i,j].set_yscale('log')

#             if i == 0:
#                 ax[i,j].set_title(glabs[j])
#             # if j==3:
#             #     col.legend(org_idx)

#     plt.setp(ax[-1, 0], xlabel='Hours')
#     plt.setp(ax[0, 0], ylabel='R1 Fraction')
#     plt.setp(ax[1, 0], ylabel='R2 Fraction')
#     plt.setp(ax[2, 0], ylabel='R3 Fraction')
#     plt.suptitle(pair_name[k] + " Individual Wells");
#     plt.savefig('image1.png')
#     plt.close

#     ## Replicate Plots
#     fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(10,5), sharey='row', dpi=400)
#     fig.patch.set_facecolor('white')

#     j=0
#     for col in ax:
#         j+=1
#         for i in range(3):
#             tmp = ra_df.loc[(ra_df.Plate%4 == j%4) & (ra_df.Well ==10 + 3*k + i)].sort_values('Plate')
            

#             col.set_prop_cycle(colors)
#             col.plot(tspan, tmp.iloc[:,3:])
#             col.set_title(glabs[j-1])
#             col.set_ylim(1e-4,1)

#             if log_scale == True:
#                 col.set_yscale('log')
            
#             if j==3:
#                 col.legend(org_idx)

#     plt.suptitle(pair_name[k] + " all");
#     plt.savefig('image2.png')
#     plt.close

#     im1 = Image.open('image1.png')
#     im2 = Image.open('image2.png')

#     cat = get_concat_v(im1, im2)
#     im_list.append(cat)

#     ## END LOOP OF PAIRS

# cat.save(pdf_filename, "PDF" ,resolution=100.0, save_all=True, append_images=im_list)
#endregion

# %% Fn for Extracting Training Data and plotting

def e_n_p(pair, actual_pair, 
                min_reads=20, min_abu_un=0.15, min_series=4, pair_name=pair_name, org_idx=org_idx, ra_df=ra_df, log_scale=True, plot_raw=False, late_start=False):

    # Input - 
        # pair: string of abbrev ors of pair well loc eg. "RA - PK"
        # actual_pair: string of abbrev ors of subcomm to train eg. "RA - PK - PA"
        # min_reads: Minimum number of reads to consider
        # min_abu_un: Minimum abundance to consider of non-subcomm orgs
        # min_series: Minimum length of timecourse with only subcomm orgs required to be accepted
        # late_start: mode which allows timecourse to start at late point with all subcomm orgs

    # Output - 
        # train_jl: (3 reps * subcomm sz) x (6 timepoints) x (4 conditions)
        # (plots)


    k=pair_name.index(pair+':')
    itdix = [ix for ix, o in enumerate(org_idx) if o in actual_pair.split()]
    itdl = [1 if o in actual_pair.split() else 0 for ix, o in enumerate(org_idx)]

    if pair == 'Full Community':
        itdix = [0, 1, 2, 3, 4, 5, 6]
        itdl = [1, 1, 1, 1, 1, 1, 1]

    # if any timecourses or timepoints are thrown out, leave as zero and julia code will ignore
    train_jl = np.zeros([3*sum(itdl),6,4])

    fw = 10 + 3*k
    prdf = ra_df.loc[(ra_df.Well >= fw) & (ra_df.Well < fw+3)].sort_values('Plate')

    if plot_raw == True:
        min_reads = 0
        min_abu_un = 0
        min_series = 0

    # clean any sample data where total reads <= 20
    prdf.at[prdf.tot < min_reads, prdf.columns[2:]] = 0

    # clean data of unintended orgs with less than 15% relative abundance for pairwise training
    for col in prdf:
        if col in org_idx and col not in actual_pair.split():
            coln = prdf[col].to_numpy()
            coln[coln < min_abu_un] = 0
            prdf[col] = coln

    # fix new corresponding relative abundances
    for i, row in prdf.iterrows():
        if sum(row[3:]) != 0:
            prdf.at[i, prdf.columns[3:]] = row[3:]/sum(row[3:])

    ## Individual Plots - One Plot
    fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(10,5), sharey=True, sharex=True, dpi=400)
    fig.patch.set_facecolor('white')

    for i in range(3):
        for j in range(4):
            tmp = prdf.loc[(prdf.Plate%4 == (j+1)%4) & (prdf.Well == 10 + 3*k + i)].sort_values('Plate')
            tmp = tmp.reset_index(drop=True)

            if late_start == True:
                ls = 0
                for ix,tp in tmp.iterrows():
                    all_itd_growing = set(np.where(tp[3:] > 0)[0]) == set(itdix)
                    if all_itd_growing:
                        gu = ls
                        break
            else:
                ls, gu = 0, 0

            for ix,tp in tmp.iterrows():
                itd_growing = set(np.where(tp[3:] > 0)[0]) <= set(itdix)
                all_dead = all(tp[3:] == 0)
                if itd_growing or all_dead:
                    gu += 1
                else:
                    break
            
            if plot_raw == True:
                gu = 6

            if gu >= min_series:
                ax[i,j].plot(tspan[ls:gu], tmp.iloc[ls:gu,3:])

            ax[i,j].set_ylim(1e-4,1)

            if log_scale == True:
                ax[i,j].set_yscale('log')
            
            if i == 0:
                ax[i,j].set_title(glabs[j])

    plt.setp(ax[-1, 0], xlabel='Hours')
    plt.setp(ax[0, 0], ylabel='R1 Fraction')
    plt.setp(ax[1, 0], ylabel='R2 Fraction')
    plt.setp(ax[2, 0], ylabel='R3 Fraction')
    plt.suptitle(pair_name[k] + " Individual Wells");
    plt.show()

    ## Replicate Plots
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(10,5), sharey='row', dpi=400)
    fig.patch.set_facecolor('white')

    j=0
    for col in ax:
        j+=1

        for i in range(3):
            tmp = prdf.loc[(prdf.Plate%4 == j%4) & (prdf.Well ==10 + 3*k + i)].sort_values('Plate')
            tmp = tmp.reset_index(drop=True)

            if late_start == True:
                ls = 0
                for ix,tp in tmp.iterrows():
                    all_itd_growing = set(np.where(tp[3:] > 0)[0]) == set(itdix)
                    if all_itd_growing:
                        gu = ls
                        break
            else:
                ls, gu = 0, 0

            for ix,tp in tmp.iterrows():
                itd_growing = set(np.where(tp[3:] > 0)[0]) <= set(itdix)
                all_dead = all(tp[3:] == 0)
                if itd_growing or all_dead:
                    gu += 1
                else:
                    break
            
            if plot_raw == True:
                gu = 6

            col.set_prop_cycle(colors)
            if gu >= min_series:
                col.plot(tspan[ls:gu], tmp.iloc[ls:gu,3:])
                
                # col.plot(tspan[:gu], tmp.iloc[:gu, 3:])
                
                # store in train_jl
                for c,ix in enumerate(itdix):
                    # train_jl[i*2 + c, :gu, j-1] = tmp.iloc[:gu, 3+ix]
                    train_jl[i*sum(itdl) + c, ls:gu, j-1] = tmp.iloc[ls:gu, 3+ix]
        
            col.set_title(glabs[j-1])
            col.set_ylim(1e-4,1)
            col.set_xlim(24,144)

            if log_scale == True:
                col.set_yscale('log')
            
            if j==3:
                col.legend(org_idx)

    plt.suptitle(pair_name[k] + " all");
    plt.show()

    return train_jl


# %% Pairs

pair = 'FG - FG' # intended well pair
actual_pair = 'PK - PA' # pair were getting data for
pkpa_g_data = e_n_p(pair, actual_pair, log_scale=False)

pair = 'RA - PK' # intended well pair
actual_pair = 'RA - PK' # pair were getting data for
rapk_g_data = e_n_p(pair, actual_pair, log_scale=False)

# %% Triplicates

pair = 'RA - FG' # intended well pair
actual_pair = 'RA - PK - PA' # pair were getting data for
rapkpa_g_data = e_n_p(pair, actual_pair, log_scale=False)

pair = 'SK - FG' # intended well pair
actual_pair = 'SK - PK - PA' # pair were getting data for
skpkpa_g_data = e_n_p(pair, actual_pair, log_scale=False)

pair = 'BM - FG' # intended well pair
actual_pair = 'PK - BM - PA' # pair were getting data for
pkbmpa_g_data = e_n_p(pair, actual_pair, log_scale=False)

pair = 'FG - FG' # intended well pair
actual_pair = 'PK - PA - FG' # pair were getting data for
pkpafg_g_data = e_n_p(pair, actual_pair, log_scale=False)

pair = 'MP - FG' # intended well pair
actual_pair = 'MP - PK - PA' # pair were getting data for
mppkpa_g_data = e_n_p(pair, actual_pair, log_scale=False)

# %% Full Community

pair = 'Full Community'
actual_pair = ':'
full_g_data = e_n_p(pair, actual_pair, plot_raw=True, log_scale=True)

# %% Export np arrays

dir_path = "/Users/willsharpless/Documents/ipy/Arkinlab/SynComm_Analysis" + "/RA_Data/"
ar_names = ["pkpa_g_data", "rapk_g_data", "rapkpa_g_data", "skpkpa_g_data", "pkbmpa_g_data", "pkpafg_g_data", "mppkpa_g_data", "full_g_data"]

for ix, ar in enumerate([pkpa_g_data, rapk_g_data, rapkpa_g_data, skpkpa_g_data, pkbmpa_g_data, pkpafg_g_data, mppkpa_g_data, full_g_data]):
    np.save(dir_path + ar_names[ix], ar)


# %%

full_g_data