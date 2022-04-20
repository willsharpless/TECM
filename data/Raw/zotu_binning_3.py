#!/usr/bin/env python
# coding: utf-8

# # Matching Zotu's to Consensus Sequences - VGVT, T0, extra 3_2 data
# willsharpless@berkeley.edu

# In[1]:

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from Bio import SeqIO
from Bio import pairwise2
from Bio.Align import substitution_matrices
import operator

import scipy as scp
import networkx as nx

from matplotlib.backends.backend_pdf import PdfPages
from cycler import cycler
from PIL import Image

strict_binning = True
log_scale = False
# pdf_filename = 'Temperature_log_sb_WS420.pdf'
path2_fna_n_csv = '/Users/willsharpless/Documents/thesis/APA003/AA_WS_3_2' 

# %%
tally_df = pd.read_csv(path2_fna_n_csv + '.tsv', sep='\t')

tally_df = tally_df.rename(columns = {'index':'Well'})

# %%
print("------  Check In-Line Mean Reads/Sample ------")
print("\n")

for R_primer in tally_df.primer_name.unique():
    # print(R_primer)
    tmp = tally_df.loc[tally_df.primer_name == R_primer]

    if tmp.tot.mean() < 500:
        warn = ' * * *'
    elif tmp.tot.mean() < 1000:
        warn = ' *'
    else:
        warn = ' '
    print(R_primer, " mean:", '%.1f' %tmp.tot.mean(),warn)

print("\n")
print("Total mean:", '%.1f' %tally_df.tot.mean(), 'reads/sample')
print("\n")

# In[2]: Pairing Zotus to Consensus seqs

## Extract zotu's and consensus sequences
cons_path = "/Users/willsharpless/Documents/Thesis/Sequencing/ag_syncom_27F_1492R.fasta"
zotu_path = path2_fna_n_csv + ".fna"

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
    # 5 base difference from Sanger max (98.93% = 369/373 bases)
    zotu_bin = zotu_align_df.loc[zotu_align_df.Identity >= 98.93]

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

# %% Export the T0 and T3_2 dataframes

extra_3_2_df = tb_df.loc[(tb_df.Plate == 5) | (tb_df.Plate == 6) | (tb_df.Plate == 7) | (tb_df.Plate == 8)]
extra_3_2_df.to_pickle("extra_tp3_temperature_df.pkl")

T0_df = tb_df.loc[tb_df.Plate == 24]
T0_df.to_pickle("T0.pkl")

# %% Make relative abundance df for VG and VT

ra_df = tb_df.loc[(tb_df.Plate == 22) | (tb_df.Plate == 23)]

def rel_abu(row):
    if row[2] > 0: return row[3:]/row[2]
    else: return row[3:]

ra_df.iloc[:,3:] = ra_df.apply(rel_abu, axis = 1, result_type='expand')

ra_df.head()

cycle_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = cycler('color', cycle_colors[:len(org_idx)])

tspan = np.linspace(0, 144, 7)
glab = ["0.31 mM -> 3.1 mM -> 0.31 mM"]
tlab = ["25 C -> 30 C -> 25 C"]

# %% Plot

#region

# from matplotlib.backends.backend_pdf import PdfPages
# from cycler import cycler
# import matplotlib.image as im

# cycle_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# colors = cycler('color', cycle_colors[:len(org_idx)])

# tspan = np.linspace(24, 144, 6)
# glabs = ["25 C", "27.5 C", "30 C", "32.5 C"]

# # with PdfPages('test2.pdf') as pdf:
# ## Individual Plots - One Plot
# fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(10,5), sharey=True, sharex=True)
# # ax.set_prop_cycle(colors)
# for i in range(3):
#     for j in range(4):
#         tmp = ra_df.loc[(ra_df.Plate%4 == j%4) & (ra_df.Well ==13+i)].sort_values('Plate')
        
#         # plt.subplot(3, 4, 4*i + j + 1)
#         # plt.set_prop_cycle(colors)
#         ax[i,j].plot(tspan, tmp.iloc[:,3:])
        
#         if i == 0:
#             ax[i,j].set_title(glabs[j-1])
#         # if j==3:
#         #     col.legend(org_idx)

# plt.setp(ax[-1, 0], xlabel='Hours')
# plt.setp(ax[0, 0], ylabel='R1 Fraction')
# plt.setp(ax[1, 0], ylabel='R2 Fraction')
# plt.setp(ax[2, 0], ylabel='R3 Fraction')
# plt.suptitle("FG " + "Individual Wells");
# # pdf.savefig()
# plt.savefig('image1.png')
# plt.close

# ## Replicate Plots
# fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(10,5), sharey='row')

# j=0
# for col in ax:
#     j+=1
#     for i in range(3):
#         tmp = ra_df.loc[(ra_df.Plate%4 == j%4) & (ra_df.Well ==13+i)].sort_values('Plate')

#         col.set_prop_cycle(colors)
#         col.plot(tspan, tmp.iloc[:,3:])
#         col.set_title(glabs[j-1])
#         if j==3:
#             col.legend(org_idx)

# plt.suptitle("FG" + " all");
# plt.savefig('image2.png')
# # pdf.savefig()
# plt.close

# with PdfPages('test3.pdf') as pdf:
#     im1 = im.imread('image1.png')
#     im2 = im.imread('image2.png')
#     stacked = np.concatenate((im1, im2))
#     fig = plt.figure(figsize=(12,12))
#     ax = plt.axes()
#     ax.imshow(stacked);
#     ax.set_xticks([])
#     ax.set_yticks([])
#     pdf.savefig()

#endregion

# %% Plot, Combine, Export as PDF

#region
# im_list = []

# def get_concat_v(im1, im2):
#     dst = Image.new('RGB', (im1.width, im1.height + im2.height))
#     dst.paste(im1, (0, 0), mask=im1.split()[3])
#     dst.paste(im2, (0, im1.height), mask=im2.split()[3])
#     return dst

# for x in range(7):
#     for y in range(x,7):
#         pair_name.append(org_idx[x] + ' - ' + org_idx[y] + ':')

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
                min_reads=20, min_abu_un=0.15, min_series=4, org_idx=org_idx, ra_df=ra_df, log_scale=True, plot_raw=False, late_start=False):

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

    if pair == 'Full Community':
        itdix = [0, 1, 2, 3, 4, 5, 6]
        itdl = [1, 1, 1, 1, 1, 1, 1]

    # if any timecourses or timepoints are thrown out, leave as zero and julia code will ignore
    train_jl = np.zeros([11*sum(itdl),7,2])

    if plot_raw == True:
        min_reads = 0
        min_abu_un = 0
        min_series = 0

    ## Replicate Plots
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5), sharey='row', dpi=400)
    fig.patch.set_facecolor('white')
    v_title = [glab, tlab]

    j=0
    for col in ax:
        j+=1

        prdf = ra_df.loc[(ra_df.Plate == 21 + j) & (ra_df.Well < 85)]

        # erase any sample data where total reads <= min_reads
        prdf.at[prdf.tot < min_reads, prdf.columns[2:]] = 0

        # erase data of unintended orgs with less than 15% relative abundance for clean pair
        for colu in prdf:
            if colu in org_idx and colu not in actual_pair.split():
                coln = prdf[colu].to_numpy()
                coln[coln < min_abu_un] = 0
                prdf[colu] = coln

        # fix new relative abundances
        for i, row in prdf.iterrows():
            if sum(row[3:]) != 0:
                prdf.at[i, prdf.columns[3:]] = row[3:]/sum(row[3:])

        for i in range(2,13):
            # tmp = prdf.loc[(prdf.Plate%4 == j%4) & (prdf.Well ==10 + 3*k + i)].sort_values('Plate')
            tmp = prdf.loc[prdf.Well%12 == i%12].sort_values('Plate')
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
                gu = 7

            col.set_prop_cycle(colors)
            if gu >= min_series:
                col.plot(tspan[ls:gu], tmp.iloc[ls:gu,3:])
                
                # store in train_jl
                for c,ix in enumerate(itdix):
                    train_jl[(i-2)*sum(itdl) + c, ls:gu, j-1] = tmp.iloc[ls:gu, 3+ix]
            
            col.set_title(v_title[j-1][0])
            col.set_ylim(1e-4,1)
            col.set_xlim(0,144)
            col.vlines([48, 96], color = 'k', ymin=1e-4, ymax = 1)

            if log_scale == True:
                col.set_yscale('log')
            
            if j==2:
                col.legend(org_idx)

    plt.suptitle("Full Community - Environment Variation");
    plt.show()

    return train_jl

# %% Full Community

pair = 'Full Community'
actual_pair = ':'
VGVT_data = e_n_p(pair, actual_pair, plot_raw=True, log_scale=False)

# %% Export np arrays

dir_path = "/Users/willsharpless/Documents/ipy/Arkinlab/SynComm_Analysis" + "/RA_Data/"
np.save(dir_path + "VGVT_data", VGVT_data)

# %%
