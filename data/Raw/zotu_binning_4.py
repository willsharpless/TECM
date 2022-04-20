#!/usr/bin/env python
# coding: utf-8

# # Matching Zotu's to Consensus Sequences - extra 3_1 data
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
path2_fna_n_csv = '/Users/willsharpless/Documents/thesis/APA003/AA_WS_3_1' 

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

# %% Export the T3_1 dataframes

extra_3_1_df = tb_df.loc[(tb_df.Plate == 5) | (tb_df.Plate == 6) | (tb_df.Plate == 7) | (tb_df.Plate == 8)]
extra_3_1_df.to_pickle("extra_tp3_glucose_df.pkl")
# %%
