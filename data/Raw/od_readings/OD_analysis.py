#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

# In[2]:


def read_od_txt_f1(filename, skiprows, fmt):
#     files with first format include: all "_1_", and "1_2_", "2_2_", "3_2_" skiprows = 6
    data = pd.read_csv(filename, sep="	", skiprows=skiprows)
    plt_name_txt = filename.split("od_readings/",1)[1]
    plt_name = plt_name_txt[0:-13]
    
    if fmt==1:
        data.drop(["Unnamed: 97"], axis=1, inplace=True)
        data.drop(["Well"], axis=1, inplace=True)
        data["Label"] = plt_name
        data.set_index(["Label"], inplace=True)
    else:
        data = data.rename(columns={"Well":"Label","600":plt_name})
        data.set_index(["Label"], inplace=True)
        data = data.T
    
    return data

def read_od_xlsx(filename):
    plt_name_txt = filename.split("will_amd_reader/",1)[1]
    plt_name = plt_name_txt[0:-14]
    data = pd.read_excel(filename, sheet_name="Sheet2", skiprows=22, nrows=96)
    data = data.rename(columns={"<>":"Label","Value":plt_name})
    data = data[["Label",plt_name]]
    data.set_index(["Label"], inplace=True)
    data = data.T
    return data


# In[3]:


apareader_data = os.getcwd()
amdreader_data = apareader_data + "/will_amd_reader"
first=True

print("Reading ...")
for entry in os.scandir(apareader_data):
    
    if (entry.path.endswith(".txt")):
        print("     ", entry.path)
        
        if "_1_" in entry.path:
            df = df.append(read_od_txt_f1(entry.path, 2, 1))
            
        elif "1_2_" in entry.path or "2_2_" in entry.path or "3_2_" in entry.path:
            df = df.append(read_od_txt_f1(entry.path, 6, 1))
            
        else:
            if first:
                df = read_od_txt_f1(entry.path, 6, 2)
                first = False
            else:
                df = df.append(read_od_txt_f1(entry.path, 6, 2))

print("\n")
print("Reading ...")
for entry in os.scandir(amdreader_data):
    print("     ", entry.path)
    df = df.append(read_od_xlsx(entry.path))
    

df.index.name = "Label"
df.reset_index(inplace=True)
df["Plate"] = df.Label.apply(lambda x: x[5])
df["Hours"] = df.Label.apply(lambda x: (int(x[7])-1)*24)


# In[4]:


df.head(56)


# In[5]:


Gluc_0p31mM = df.loc[df['Plate'] == '1']
Gluc_0p31mM = Gluc_0p31mM.sort_values(by=["Hours"])

Gluc_1mM = df.loc[df['Plate'] == '2']
Gluc_1mM = Gluc_1mM.sort_values(by=["Hours"])

Gluc_3p1mM = df.loc[df['Plate'] == '3']
Gluc_3p1mM = Gluc_3p1mM.sort_values(by=["Hours"])

Gluc_10mM = df.loc[df['Plate'] == '4']
Gluc_10mM = Gluc_10mM.sort_values(by=["Hours"])

Temp_25 = df.loc[df['Plate'] == '5']
Temp_25 = Temp_25.sort_values(by=["Hours"])

Temp_27p5 = df.loc[df['Plate'] == '6']
Temp_27p5 = Temp_27p5.sort_values(by=["Hours"])

Temp_30 = df.loc[df['Plate'] == '7']
Temp_30 = Temp_30.sort_values(by=["Hours"])

Temp_30p5 = df.loc[df['Plate'] == '8']
Temp_30p5 = Temp_30p5.sort_values(by=["Hours"])


# In[6]:


Gluc_0p31mM.head(8)


# In[7]:


lets = ["A","B","C","D","E","F","G","H"]
dfs_gluc = {'0.31mM Glucose':Gluc_0p31mM, '1mM Glucose':Gluc_1mM, '3.1mM Glucose':Gluc_3p1mM, '10mM Glucose':Gluc_10mM}
dfs_gluc_keys = list(dfs_gluc.keys())
dfs_temp = {'25C':Temp_25, '27.5C':Temp_27p5, '30C':Temp_30, '32.5C':Temp_30p5}
dfs_temp_keys = list(dfs_temp.keys())


# In[8]:


for key,value in dfs_gluc.items():

    df_plot = value

    fig, ax = plt.subplots(figsize=(6,6))
    for i in range(0,8):
        for ii in range(0,12):
            wellc1 = lets[i]+str(ii+1)
            ax.plot(df_plot['Hours'], df_plot[wellc1], marker='o', linestyle='-')

    # ax.legend()
    ax.set_title('Growth in '+ key, color='black', fontsize='17', ha = 'center')
    ax.set_ylabel('OD', color = 'black', fontsize='13')
    ax.set_xlabel('Hours', color = 'black', fontsize='13')
    plt.show()


# In[9]:


for key,value in dfs_temp.items():

    df_plot = value

    fig, ax = plt.subplots(figsize=(6,6))
    for i in range(0,8):
        for ii in range(0,12):
            wellc1 = lets[i]+str(ii+1)
            ax.plot(df_plot['Hours'], df_plot[wellc1], marker='o', linestyle='-')

    # ax.legend()
    ax.set_title('Growth in '+ key, color='black', fontsize='17', ha = 'center')
    ax.set_ylabel('OD', color = 'black', fontsize='13')
    ax.set_xlabel('Hours', color = 'black', fontsize='13')
    plt.show()


# In[10]:


Gluc_0p31mM.head(8)


# In[11]:


# lets = ["B","C","D","E","F","G","H"] 
# dfc = Gluc_0p31mM

# c_7 = 0
# c_12 = 1
# for i in range(1,8):
#     for j in range(i,8):
#         pair = str(i)+"-"+str(j)
#         well1, well2, well3 = lets[c_7] + str(c_12), lets[c_7] + str(c_12 + 1), lets[c_7] + str(c_12 + 2)
#         dfc[pair] = (dfc[well1] + dfc[well2] + dfc[well3])/3
# #         print("Pair: ",pair," At: ",well1," ",well2," ",well3)
#         c_12 = c_12 + 3
#         if c_12 == 13:
#             c_12 = 1
#             c_7 = c_7 + 1


# In[12]:


lets = ["B","C","D","E","F","G","H"] 

for key,value in dfs_gluc.items():
    
    fig, ax = plt.subplots(figsize=(6,6))
    
    c_7 = 0
    c_12 = 1
    for i in range(1,8):
        for j in range(i,8):
            pair = str(i)+"-"+str(j)
            well1, well2, well3 = lets[c_7] + str(c_12), lets[c_7] + str(c_12 + 1), lets[c_7] + str(c_12 + 2)
            value[pair] = (value[well1] + value[well2] + value[well3])/3           
            # could do std dev here as well
    #         print("Pair: ",pair," At: ",well1," ",well2," ",well3)
    
            c_12 = c_12 + 3
            if c_12 == 13:
                c_12 = 1
                c_7 = c_7 + 1
            
            ax.plot(value['Hours'], value[pair], marker='o', linestyle='-')
    
    ax.set_title('Replicate-Averaged Growth in '+ key, color='black', fontsize='17', ha = 'center')
    ax.set_ylabel('OD', color = 'black', fontsize='13')
    ax.set_xlabel('Hours', color = 'black', fontsize='13')
    plt.show()


# In[13]:


lets = ["B","C","D","E","F","G","H"] 

for key,value in dfs_temp.items():
    
    fig, ax = plt.subplots(figsize=(6,6))
    
    c_7 = 0
    c_12 = 1
    for i in range(1,8):
        for j in range(i,8):
            pair = str(i)+"-"+str(j)
            well1, well2, well3 = lets[c_7] + str(c_12), lets[c_7] + str(c_12 + 1), lets[c_7] + str(c_12 + 2)
            value[pair] = (value[well1] + value[well2] + value[well3])/3           
            # could do std dev here as well
    #         print("Pair: ",pair," At: ",well1," ",well2," ",well3)
    
            c_12 = c_12 + 3
            if c_12 == 13:
                c_12 = 1
                c_7 = c_7 + 1
            
            ax.plot(value['Hours'], value[pair], marker='o', linestyle='-')
    
    ax.set_title('Replicate-Averaged Growth in '+ key, color='black', fontsize='17', ha = 'center')
    ax.set_ylabel('OD', color = 'black', fontsize='13')
    ax.set_xlabel('Hours', color = 'black', fontsize='13')
    plt.show()


# In[ ]:

# Tecan -> Synergy Standard Curve to apply to last two timepoints

synergy = "1.006	0.862	0.254	0.331	0.039	0.042	0.07	0.065	0.041	0.039	0.04	0.065	0.656	0.573	0.16	0.204	0.04	0.04	0.035	0.04	0.036	0.037	0.041	0.042	0.564	0.509	0.143	0.191	0.04	0.041	0.04	0.041	0.042	0.042	0.041	0.041	0.472	0.412	0.117	0.16	0.04	0.041	0.042	0.359	0.113	0.156	0.041	0.041	0.377	0.341	0.102	0.136	0.04	0.04	0.041	0.367	0.124	0.154	0.041	0.04	0.288	0.247	0.08	0.099	0.04	0.04	0.042	0.495	0.14	0.214	0.04	0.04	0.196	0.173	0.07	0.062	0.04	0.04	0.041	0.621	0.18	0.256	0.04	0.039	0.112	0.111	0.048	0.051	0.04	0.039	0.04	0.703	0.201	0.304	0.041	0.042"

tecan = "0.801	0.704	0.235	0.291	0.034	0.037	0.075	0.065	0.048	0.052	0.035	0.051	0.516	0.459	0.137	0.149	0.034	0.034	0.035	0.037	0.033	0.033	0.035	0.034	0.446	0.408	0.122	0.138	0.033	0.034	0.033	0.035	0.034	0.034	0.034	0.033	0.368	0.33	0.102	0.118	0.033	0.034	0.035	0.247	0.083	0.096	0.034	0.034	0.341	0.326	0.115	0.131	0.034	0.035	0.035	0.282	0.106	0.117	0.035	0.035	0.24	0.212	0.075	0.083	0.034	0.034	0.035	0.363	0.126	0.146	0.033	0.034	0.132	0.089	0.051	0.056	0.034	0.034	0.034	0.45	0.15	0.186	0.032	0.033	0.09	0.066	0.041	0.043	0.033	0.033	0.033	0.54	0.174	0.212	0.033	0.034"

synergy_data = [float(i) for i in synergy.split()]
tecan_data = [float(i) for i in tecan.split()]
curve_df = pd.DataFrame(synergy_data, tecan_data)
curve_df.sort_index(inplace=True)

fit = linregress(curve_df.index.values[40:80], curve_df.iloc[:,0].values[40:80])

# x = 0:0.025:0.8
x = np.linspace(0.03, 0.8)
plt.plot(x, fit.slope*x + fit.intercept, 'r', label="Fit: y="+str(round(fit.slope,3))+"x + "+str(round(fit.intercept,3)));
plt.scatter(curve_df.index.values, curve_df.iloc[:,0].values, label="data");
plt.xlabel("Tecan"); 
plt.ylabel("Synergy")
plt.legend()
plt.title("Tecan to Synergy Std Curve");

# In[14]:

orgs = ['RA', 'SK', 'MP', 'PK', 'BM', 'PA', 'FG']

# In[14]:

dfs_gluc_cleaned = dfs_gluc

for key, value in dfs_gluc_cleaned.items():
    
    # bound biofilm reads
    value.iloc[:,1:97] = value.iloc[:,1:97].clip(upper=0.4)

    # fix tecan reads
    value.iloc[-2:,1:97] = fit.slope*value.iloc[-2:,1:97].values + fit.intercept
    
    fig, ax = plt.subplots(figsize=(6,6))
    
    c_7 = 0
    c_12 = 1

    value["Full Community"] = (value["A10"] + value["A11"] + value["A12"])/3

    for i in range(1,8):
        for j in range(i,8):
            pair = orgs[i-1] + " - " + orgs[j-1]
            well1, well2, well3 = lets[c_7] + str(c_12), lets[c_7] + str(c_12 + 1), lets[c_7] + str(c_12 + 2)
            value[pair] = (value[well1] + value[well2] + value[well3])/3           
            # could do std dev here as well
    #         print("Pair: ",pair," At: ",well1," ",well2," ",well3)
    
            c_12 = c_12 + 3
            if c_12 == 13:
                c_12 = 1
                c_7 = c_7 + 1
            
            ax.plot(value['Hours'][:], value[pair][:], marker='o', linestyle='-')
    
    ax.set_title('Cleaned, Replicate-Averaged Growth in '+ key, color='black', fontsize='17', ha = 'center')
    ax.set_ylabel('OD', color = 'black', fontsize='13')
    ax.set_xlabel('Hours', color = 'black', fontsize='13')
    plt.show()


# In[15]:

dfs_temp_cleaned = dfs_temp

for key, value in dfs_temp_cleaned.items():
    
    # bound biofilm reads
    value.iloc[:,1:97] = value.iloc[:,1:97].clip(upper=0.4)

    # fix tecan reads
    value.iloc[-2:,1:97] = fit.slope*value.iloc[-2:,1:97].values + fit.intercept
    
    fig, ax = plt.subplots(figsize=(6,6))
    
    c_7 = 0
    c_12 = 1

    value["Full Community"] = (value["A10"] + value["A11"] + value["A12"])/3

    for i in range(1,8):
        for j in range(i,8):
            pair = orgs[i-1] + " - " + orgs[j-1]
            well1, well2, well3 = lets[c_7] + str(c_12), lets[c_7] + str(c_12 + 1), lets[c_7] + str(c_12 + 2)
            value[pair] = (value[well1] + value[well2] + value[well3])/3           
            # could do std dev here as well
    #         print("Pair: ",pair," At: ",well1," ",well2," ",well3)
    
            c_12 = c_12 + 3
            if c_12 == 13:
                c_12 = 1
                c_7 = c_7 + 1
            
            ax.plot(value['Hours'][:], value[pair][:], marker='o', linestyle='-')
    
    ax.set_title('Cleaned, Replicate-Averaged Growth in '+ key, color='black', fontsize='17', ha = 'center')
    ax.set_ylabel('OD', color = 'black', fontsize='13')
    ax.set_xlabel('Hours', color = 'black', fontsize='13')
    plt.show()

# In[17]:

# converting to np arrays for exporting to julia
all_df = dfs_gluc_cleaned
all_df.update(dfs_temp_cleaned)
od_cond_order = np.array([])
od_data = np.zeros((7, 29))

for key, value in all_df.items():
    od_data = np.dstack((od_data, value.iloc[:,-29:].to_numpy()))
    od_cond_order = np.hstack([od_cond_order, key]) if od_cond_order.size else np.array([key])

od_pair_order = np.array(export["25C"].columns).astype(str)
od_data = np.delete(od_data, 0, axis=2)

np.save("od_data",od_data)
np.save("od_cond_order",od_cond_order)
np.save("od_pair_order",od_pair_order)

# %% Import and convert VGVT OD into numpys mats


