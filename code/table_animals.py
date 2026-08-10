# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 22:56:46 2023

@author: Tiffany
"""

import os
import pandas as pd
import numpy as np
# path = 'G:/Mi unidad/WORKING_MEMORY/PAPER/WM_manuscript_FIGURES/Fig. 3. Pharma/'
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import FIGURES_OUT, DATA_DIR, EPHYS_SESSIONS

path = str(DATA_DIR) + '/'

file_name = 'global_behavior_10s'
# file_name = 'GLMM_final_data'
df = pd.read_csv(path+file_name+'.csv', index_col=0)
df = df.loc[df.presented_delays == '0.01,1.0,3.0,10.0']

# -----------------------------------
# behavioral animals
file_name = 'global_behavior'
df = pd.read_csv(path+file_name+'.csv', index_col=0)
animal_list = ['C13', 'C10b', 'C19', 'N08', 'N07', 'N03', 'N04', 'N02', 'N05',
       'C39', 'C37', 'C38', 'C36', 'N19', 'N27', 'N24', 'N28', 'N26',
       'N25', 'E08', 'E10', 'E05', 'E03', 'E07', 'E11', 'E13', 'E14',
       'E12', 'E22', 'E16', 'E20', 'E17', 'E19', 'E04','E15','C12', 'C15', 'C18', 'C20', 'C22',
       'N09', 'N11', 'C28', 'C34','N22', 'N21', 'E06', 'E21', 'E18']

df = df.loc[df['subject'].isin(animal_list)]
df = df.loc[df.presented_delays == '0.01,1.0,3.0']

animal_values = []
for subject in df.subject.unique():
    animal_values.append(np.mean(df.loc[df['subject'] == subject].groupby(['session'])['trials'].max()))

print(np.mean(animal_values), np.std(animal_values))
print(min(animal_values), max(animal_values))

values = []
for animal in df.subject.unique():
    values.append(len(df.loc[df['subject'] == animal].session.unique()))
print(np.mean(values), np.std(values))
print(min(values), max(values))
print(len(df))

# -----------------------------
# pharma animals
file_name = 'drug_experiments_df'
path = str(FIGURES_OUT / 'Fig. 3. Pharma') + '/'
df = pd.read_csv(path+file_name+'.csv', index_col=0)
df = df.loc[df.drug != 'Rest']

animal_values = []
for subject in df.subject.unique():
    animal_values.append(np.mean(df.loc[df['subject'] == subject].groupby(['session'])['total_trial'].max()))

print(np.mean(animal_values), np.std(animal_values))
print(min(animal_values), max(animal_values))

values = []
for animal in df.subject.unique():
    values.append(len(df.loc[df['subject'] == animal].session.unique()))
print(np.mean(values), np.std(values))
print(min(values), max(values))

print(len(df))

# ------------------------------------------------
# ephys sessions and animals
os.chdir(EPHYS_SESSIONS)
df_cum_res = pd.DataFrame()
df_cum_sti = pd.DataFrame()
df_cum_shuffle = pd.DataFrame()

values=[]
neurons=[]
animal=[]
for filename in os.listdir(os.getcwd()):
    if filename[-3:] != 'pdf':
        df = pd.read_csv(filename, sep=',', index_col=0)
        print(filename)
    else:
        continue

    neurons.append(len(df.cluster_id.unique()))
    values.append(len(df.new_trial.unique()))
    animal.append(filename[:3])

from collections import Counter

# Using Counter to count repetitions
count_dict = Counter(animal)

# Displaying the counts
session = []
for key, value in count_dict.items():
    session.append(value)
    print(f"{key}: {value} times")

