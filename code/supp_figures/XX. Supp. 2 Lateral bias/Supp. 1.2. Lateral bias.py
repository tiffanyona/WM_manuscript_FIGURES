# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 11:51:57 2022

@author: Tiffany
"""

COLORLEFT = 'teal'
COLORRIGHT = '#FF8D3F'

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy import stats
import os
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import ROOT, FIGURES_OUT, DATA_DIR
sys.path.insert(0, str(ROOT / 'src'))
from functions import add_stat_annotation, repeat_reward_side

path = str(DATA_DIR) + '/'
os.chdir(path)

sns.set_context('paper', rc={'axes.labelsize': 7,
                            'lines.linewidth': 1, 
                            'lines.markersize': 3, 
                            'legend.fontsize': 7,  
                            'xtick.major.size': 1,
                            'xtick.labelsize': 6, 
                            'ytick.major.size': 1, 
                            'ytick.labelsize': 6,
                            'xtick.major.pad': 0,
                            'ytick.major.pad': 0,
                            'xlabel.labelpad': -10})
cm = 1/2.54

# Create a figure with 6 subplots using a GridSpec
fig = plt.figure(figsize=(17*cm, 15*cm))
gs = gridspec.GridSpec(nrows=4, ncols=8, figure=fig)

# Create the subplots
a = fig.add_subplot(gs[0, 0:2])
b = fig.add_subplot(gs[0, 2:4])
c = fig.add_subplot(gs[0, 4:6])
d = fig.add_subplot(gs[0, 6:8])
e = fig.add_subplot(gs[1, 0:2])
f = fig.add_subplot(gs[1, 2:4])
g = fig.add_subplot(gs[1, 4:6]) 
# d = fig.add_subplot(gs[1, 3:5])  # span 2 rows and 2 columns
h = fig.add_subplot(gs[1, 6:8])  # span 2 rows and 1 column
i = fig.add_subplot(gs[2, 0:2])
j = fig.add_subplot(gs[2, 2:4])
k = fig.add_subplot(gs[2, 4:6])
l = fig.add_subplot(gs[2, 6:8])

m = fig.add_subplot(gs[3, 0:2])
n = fig.add_subplot(gs[3, 2:4])
o = fig.add_subplot(gs[3, 4:6])
p = fig.add_subplot(gs[3, 6:8])

fig.text(0.01, 1, 'a', fontsize=10, fontweight='bold', va='top')
fig.text(0.25, 1, 'b', fontsize=10, fontweight='bold', va='top')
fig.text(0.5, 1, 'c', fontsize=10, fontweight='bold', va='top')
fig.text(0.75, 1, 'd', fontsize=10, fontweight='bold', va='top')
fig.text(0.01, 0.75, 'e', fontsize=10, fontweight='bold', va='top')
fig.text(0.25, 0.75, 'f', fontsize=10, fontweight='bold', va='top')
fig.text(0.5, 0.75, 'g', fontsize=10, fontweight='bold', va='top')
fig.text(0.75, 0.75, 'h', fontsize=10, fontweight='bold', va='top')
fig.text(0.01, 0.5, 'i', fontsize=10, fontweight='bold', va='top')
fig.text(0.25, 0.5, 'j', fontsize=10, fontweight='bold', va='top')
fig.text(0.5, 0.5, 'k', fontsize=10, fontweight='bold', va='top')
fig.text(0.75, 0.5, 'j', fontsize=10, fontweight='bold', va='top')
fig.text(0.01, 0.25, 'l', fontsize=10, fontweight='bold', va='top')
fig.text(0.25, 0.25, 'm', fontsize=10, fontweight='bold', va='top')
fig.text(0.5, 0.25, 'n', fontsize=10, fontweight='bold', va='top')
fig.text(0.75, 0.25, 'o', fontsize=10, fontweight='bold', va='top')

def plot_lateral_delay(df, animal, panel):
    df_subject= df.loc[df.subject==animal]
    
    sns.lineplot(x='delay_times',y='hit', hue='reward_side', palette=[COLORLEFT,COLORRIGHT], marker='o', data=df_subject, ax=panel, legend=False)
    sns.lineplot(x='delay_times',y='hit', color='black', data=df_subject, ax=panel, marker='o')
    panel.set_ylim(0.45,1)
    panel.hlines(xmin=0, xmax=10, y=0.5, linestyles=':')
    panel.set_ylabel('Accuracy')
    panel.set_xlabel('Delay duration (s)')
    panel.text(x=4,y=0.95, s=animal, fontsize=8)
    panel.set_xticks([0,1,3,10])
    panel.locator_params(nbins=4)
    
# -----------------###############################################################-----------------------

#-----------------############################## A Panel #################################-----------------------
file_name = 'global_behavior_10s'
df = pd.read_csv(path+file_name+'.csv', index_col=0)

list_subjects=['C10b','N07',  'N24', 'C36',  #Increasing
               'N25', 'E05', 'E07', 'E10', 'E19',                        #decreasing
               'N08','E11','E12','E22']                        # Crossing
list_panels=[a,b,c,d,e,f,g,h,i,j,k,l,m]
for animal,panel in zip(list_subjects,list_panels):
    plot_lateral_delay(df, animal, panel)

# ----------------------------------------------------------------------------------------------------------------

#-----------------############################## A Panel #################################-----------------------
path = str(DATA_DIR) + '/'
file_name = 'GLMM_final_data'
coef_matrix = pd.read_csv(path+file_name+'.csv',  index_col=0)

coef_matrix['SR_SL'] = coef_matrix['SR'] + coef_matrix['SL']
coef_matrix['SR:D_SL:D'] = coef_matrix['SR:D'] + coef_matrix['SL:D']

coef_matrix['SR_SL_sig'] = np.where(coef_matrix['SR_SL']>.3, 1, 0)
coef_matrix['SR_SL_sig'] = np.where(coef_matrix['SR_SL']<-.3, -1, coef_matrix['SR_SL_sig'])

coef_matrix['SR:D_SL:D_sig'] = np.where(coef_matrix['SR:D_SL:D']>.03, 1, 0)
coef_matrix['SR:D_SL:D_sig'] = np.where(coef_matrix['SR:D_SL:D']<-.03, -1, coef_matrix['SR:D_SL:D_sig'])

coef_matrix['SR:D'] =-coef_matrix['SR:D'].values
coef_matrix['SL'] = abs(coef_matrix['SL'].values)

y='SR:D_SL:D'
x='SR_SL'
panel = n
sns.scatterplot(x=x, y=y,data=coef_matrix,color='black', ax=panel, legend=False)
sns.regplot(x=x, y=y,data=coef_matrix, color='black', ax=panel, marker='')

slope, intercept, r_value, p_value, std_err = stats.linregress((np.array(coef_matrix[x]),np.array(coef_matrix[y])))
# plt.locator_params(nbins=3)
panel.text(x=0.3,y=0.2, s= 'R= '+ str(round(r_value,3))+'\nP= '+ str(round(p_value,3)), fontsize=5)

# panel.plot([-0.05, 0.25], [-0.05, 0.25], ls="--", c=".3")
# panel.plot([-0.5, 0], [0, 0], ls="--", c=".3")
# panel.plot([0, 0], [-0.05, 0], ls="--", c=".3")

# panel.xlim(-0.05, 0.25)
# panel.ylim(-0.05, 0.25)

# ----------------------------------------------------------------------------------------------------------------
#-----------------############################## A Panel #################################-----------------------
coef_matrix['SR:D_SL:D'] = abs(coef_matrix['SR:D_SL:D'].values)
coef_matrix['SR_SL'] = abs(coef_matrix['SR_SL'].values)

y='SL:D'
x='SR:D'
panel = p
sns.scatterplot(x=x, y=y,data=coef_matrix, hue='SR:D_SL:D_sig', palette=[COLORLEFT,'grey',COLORRIGHT], ax=panel, legend=False)
panel.plot([-0.05, 0.25], [-0.05, 0.25], ls="--", c=".3")
panel.plot([-0.5, 0], [0, 0], ls="--", c=".3")
panel.plot([0, 0], [-0.05, 0], ls="--", c=".3")

panel.set_xlim(-0.05, 0.26)
panel.set_ylim(-0.05, 0.26)

# ----------------------------------------------------------------------------------------------------------------
#-----------------############################## A Panel #################################-----------------------

y='SL'
x='SR'
panel = o
sns.scatterplot(x=x, y=y,data=coef_matrix, hue='SR_SL_sig', palette=[COLORLEFT,'grey',COLORRIGHT], ax=panel, legend=False)
panel.plot([0.5, 2.5], [0.5, 2.5], ls="--", c=".3")
panel.set_xlim(0.5, 2.5)
panel.set_ylim(0.5, 2.5)

# ----------------------------------------------------------------------------------------------------------------

sns.despine()
plt.subplots_adjust(left=0.07,
                    bottom=0.07,
                    right=0.97,
                    top=0.97,
                    wspace=2.1,
                    hspace=0.7)
path = str(FIGURES_OUT / 'supp_figures' / 'supp_fig_10') + '/'
# plt.tight_layout()
plt.savefig(path+'/Fig. 1 Supp. 2 Lateral bias.svg', bbox_inches='tight',dpi=300)
plt.savefig(path+'/Fig. 1 Supp. 2 Lateral bias.png', bbox_inches='tight',dpi=300)

plt.show()