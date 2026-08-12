# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 12:06:44 2022

@author: Tiffany
"""
COLORLEFT = 'teal'
COLORRIGHT = '#FF8D3F'

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import pandas as pd
import seaborn as sns
#Import all needed libraries
from functions import add_stat_annotation
# from datahandler import Utils  # legacy; Utils not used

# from plots import functions_Fig_4 as plots
# import functions_Fig_4 as plots

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import ROOT, FIGURES_OUT, DATA_DIR
sys.path.insert(0, str(ROOT / 'src'))
import functions as plots

save_path = str(FIGURES_OUT / 'supp_figures' / 'supp_fig_10_errors_different_delays') + '/'
cm = 1/2.54
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

# Create a figure with 6 subplots using a GridSpec
fig = plt.figure(figsize=(17*cm, 15*cm))
gs = gridspec.GridSpec(nrows=3, ncols=8, figure=fig)

# Create the subplots
a1 = fig.add_subplot(gs[0, 1:4])
a2 = fig.add_subplot(gs[1, 1:4])
a3 = fig.add_subplot(gs[2, 1:4])

b1 = fig.add_subplot(gs[0, 4:7])
b2 = fig.add_subplot(gs[1, 4:7])
b3 = fig.add_subplot(gs[2, 4:7])

fig.text(0.01, 0.99, 'a', fontsize=10, fontweight='bold', va='top')
fig.text(0.5, 0.99, 'b', fontsize=10, fontweight='bold', va='top')
fig.text(0.01, 0.64, 'c', fontsize=10, fontweight='bold', va='top')
fig.text(0.5, 0.64, 'd', fontsize=10, fontweight='bold', va='top')
fig.text(0.01, 0.3, 'e', fontsize=10, fontweight='bold', va='top')
fig.text(0.5, 0.3, 'f', fontsize=10, fontweight='bold', va='top')

# ------############################## J Panel -- WM error trials individual example for stimulus #################################-----------------------

path = str(DATA_DIR / 'supp_figures' / 'supp_fig_10_errors_different_delays') + '/'

colors=['crimson','darkgreen']
variables = ['WM_roll','WM_roll']
hits = [0,1]
variables_combined=[variables[0]+'_'+str(hits[0]),variables[1]+'_'+str(hits[1])]

# file_name = 'single_delay_WM_roll0.6_0.5_folded5_sti_all'
file_name = 'single_delay_WM_roll0.6_stimulus_0.25_V1'

df_cum_sti = pd.read_csv(path+file_name+'.csv', index_col=0)

scores = df_cum_sti.groupby('session').score.mean().reset_index()
list_exclude = scores.loc[scores.score<0.55].session.unique()
df_cum_sti = df_cum_sti[~df_cum_sti['session'].isin(list_exclude)]

delay=1
panel=a1
plots.plotsingledelay(df_cum_sti, panel, colors, variables_combined, delay, baseline=0.0, invert_list=[True, False])
panel.set_xlim(-2,4)
panel.set_ylim(-0.1,0.3)

delay=3
panel=b1
plots.plotsingledelay(df_cum_sti, panel, colors, variables_combined, delay, baseline=0.0, invert_list=[True, False])
panel.set_xlim(-2,6)
panel.set_ylim(-0.1,0.3)

# ------#########################################################################################-----------------------
# ------############################## K Panel -- WM error trials  for response #################################-----------------------


save_path = str(DATA_DIR / 'supp_figures' / 'supp_fig_10_errors_different_delays') + '/'
os.chdir(save_path)
# file_name = 'single_delay_WM_roll0.65_0.5_V4'
file_name = 'single_delay_WM_roll0.6_delay_0.25_all_V8'
df_cum_sti = pd.read_csv(file_name+'.csv', index_col=0)

scores = df_cum_sti.groupby('session').score.mean().reset_index()
list_exclude = scores.loc[scores.score<0.55].session.unique()
df_cum_sti = df_cum_sti[~df_cum_sti['session'].isin(list_exclude)]

delay=1
panel=a2
plots.plotsingledelay(df_cum_sti, panel, colors, variables_combined, delay, baseline=0.0)
panel.set_xlim(-2,4)
panel.set_ylim(-0.1,0.3)

delay=3
panel=b2
plots.plotsingledelay(df_cum_sti, panel, colors, variables_combined, delay, baseline=0.0)
panel.set_xlim(-2,6)
panel.set_ylim(-0.1,0.3)

# ------############################## K Panel -- WM error trials for delay #################################-----------------------

save_path = str(DATA_DIR / 'supp_figures' / 'supp_fig_10_errors_different_delays') + '/'
os.chdir(save_path)
file_name = 'single_delay_WM_roll0.6_lick_0.25_V1'
df_cum_sti = pd.read_csv(file_name+'.csv', index_col=0)

# scores = df_cum_sti.groupby('session').score.mean().reset_index()
# list_exclude = scores.loc[scores.score<0.55].session.unique()
# df_cum_sti = df_cum_sti[~df_cum_sti['session'].isin(list_exclude)]

delay=1
panel=a3
plots.plotsingledelay(df_cum_sti, panel, colors, variables_combined, delay, baseline=0.0)
panel.set_xlim(-2,4)
panel.set_ylim(-0.1,0.4)
panel.set_xlabel('Time from stimulus onset (s)')

delay=3
panel=b3
plots.plotsingledelay(df_cum_sti, panel, colors, variables_combined, delay, baseline=0.0)
panel.set_xlim(-2,6)
panel.set_ylim(-0.1,0.4)
panel.set_xlabel('Time from stimulus onset (s)')
panel.locator_params(nbins=3)

# ------#########################################################################################-----------------------


# ------#########################################################################################-----------------------

# Show the figure
save_path = str(FIGURES_OUT / 'supp_figures' / 'supp_fig_10_errors_different_delays') + '/'
plt.subplots_adjust(left=0.07,
                    bottom=0.07,
                    right=0.97,
                    top=0.97,
                    wspace=1.5,
                    hspace=0.45)
sns.despine()
# plt.savefig(save_path+'/supp_fig_10_errors.svg', bbox_inches='tight',dpi=300)
# plt.savefig(save_path+'/supp_fig_10_errors.png', bbox_inches='tight',dpi=300)

plt.show()
