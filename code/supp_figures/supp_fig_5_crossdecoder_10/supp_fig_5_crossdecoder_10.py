# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 12:06:44 2022

@author: Tiffany
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
import seaborn as sns

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import ROOT, FIGURES_OUT, DATA_DIR
sys.path.insert(0, str(ROOT / 'src'))
import functions as plots

save_path = str(FIGURES_OUT / 'supp_figures' / 'supp_fig_5_crossdecoder_10')
path = str(DATA_DIR / 'supp_figures' / 'supp_fig_5_crossdecoder_10')

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
gs = gridspec.GridSpec(nrows=4, ncols=8, figure=fig)

# Create the subplots
a = fig.add_subplot(gs[0:3, 0:4])
b = fig.add_subplot(gs[0, 4:8])
c = fig.add_subplot(gs[1, 4:8])
d = fig.add_subplot(gs[2, 4:8])
d1 = fig.add_subplot(gs[3, 4:8])

fig.text(0.01, 1, 'a', fontsize=10, fontweight='bold', va='top')
fig.text(0.5, 1, 'b', fontsize=10, fontweight='bold', va='top')
# fig.text(0.01, 0.75, 'e', fontsize=10, fontweight='bold', va='top')
# fig.text(0.26, 0.75, 'f', fontsize=10, fontweight='bold', va='top')
# fig.text(0.5, 0.75, 'g', fontsize=10, fontweight='bold', va='top')
# fig.text(0.01, 0.51, 'h', fontsize=10, fontweight='bold', va='top')


# #########################################################################################
# Panel a: cross-decoder heatmap (10s delay)
# #########################################################################################

file_name = '/panel_a_cross_temporal_choice_decoding_10s_delay_shuffle_subtracted'
df_animal_sti = pd.read_csv(path+file_name+'.csv', index_col = 0)

color= sns.diverging_palette(220, 20, as_cmap=True)

# # Columns to exclude from subtraction
# exclude_columns = ['subject','train', 'fold', 'session']

# # Value to subtract
# value_to_subtract = 0.505

# # Subtract the value from all columns except for the excluded ones
# df_animal_sti.loc[:, df_animal_sti.columns.difference(exclude_columns)] -= value_to_subtract

# df_animal_sti.to_csv(path+'crossdecoder_WMroll1_10s_r0.25_substracted.csv')

panel = a
df_new = df_animal_sti.loc[:, df_animal_sti.columns != 'fold'].groupby(['subject','train']).mean()
df_new.reset_index(inplace=True)
df_new = df_new.groupby('train').mean()
df_new = df_new.reindex(index=df_animal_sti.train.unique())

sns.heatmap(df_new, fmt='', linewidth=0.0, rasterized=True, square=True, vmin=-0.1, vmax=0.3, center=0.0, ax=panel, xticklabels=df_new.columns).invert_yaxis()

def _tick_labels(values):
    """Label only integer-second positions (-2 to 14), blank otherwise."""
    labels = []
    for v in values:
        try:
            t = float(v)
        except (ValueError, TypeError):
            parts = str(v).split('_')
            t = (float(parts[0]) + float(parts[1])) / 2
        t_r = round(t)
        labels.append(str(t_r) if abs(t - t_r) < 0.01 and -2 <= t_r <= 14 else '')
    return labels

panel.set_xticklabels(_tick_labels(df_new.columns), rotation=0)

# Set yticks only at integer-second positions to avoid count mismatch
y_tick_pos, y_tick_lab = [], []
for i, v in enumerate(df_new.index):
    parts = str(v).split('_')
    t = (float(parts[0]) + float(parts[1])) / 2
    t_r = round(t)
    if abs(t - t_r) < 0.01 and -2 <= t_r <= 14:
        y_tick_pos.append(i + 0.5)
        y_tick_lab.append(str(t_r))
panel.set_yticks(y_tick_pos)
panel.set_yticklabels(y_tick_lab)

# Recover the diagonal for all the animals
first=True
df_temp=pd.DataFrame()
train_value_list = df_animal_sti.train.unique()

for train_value in train_value_list:
    real_value = (float(train_value.split('_')[0]) + float(train_value.split('_')[1]))/2
    if real_value == 14.875:
        continue
    df_temp = df_animal_sti.loc[df_animal_sti.train==train_value].groupby('session')[[str(real_value)]].mean().reset_index()
    if first:
        df_diagonal = df_temp
        first=False
    else:
        df_diagonal = pd.merge(df_diagonal, df_temp, on=['session'])

# #########################################################################################
# Panels b/c/d/d1: decoder traces for selected training windows
# #########################################################################################

# This when we want to recover the traces of the crossdecoder
for panel, df_cum_sti, upper_limit in zip([b,c,d,d1],[df_animal_sti.loc[df_animal_sti.train == '0.0_0.25'],
                                      df_animal_sti.loc[df_animal_sti.train == '10.0_10.25'],
                                      df_animal_sti.loc[df_animal_sti.train == '10.75_11.0'],
                                      df_diagonal],[0.3,0.2,0.4,0.4]):
    plots.plot_decoder([panel], df_cum_sti, baseline=0.0, individual_sessions=False, upper_limit=upper_limit, epoch_markers=[(0, 0.45, 'lightgrey', 1), (10.45, 10.65, 'lightgrey', 1)])
    panel.margins(x=0)

# Add ticks to trace panels
for panel_ax, ul in zip([b, c, d, d1], [0.3, 0.2, 0.4, 0.4]):
    panel_ax.set_xticks([0, 5, 10])
    panel_ax.set_xticklabels([0, 5, 10])
    panel_ax.set_yticks([0, ul])
    panel_ax.set_yticklabels([0, ul])

# Show the figure
sns.despine()
plt.subplots_adjust(left=0.07,
                    bottom=0.07,
                    right=0.97,
                    top=0.97,
                    wspace=0.8,
                    hspace=0.6)

# plt.savefig(save_path+'/supp_fig5_crossdecoder_10.svg', bbox_inches='tight',dpi=300)

plt.show()
