# -*- coding: utf-8 -*-
COLORLEFT = 'teal'
COLORRIGHT = '#FF8D3F'

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import os
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*FigureCanvasAgg.*')

from neo.core import SpikeTrain
from quantities import ms
from elephant.statistics import time_histogram, instantaneous_rate
from elephant.kernels import GaussianKernel
from cycler import cycler

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import ROOT, FIGURES_OUT, DATA_DIR
sys.path.insert(0, str(ROOT / 'src'))
from functions import convolveandplot, plot_decoder_shuffle

save_path = str(FIGURES_OUT / 'fig_3_ephys_wm') + '/'
path = str(DATA_DIR / 'fig_3_ephys_wm') + '/'
os.chdir(path)

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

fig = plt.figure(figsize=(15*cm, 30.*cm))
gs = gridspec.GridSpec(nrows=7, ncols=9, figure=fig,
                       height_ratios=[1.2, 1.0, 1.2, 1.0, 0.9, 0.9, 0.9],
                       left=0.09, right=0.97, top=0.93, bottom=0.05,
                       wspace=0.8, hspace=0.55)

# Right column
a  = fig.add_subplot(gs[0:4, 4:9])   # heatmap -> panel c
b  = fig.add_subplot(gs[4,   4:9])   # decoder d
c  = fig.add_subplot(gs[5,   4:9])   # decoder e
d  = fig.add_subplot(gs[6,   4:9])   # decoder f

# Left column panel a: nested raster(3):PSTH(2)
gs_a = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0:2, 0:4],
                                         hspace=0.05, height_ratios=[3, 2])
h2 = fig.add_subplot(gs_a[0])
h1 = fig.add_subplot(gs_a[1])

# Left column panel b: nested raster(3):PSTH(2)
gs_b = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[2:4, 0:4],
                                         hspace=0.05, height_ratios=[3, 2])
e  = fig.add_subplot(gs_b[0])
f  = fig.add_subplot(gs_b[1])

# Panel g: same height as one decoder row
g1 = fig.add_subplot(gs[4:6, 0:4])

# Panel labels
fig.text(0.01, 0.99, 'a', fontsize=10, fontweight='bold', va='top')
fig.text(0.01, 0.67, 'b', fontsize=10, fontweight='bold', va='top')
fig.text(0.50, 0.99, 'c', fontsize=10, fontweight='bold', va='top')
fig.text(0.50, 0.55, 'd', fontsize=10, fontweight='bold', va='top')
fig.text(0.50, 0.38, 'e', fontsize=10, fontweight='bold', va='top')
fig.text(0.50, 0.22, 'f', fontsize=10, fontweight='bold', va='top')
fig.text(0.01, 0.55, 'g', fontsize=10, fontweight='bold', va='top')

# ---------------------------------------------------------------------------
# Panel c — heatmap
# ---------------------------------------------------------------------------
file_name = 'crossdecoder_WM_roll1_3s_r0.25_choice_Stimulus_ON_substraction_V3'
df_animal_shuffle = pd.read_csv(path + file_name + '_shuffle.csv', index_col=0, low_memory=False)
df_animal_sti     = pd.read_csv(path + file_name + '_sti.csv',     index_col=0, low_memory=False)

panel = a
df_shuffle_mean = pd.DataFrame()
for epoch in df_animal_shuffle.train.unique():
    df_shuffle_mean[epoch] = (df_animal_shuffle.loc[df_animal_shuffle.train == epoch]
                              .groupby(['times']).mean(numeric_only=True)
                              .drop(columns='fold').mean(axis=1).values)
df_shuffle_mean.index = df_animal_shuffle.groupby('times').mean(numeric_only=True).index

df_new = (df_animal_sti.loc[:, df_animal_sti.columns != 'fold']
          .groupby(['subject', 'train']).mean(numeric_only=True)
          .reset_index()
          .groupby('train').mean(numeric_only=True)
          .reindex(index=df_animal_sti.train.unique()))
df_shuffle_mean.index = df_new.columns

hm = sns.heatmap(df_new - df_shuffle_mean.T, fmt='', linewidth=0.0, rasterized=True,
                 square=True, vmin=-0.1, vmax=0.3, center=0.0,
                 ax=panel, xticklabels=df_new.columns, cbar=False)
hm.invert_yaxis()

_xlabels = ["-2",'','',"","","","","","0",'',"","","","","","","2","","",
            "","","","","","4","","","","","","","","6","","","","","","","","8"]
panel.set_xticks(range(len(df_new.columns)))
panel.set_xticklabels(_xlabels[:len(df_new.columns)], fontsize=5)
_ylabels = ["-2",'','',"","0",'',"","","2","","","","4","","","","6","","","","8"]
panel.set_yticks(range(len(df_new.index)))
panel.set_yticklabels((_ylabels + [''] * len(df_new.index))[:len(df_new.index)], fontsize=5)
panel.set_xlabel("Testing time from stimulus onset (s)")
panel.set_ylabel("Training time from stimulus onset (s)")

n_rows = len(df_new.index)
train_times = np.array([np.mean([float(v.split('_')[0]), float(v.split('_')[1])])
                        for v in df_new.index])
row_t0  = int(np.argmin(np.abs(train_times - 0.0)))
row_t35 = int(np.argmin(np.abs(train_times - 3.5)))
for row_y in [row_t0, row_t35]:
    panel.axhline(y=row_y, color='white', linewidth=0.8, zorder=5)
panel.text(-0.3, row_t0 / 2,           'Stimulus', color='white', fontsize=5,
           va='center', ha='right', clip_on=False)
panel.text(-0.3, (row_t0 + row_t35)/2, 'Delay',    color='white', fontsize=5,
           va='center', ha='right', clip_on=False)
panel.text(-0.3, (row_t35 + n_rows)/2, 'Response', color='white', fontsize=5,
           va='center', ha='right', clip_on=False)

test_times = np.array([float(col) for col in df_new.columns])
col_stim = int(np.argmin(np.abs(test_times - 0.175)))
col_go   = int(np.argmin(np.abs(test_times - 3.45)))
panel.text(col_stim, n_rows + 0.4, 'Stim', color='black', fontsize=5,
           ha='center', va='bottom', clip_on=False)
panel.text(col_go,   n_rows + 0.4, 'Go',   color='black', fontsize=5,
           ha='center', va='bottom', clip_on=False)

first = True
df_diagonal = pd.DataFrame()
for train_value in df_animal_sti.train.unique():
    real_value = np.around(
        (float(train_value.split('_')[0]) + float(train_value.split('_')[1])) / 2, 3)
    if real_value in (7.875, 7.9):
        continue
    df_temp_d = (df_animal_sti.loc[df_animal_sti.train == train_value]
                 .groupby('session')[[str(real_value)]].mean().reset_index())
    if first:
        df_diagonal = df_temp_d; first = False
    else:
        df_diagonal = pd.merge(df_diagonal, df_temp_d, on=['session'])

# ---------------------------------------------------------------------------
# Panels d / e / f
# ---------------------------------------------------------------------------
for panel, df_cum_sti, df_shuffle, upper_limit in zip(
        [b, c, d],
        [df_animal_sti.loc[df_animal_sti.train == '0.0_0.25'],
         df_animal_sti.loc[df_animal_sti.train == '3.0_3.25'],
         df_animal_sti.loc[df_animal_sti.train == '3.75_4.0']],
        [df_animal_shuffle.loc[df_animal_shuffle.train == '0.25_0.5'],
         df_animal_shuffle.loc[df_animal_shuffle.train == '3.0_3.25'],
         df_animal_shuffle.loc[df_animal_shuffle.train == '3.75_4.0']],
        [0.25, 0.25, 0.4]):
    plot_decoder_shuffle(panel, df_cum_sti, df_shuffle, baseline=0.0,
                         individual_sessions=False, upper_limit=upper_limit)
    panel.margins(x=0)
    panel.locator_params(nbins=5)
    sns.despine(offset=2, ax=panel)
    panel.set_xlim(-2, 8)

b.set_title('Stimulus code', fontweight='bold', fontsize=7)
c.set_title('Delay code',    fontweight='bold', fontsize=7)
d.set_title('Response code', fontweight='bold', fontsize=7)

# ---------------------------------------------------------------------------
# Panel a — population raster (h2) + PSTH (h1)
# Set viridis cycler immediately before plotting h1 lines
# ---------------------------------------------------------------------------
big_data = pd.read_csv(path + 'single_trial_example_237.csv', index_col=0)
n_neurons = big_data['neuron'].nunique()

dft = pd.read_csv(path + 'single_trial_example_df_237.csv', index_col=0)
dft['a_Stimulus_ON'] = dft['fixed_times'] - dft['Stimulus_ON']

delay  = 3
align  = 'Stimulus_ON'
cue_on = 0; cue_off = 0.38; start = -2
stop   = max(dft['a_' + align])

big_data['time_centered'] = np.round(
    (big_data['times'] - big_data['Stimulus_ON']) / 1000, 2)
big_data['firing_'] = big_data['firing'] * 1000

df_results = pd.DataFrame(dtype=float)
df_results['firing'] = (big_data.loc[big_data.time_centered <= stop]
                        .groupby(['time_centered', 'neuron'])['firing_'].mean())
df_results.reset_index(inplace=True)

# Set viridis right before the h1 plot loop
viridis_colors = plt.cm.viridis(np.linspace(0, 1, n_neurons))
h1.set_prop_cycle(cycler(color=viridis_colors))

panel = h1
for N in df_results.neuron.unique():
    panel.plot(df_results.loc[df_results.neuron == N].time_centered,
               df_results.loc[df_results.neuron == N].firing, alpha=0.5)
panel.set_xlim(start, stop)
panel.set_xlabel('Time from stimulus onset (s)')
y = np.arange(0, 80, 0.1)
panel.fill_betweenx(y, cue_on, cue_off,                 color='lightgrey', alpha=1, linewidth=0)
panel.fill_betweenx(y, cue_off+delay, cue_off+delay+.2, color='lightgrey', alpha=1, linewidth=0)
panel.set_ylabel('Firing rate (spks/s)')

panel = h2
FR_mean = []; cluster_id_list = []
for N in dft.cluster_id.unique():
    spk = dft.loc[(dft.cluster_id == N) & (dft['a_'+align] > 0.2) &
                  (dft['a_'+align] < delay)]['a_'+align].values
    FR_mean.append(len(spk) / delay); cluster_id_list.append(N)
df_spikes = pd.DataFrame({'cluster_id': cluster_id_list, 'FR': FR_mean})
df_spikes = df_spikes.sort_values('FR')
df_spikes['new_order'] = np.arange(len(df_spikes))
dft = pd.merge(df_spikes, dft, on=['cluster_id'])

j = 0
for N in dft.new_order.unique():
    spk = dft.loc[dft.new_order == N]['a_'+align].values
    j += 1
    panel.plot(spk, np.repeat(j, len(spk)), '|',
               markersize=0.5, color='black', zorder=1)
panel.set_ylabel('Single units')
panel.set_ylim(0, j); panel.set_xlim(start, stop)
panel.axes.get_xaxis().set_visible(False)
y = np.arange(0, j+1, 0.1)
panel.fill_betweenx(y, cue_on, cue_off,                 color='grey',      alpha=1, linewidth=0)
panel.fill_betweenx(y, cue_off+delay, cue_off+delay+.2, color='lightgrey', alpha=1, linewidth=0)
h2.set_title('Session E20 2022-02-14', fontsize=7)

# ---------------------------------------------------------------------------
# Panel b — single neuron (uses COLORRIGHT/COLORLEFT via convolveandplot)
# ---------------------------------------------------------------------------
df_sn   = pd.read_csv(path + 'single_neuron_10s.csv', index_col=0)
df_sn   = df_sn.loc[df_sn.WM_roll > 0.6]
temp_df = df_sn.loc[(df_sn.WM_roll > 0.6) & (df_sn.hit == 1)]
j = convolveandplot(temp_df, e, f, variable='reward_side',
                    cluster_id=153, delay=10, j=1)
e.set_title('Session E20 2022-02-14 - Cluster 153', fontsize=7)

# ---------------------------------------------------------------------------
# Panel g — code overlap
# ---------------------------------------------------------------------------
df_temp   = pd.read_csv(path + 'parsed_weights for the modelling_late3.csv',
                        index_col=0).reset_index(drop=True)
panel     = g1
orderlist = ['Stim x Late Delay', 'Late Delay x Response',
             'Early x Late Delay', 'Late Delay x Late Delay*']

sns.stripplot(x='condition', y='vector', data=df_temp, order=orderlist,
              jitter=0.2, size=3, color='black', edgecolor='black',
              ax=panel, linewidth=0.1, zorder=3)
sns.boxplot(x='condition', y='vector', data=df_temp, order=orderlist,
            color='lightgrey', width=0.45, fliersize=3, linewidth=0.8,
            ax=panel, zorder=2)

panel.hlines(y=0., xmin=-0.5, xmax=len(orderlist)-0.5, linestyle=':')
panel.set_xlabel('')
panel.set_ylim(-1.2, 1.4)
panel.set_xticks(range(len(orderlist)))
panel.set_xticklabels(['Stimulus\nLate delay', 'Response\nLate delay',
                       'Early delay\nLate delay', 'Late delay\nLate delay*'], fontsize=6)
panel.set_ylabel('Code overlap')

for i, cond in enumerate(orderlist):
    vals = df_temp.loc[df_temp.condition == cond, 'vector'].values
    _, p = stats.ttest_1samp(vals, 0)
    star  = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
    y_top = df_temp.loc[df_temp.condition == cond, 'vector'].max()
    panel.text(i, y_top + 0.05, star, ha='center', va='bottom', fontsize=7)

vals_e = df_temp.loc[df_temp.condition == 'Early x Late Delay',       'vector'].values
vals_l = df_temp.loc[df_temp.condition == 'Late Delay x Late Delay*', 'vector'].values
_, p_rel  = stats.ttest_rel(vals_e, vals_l)
star_rel  = 'ns' if p_rel >= 0.05 else ('***' if p_rel < 0.001 else ('**' if p_rel < 0.01 else '*'))
yb = max(vals_e.max(), vals_l.max()) + 0.25
panel.plot([2, 2, 3, 3], [yb, yb+0.04, yb+0.04, yb], color='black', linewidth=0.8)
panel.text(2.5, yb + 0.05, star_rel, ha='center', va='bottom', fontsize=7)

# ---------------------------------------------------------------------------
# Finalise + colorbar
# ---------------------------------------------------------------------------
sns.despine()
plt.locator_params(nbins=5)

# Draw to get accurate heatmap position, then place colorbar just above it
fig.canvas.draw()
pos = a.get_position()
cax = fig.add_axes([pos.x0, pos.y1 + 0.01, pos.width * 0.85, 0.012])
norm = mpl.colors.TwoSlopeNorm(vmin=-0.1, vcenter=0.0, vmax=0.3)
cmap = hm.collections[0].cmap
cb   = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='horizontal')
cb.ax.xaxis.set_ticks_position('top')
cb.ax.xaxis.set_label_position('top')
cb.set_label('Excess decoding', fontsize=6, labelpad=2)
cb.ax.tick_params(labelsize=5)

# plt.savefig(save_path + '/Fig_3_ephys_wm_revised.svg', bbox_inches='tight', dpi=300)
with warnings.catch_warnings():
    warnings.simplefilter('ignore', UserWarning)
    plt.show()
