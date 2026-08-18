# -*- coding: utf-8 -*-
COLORLEFT = 'teal'
COLORRIGHT = '#FF8D3F'

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from matplotlib.ticker import FixedLocator, FixedFormatter
import os
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*FigureCanvasAgg.*')
from cycler import cycler

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import ROOT, FIGURES_OUT, DATA_DIR
sys.path.insert(0, str(ROOT / 'src'))
from functions import convolveandplot
import functions as plots

save_path = str(FIGURES_OUT / 'fig_3_ephys_wm') + '/'
Path(save_path).mkdir(parents=True, exist_ok=True)
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
gs = gridspec.GridSpec(nrows=7, ncols=10, figure=fig,
                       height_ratios=[1.2, 1.0, 1.2, 1.0, 1.2, 0.9, 0.9],
                       left=0.09, right=0.97, top=0.91, bottom=0.05,
                       wspace=0.9, hspace=0.95)

# Right column
a  = fig.add_subplot(gs[0:4, 5:10])   # heatmap -> panel c
b  = fig.add_subplot(gs[4,   5:10])   # decoder d
c  = fig.add_subplot(gs[5,   5:10])   # decoder e
d  = fig.add_subplot(gs[6,   5:10])   # decoder f

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

# Panel g
g1 = fig.add_subplot(gs[4, 0:4])

# Panel labels
fig.text(0.01, 0.93, 'a', fontsize=10, fontweight='bold', va='top')
fig.text(0.01, 0.661,'b', fontsize=10, fontweight='bold', va='top')
fig.text(0.50, 0.93, 'c', fontsize=10, fontweight='bold', va='top')
fig.text(0.50, 0.392,'d', fontsize=10, fontweight='bold', va='top')
fig.text(0.50, 0.270,'e', fontsize=10, fontweight='bold', va='top')
fig.text(0.50, 0.160,'f', fontsize=10, fontweight='bold', va='top')
fig.text(0.01, 0.392,'g', fontsize=10, fontweight='bold', va='top')

# #########################################################################################
# Panel a: population raster (h2) and PSTH (h1)
# #########################################################################################

big_data = pd.read_csv(path + 'panel_a_population_firing_rates_trial237_session_E20.csv', index_col=0)
n_neurons = big_data['neuron'].nunique()

dft = pd.read_csv(path + 'panel_a_population_spike_times_trial237_session_E20.csv', index_col=0)
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

viridis_colors = plt.cm.viridis(np.linspace(0.05, 0.95, n_neurons))
h1.set_prop_cycle(cycler(color=viridis_colors))

panel = h1
for N in df_results.neuron.unique():
    panel.plot(df_results.loc[df_results.neuron == N].time_centered,
               df_results.loc[df_results.neuron == N].firing, alpha=0.5)
panel.set_xlim(start, stop)
panel.set_xlabel('Time from stimulus onset (s)')
y = np.arange(0, 80, 0.1)
panel.fill_betweenx(y, cue_on, cue_off,                 color='lightgrey', alpha=1, linewidth=0, zorder=0)
panel.fill_betweenx(y, cue_off+delay, cue_off+delay+.2, color='lightgrey', alpha=1, linewidth=0, zorder=0)
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
panel.xaxis.set_visible(False)
panel.fill_betweenx(y, cue_on, cue_off,                 color='lightgrey',      alpha=1, linewidth=0, zorder=0)
panel.fill_betweenx(y, cue_off+delay, cue_off+delay+.2, color='lightgrey', alpha=1, linewidth=0, zorder=0)
h2.set_title('Session E20 2022-02-14', fontsize=7)

# #########################################################################################
# Panel b: single neuron raster and PSTH
# #########################################################################################

df_sn   = pd.read_csv(path + 'panel_b_neuron153_spike_times_all_10s_delay_trials.csv', index_col=0)
df_sn   = df_sn.loc[df_sn.WM_roll > 0.6]
temp_df = df_sn.loc[(df_sn.WM_roll > 0.6) & (df_sn.hit == 1)]
j = convolveandplot(temp_df, e, f, variable='reward_side',
                    cluster_id=153, delay=10, j=1)
e.set_title('Session E20 2022-02-14 - Cluster 153', fontsize=7)

for ax in [e, f]:
    leg = ax.get_legend()
    if leg is not None:
        for text in leg.get_texts():
            t = text.get_text()
            if 'right' in t.lower():
                text.set_text('Right')
            elif 'left' in t.lower():
                text.set_text('Left')

# #########################################################################################
# Panel c: cross-temporal decoding heatmap
# #########################################################################################

df_animal_shuffle = pd.read_csv(path + 'panels_c-f_cross_temporal_choice_decoding_3s_delay_shuffle_control.csv', index_col=0, low_memory=False)
df_animal_sti     = pd.read_csv(path + 'panels_c-f_cross_temporal_choice_decoding_3s_delay.csv',                 index_col=0, low_memory=False)

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

# X ticks
_test_times = np.array([float(col) for col in df_new.columns])
_x_int_ticks = [-2, 0, 2, 4, 6, 8]
_x_tick_pos  = [np.argmin(np.abs(_test_times - t)) for t in _x_int_ticks]
panel.set_xticks(_x_tick_pos)
panel.set_xticklabels([str(t) for t in _x_int_ticks], fontsize=5)

# Y ticks
_train_times_all = np.array([np.mean([float(v.split('_')[0]), float(v.split('_')[1])])
                              for v in df_new.index])
_y_int_ticks = [-2, 0, 2, 4, 6, 8]
_y_tick_pos  = [np.argmin(np.abs(_train_times_all - t)) for t in _y_int_ticks]
panel.set_yticks(_y_tick_pos)
panel.set_yticklabels([str(t) for t in _y_int_ticks], fontsize=5)
panel.set_xlabel("Testing time from stimulus onset (s)")
panel.set_ylabel("Training time from stimulus onset (s)")

# ← Force y-axis to left LAST, after all seaborn heatmap calls
panel.yaxis.set_label_position('left')
panel.yaxis.tick_left()
panel.tick_params(axis='y', left=True, right=False,
                  labelleft=True, labelright=False)

n_rows = len(df_new.index)
train_times = np.array([np.mean([float(v.split('_')[0]), float(v.split('_')[1])])
                        for v in df_new.index])
row_t0  = int(np.argmin(np.abs(train_times - 0.0)))
row_t35 = int(np.argmin(np.abs(train_times - 3.5)))

# White horizontal separator lines between epochs
for row_y in [row_t0, row_t35]:
    panel.axhline(y=row_y, color='white', linewidth=0.8, zorder=5)

# Epoch labels — use axes-fraction x so they're independent of data coords
panel.text(-0.12, row_t0/2,           'Stimulus', color='white', fontsize=5,
           va='center', ha='right', clip_on=False,
           transform=panel.get_yaxis_transform())
panel.text(-0.12, (row_t0+row_t35)/2, 'Delay',    color='white', fontsize=5,
           va='center', ha='right', clip_on=False,
           transform=panel.get_yaxis_transform())
panel.text(-0.12, (row_t35+n_rows)/2, 'Response', color='white', fontsize=5,
           va='center', ha='right', clip_on=False,
           transform=panel.get_yaxis_transform())

# Stim / Go labels inside top of heatmap
test_times = np.array([float(col) for col in df_new.columns])
col_stim = int(np.argmin(np.abs(test_times - 0.175)))
col_go   = int(np.argmin(np.abs(test_times - 3.45)))
panel.text(col_stim, n_rows - 1, 'Stim', color='white', fontsize=5,
           ha='center', va='top', clip_on=True)
panel.text(col_go,   n_rows - 1, 'Go',   color='white', fontsize=5,
           ha='center', va='top', clip_on=True)

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

# #########################################################################################
# Panels d / e / f: decoder time courses (stimulus, delay, response codes)
# #########################################################################################

for panel, df_cum_sti, df_shuffle, upper_limit in zip(
        [b, c, d],
        [df_animal_sti.loc[df_animal_sti.train == '0.0_0.25'],
         df_animal_sti.loc[df_animal_sti.train == '3.0_3.25'],
         df_animal_sti.loc[df_animal_sti.train == '3.75_4.0']],
        [df_animal_shuffle.loc[df_animal_shuffle.train == '0.25_0.5'],
         df_animal_shuffle.loc[df_animal_shuffle.train == '3.0_3.25'],
         df_animal_shuffle.loc[df_animal_shuffle.train == '3.75_4.0']],
        [0.25, 0.25, 0.4]):
    plots.plot_decoder([panel], df_cum_sti, baseline=0.0,
                       individual_sessions=False, upper_limit=upper_limit,
                       shuffle_df=df_shuffle)
    panel.set_xlabel('Time from Cue onset (s)')
    panel.set_ylabel('Decoding\n accuracy')
    panel.margins(x=0)
    panel.locator_params(nbins=5)
    sns.despine(offset=2, ax=panel)
    panel.set_xlim(-2, 8)

b.set_title('Stimulus code', fontweight='bold', fontsize=7)
c.set_title('Delay code',    fontweight='bold', fontsize=7)
d.set_title('Response code', fontweight='bold', fontsize=7)
b.set_ylabel('Excess decoding')
c.set_ylabel('Excess decoding')
d.set_ylabel('Excess decoding')
d.set_xlabel('Time from stimulus onset (s)')

# #########################################################################################
# Panel g: code overlap (weight vector comparisons)
# #########################################################################################

df_temp   = pd.read_csv(path + 'panel_g_decoder_weight_vectors_epoch_overlap.csv',
                        index_col=0).reset_index(drop=True)
panel     = g1
orderlist = ['Stim x Late Delay', 'Late Delay x Response',
             'Early x Late Delay', 'Late Delay x Late Delay*']

sns.stripplot(x='condition', y='vector', data=df_temp, order=orderlist,
              jitter=0.2, size=2, color='black', edgecolor='black',
              ax=panel, linewidth=0.1, zorder=3)
sns.boxplot(x='condition', y='vector', data=df_temp, order=orderlist,
            color='lightgrey', width=0.5, fliersize=0, medianprops={'color': 'black', 'linewidth': 0.8}, linewidth=0,
            ax=panel, zorder=2)

panel.hlines(y=0., xmin=-0.5, xmax=len(orderlist)-0.5, linestyle=':', color='black')
panel.set_xlabel('')
panel.set_ylim(-1.2, 1.4)
panel.set_xticks(range(len(orderlist)))
panel.set_xticklabels(['Stimulus\nLate delay', 'Response\nLate delay',
                       'Early delay\nLate delay', 'Late delay\nLate delay*'], fontsize=6)
panel.set_ylabel('Code overlap')

for i, cond in enumerate(orderlist):
    vals = df_temp.loc[df_temp.condition == cond, 'vector'].values
    _, p = stats.ttest_1samp(vals, 0)
    star  = plots.p_to_stars(p)
    y_top = df_temp.loc[df_temp.condition == cond, 'vector'].max()
    panel.text(i, y_top + 0.05, star, ha='center', va='bottom', fontsize=7)

vals_e = df_temp.loc[df_temp.condition == 'Early x Late Delay',       'vector'].values
vals_l = df_temp.loc[df_temp.condition == 'Late Delay x Late Delay*', 'vector'].values
_, p_rel  = stats.ttest_rel(vals_e, vals_l)
star_rel  = 'ns' if p_rel >= 0.05 else ('***' if p_rel < 0.001 else ('**' if p_rel < 0.01 else '*'))
yb = max(vals_e.max(), vals_l.max()) + 0.25
panel.plot([2, 2, 3, 3], [yb, yb+0.04, yb+0.04, yb], color='black', linewidth=0.8)
panel.text(2.5, yb + 0.05, star_rel, ha='center', va='bottom', fontsize=7)

sns.despine()
plt.locator_params(nbins=5)

for _ax in [h2, e, b, c]:
    _ax.spines['bottom'].set_visible(False)
    _ax.spines['top'].set_visible(False)
    _ax.spines['right'].set_visible(False)
    _ax.tick_params(bottom=False, labelbottom=False)
h2.set_xlabel('')
e.set_xlabel('')
b.set_xlabel('')
c.set_xlabel('')

# Draw so square=True has fully resized axes before we measure position
fig.canvas.draw()

# Use get_window_extent to get true post-square=True pixel bounds
renderer  = fig.canvas.get_renderer()
bbox_disp = a.get_window_extent(renderer=renderer)
bbox_fig  = bbox_disp.transformed(fig.transFigure.inverted())

cax = fig.add_axes([bbox_fig.x0 +0.01,
                    bbox_fig.y1 + 0.1,
                    bbox_fig.width * 0.85,
                    0.010])

norm      = mpl.colors.Normalize(vmin=-0.1, vmax=0.3)
cmap_orig = hm.collections[0].cmap
cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap_orig, norm=norm, orientation='horizontal')
cb.ax.invert_xaxis()

# ← FixedLocator prevents matplotlib auto-ticking from overriding our values
cb.ax.xaxis.set_major_locator(FixedLocator([-0.1, 0.0, 0.1, 0.2, 0.3]))
cb.ax.xaxis.set_major_formatter(FixedFormatter(['-0.1', '0', '0.1', '0.2', '0.3']))
cb.ax.xaxis.set_ticks_position('top')
cb.ax.xaxis.set_label_position('top')
cb.ax.set_xlabel('Excess decoding', fontsize=6, labelpad=2)
cb.ax.tick_params(labelsize=5)

# plt.savefig(save_path + '/Fig_3_ephys_wm_revised.svg', bbox_inches='tight', dpi=300)
with warnings.catch_warnings():
    warnings.simplefilter('ignore', UserWarning)
    plt.show()
