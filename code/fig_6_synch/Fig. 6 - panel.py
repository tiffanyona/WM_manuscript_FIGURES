# -*- coding: utf-8 -*-
COLORLEFT = 'teal'
COLORRIGHT = '#FF8D3F'

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib as mpl
import os
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from cycler import cycler
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import ROOT, FIGURES_OUT, DATA_DIR
sys.path.insert(0, str(ROOT / 'src'))
from functions import add_stat_annotation, synch_trial

save_path = str(FIGURES_OUT / 'fig_6_synch') + '/'
path = str(DATA_DIR / 'fig_6_synch') + '/'
cm = 1/2.54
sns.set_context('paper', rc={
    'axes.labelsize': 7, 'lines.linewidth': 1, 'lines.markersize': 3,
    'legend.fontsize': 6, 'xtick.major.size': 1, 'xtick.labelsize': 6,
    'ytick.major.size': 1, 'ytick.labelsize': 6,
    'xtick.major.pad': 0, 'ytick.major.pad': 0, 'xlabel.labelpad': -10})

# ── Figure & GridSpec ─────────────────────────────────────────────────────
fig = plt.figure(figsize=(21*cm, 15*cm))
gs = gridspec.GridSpec(nrows=2, ncols=9, figure=fig,
                       left=0.07, bottom=0.07, right=0.97, top=0.97,
                       wspace=1.0, hspace=0.5)

gs_a = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[0, 0:3],
                                        hspace=0, height_ratios=[2, 2, 1])
a2 = fig.add_subplot(gs_a[0])
a1 = fig.add_subplot(gs_a[1])
a3 = fig.add_subplot(gs_a[2])

gs_b = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs[1, 0:3],
                                        hspace=0.05, wspace=0.3,
                                        height_ratios=[1, 1, 2])
g1 = fig.add_subplot(gs_b[0, 0])
f1 = fig.add_subplot(gs_b[0, 1])
h1 = fig.add_subplot(gs_b[0, 2])
g2 = fig.add_subplot(gs_b[1, 0])
f2 = fig.add_subplot(gs_b[1, 1])
h2 = fig.add_subplot(gs_b[1, 2])
d  = fig.add_subplot(gs_b[2, 0:3])

c1 = fig.add_subplot(gs[0, 3:5])
i1 = fig.add_subplot(gs[1, 3:5])
j1 = fig.add_subplot(gs[0, 5:7])
j2 = fig.add_subplot(gs[0, 7:9])
k1 = fig.add_subplot(gs[1, 5:7])
k2 = fig.add_subplot(gs[1, 7:9])

fig.text(0.01, 1,    'a', fontsize=10, fontweight='bold', va='top')
fig.text(0.01, 0.5,  'b', fontsize=10, fontweight='bold', va='top')
fig.text(0.36, 1,    'c', fontsize=10, fontweight='bold', va='top')
fig.text(0.36, 0.5,  'd', fontsize=10, fontweight='bold', va='top')
fig.text(0.56, 1,    'e', fontsize=10, fontweight='bold', va='top')
fig.text(0.56, 0.5,  'f', fontsize=10, fontweight='bold', va='top')

# ── Panel a ───────────────────────────────────────────────────────────────
os.chdir(save_path)
color = plt.cm.viridis(np.linspace(0, 1, 4))
mpl.rcParams['axes.prop_cycle'] = cycler(color=color)

file_name = 'single_trial_example_432'
big_data = pd.read_csv(path+file_name+'.csv', index_col=0)
file_name = 'single_trial_example_df_432'
dft = pd.read_csv(path+file_name+'.csv', index_col=0)
dft['a_Stimulus_ON'] = dft['fixed_times'] - dft['Stimulus_ON']

delay = 3
align = 'Stimulus_ON'
cue_on, cue_off = 0, 0.4
start = -2
stop = max(dft['a_'+align])

big_data['time_centered'] = big_data['times'] - big_data['Stimulus_ON']
big_data['time_centered'] = np.round(big_data.time_centered/1000, 2)
big_data['firing_'] = big_data['firing']*1000

df_results = pd.DataFrame(dtype=float)
df_results['firing'] = big_data.loc[big_data.time_centered <= stop].groupby(['time_centered','neuron'])['firing_'].mean()
df_results['error']  = big_data.loc[big_data.time_centered <= stop].groupby(['time_centered','neuron'])['firing_'].std()
df_results.reset_index(inplace=True)

# a2 — raster
panel = a2
cluster_id, FR_mean = [], []
start_FR, stop_FR = 0.2, delay
for N in dft.cluster_id.unique():
    spikes = dft.loc[(dft.cluster_id==N)&(dft['a_'+align]>start_FR)&(dft['a_'+align]<stop_FR)]['a_'+align].values
    FR_mean.append(len(spikes)/abs(stop_FR-start_FR))
    cluster_id.append(N)
df_spikes = pd.DataFrame(list(zip(cluster_id, FR_mean)), columns=['cluster_id','FR'])
df_spikes = df_spikes.sort_values('FR')
df_spikes['new_order'] = np.arange(len(df_spikes))
dft = pd.merge(df_spikes, dft, on=['cluster_id'])

j = 0
for N in dft.new_order.unique():
    spikes = dft.loc[dft.new_order==N]['a_'+align].values
    j += 1
    panel.plot(spikes, np.repeat(j, len(spikes)), '|', markersize=0.5, color='black', zorder=1)
panel.set_ylabel('Neurons')
panel.set_ylim(0, j)
panel.set_xlim(start, stop)
panel.set_title('Session E20_2022-02-14')
panel.tick_params(bottom=False, labelbottom=False)
y = np.arange(0, j+1, 0.1)
panel.fill_betweenx(y, cue_on, cue_off, color='lightgrey', alpha=1, linewidth=0)
panel.fill_betweenx(y, cue_off+delay, cue_off+delay+.2, color='lightgrey', alpha=1, linewidth=0)
sns.despine(ax=panel)
panel.spines['bottom'].set_visible(False)

# a1 — PSTH (viridis colormap applied via cycler set above)
panel = a1
for N in df_results.neuron.unique():
    panel.plot(df_results.loc[df_results.neuron==N].time_centered,
               df_results.loc[df_results.neuron==N].firing, label=N, alpha=0.5)
panel.set_xlim(start, stop)
panel.tick_params(bottom=False, labelbottom=False)
y = np.arange(0, 120, 0.1)
panel.fill_betweenx(y, cue_on, cue_off, color='lightgrey', alpha=1, linewidth=0)
panel.fill_betweenx(y, cue_off+delay, cue_off+delay+.2, color='lightgrey', alpha=1, linewidth=0)
panel.set_ylabel('Firing rate\n(spks/s)')
sns.despine(ax=panel)
panel.spines['bottom'].set_visible(False)

# a3 — synch rate (keeps x-axis)
panel = a3
T = 432
synch_trial(dft, T, a3, trial=T, start=-2, stop=10, bins=30, color='black')
panel.set_ylim(-5, 25)
panel.set_xlim(start, stop)
panel.set_xlabel('Time from Stimulus onset (s)')
y = np.arange(-5, 25, 0.1)
panel.fill_betweenx(y, cue_on, cue_off, color='lightgrey', alpha=1, linewidth=0)
panel.fill_betweenx(y, cue_off+delay, cue_off+delay+.2, color='lightgrey', alpha=1, linewidth=0)
sns.despine(ax=panel)

# ── Panel b ───────────────────────────────────────────────────────────────
file_name = 'synch_data_trials_2beforeSti'
df_final = pd.read_csv(path+file_name+'.csv', index_col=0)
df_final['state'] = np.where(df_final['WM_roll'] > 0.5, 0, 1)

animal = "E22_2022-01-13_16-34-24.csv"
threshold = 0.5
df_session = df_final.loc[df_final.animal == animal]

fig.text(0.07, 0.505, 'Mouse E22 13-01', fontsize=7, fontweight='bold', ha='left', va='bottom')

for ax_r, ax_fr, T in [(g1,g2,153),(f1,f2,212),(h1,h2,340)]:
    filename = 'single_trial_synch_'+str(T)
    df = pd.read_csv(path+filename+'.csv', sep=',', index_col=0)
    synch_trial(df, T, ax_r, ax_fr, trial=T)
    ax_r.set_title(str(T), fontsize=6)
    ax_r.tick_params(bottom=False, labelbottom=False)
    ax_r.set_xlabel('')
    ax_fr.tick_params(bottom=False, labelbottom=False)
    ax_fr.set_xlabel('')
    # no spines on raster — applied immediately after synch_trial
    for sp in ax_r.spines.values(): sp.set_visible(False)
    ax_r.tick_params(left=False, labelleft=False)
    # no spines on FR except left of first (g2)
    for sp in ['top','right','bottom']: ax_fr.spines[sp].set_visible(False)
    if ax_fr != g2:
        ax_fr.spines['left'].set_visible(False)
        ax_fr.tick_params(left=False, labelleft=False)

# Session synch trace
panel = d
panel.fill_between(df_session['trial'], 0.9, 2.5,
                   where=df_session['WM_roll'] <= threshold,
                   facecolor='indigo', alpha=0.3)
panel.fill_between(df_session['trial'], 0.9, 2.5,
                   where=df_session['WM_roll'] >= threshold,
                   facecolor='darkgreen', alpha=0.3)
sns.lineplot(x="trial", y="synch_window", data=df_session, color='black', ci=68, ax=panel)
panel.set_ylabel('Synch')
panel.set_ylim(0.9, max(df_session.synch_window)+0.3)
panel.set_xlabel('Trials')

synch_vals = {T: df_session.loc[df_session['trial']==T, 'synch_window'].mean()
              for T in [153, 212, 340]}
for T_mark in [153, 212, 340]:
    y_val = synch_vals.get(T_mark, 1.5)
    panel.plot(T_mark, y_val, 'o', color='crimson', markersize=5,
               markerfacecolor='none', markeredgewidth=1.2, zorder=5)

# Arrows from synch trace up to each mini FR panel
fig.canvas.draw()
for T_mark, target_ax in [(153, g2), (212, f2), (340, h2)]:
    y_val = synch_vals.get(T_mark, 1.5)
    start_pt = fig.transFigure.inverted().transform(
        d.transData.transform([T_mark, y_val + 0.05]))
    end_pt = fig.transFigure.inverted().transform(
        target_ax.transAxes.transform([0.5, 0.0]))
    d.annotate('', xy=(end_pt[0], end_pt[1]), xytext=(start_pt[0], start_pt[1]),
        xycoords='figure fraction', textcoords='figure fraction',
        arrowprops=dict(arrowstyle='->', color='crimson', lw=0.8),
        annotation_clip=False)

sns.despine(ax=d)

# ── Panel c ───────────────────────────────────────────────────────────────
panel = c1
df_results = pd.DataFrame()
df_results['synch'] = df_final.groupby(['animal','state']).synch.mean()
df_results.reset_index(inplace=True)
df_results['state'] = pd.Categorical(df_results['state'], categories=[0,1], ordered=True)
df_results['state'] = np.where(df_results.state==0, 'STM', 'RepL')

palette = sns.color_palette(['black'], len(df_results.animal.unique()))
sns.lineplot(x="state", y="synch", data=df_results, hue='animal', alpha=0.8,
             palette=palette, ax=panel, linewidth=0.2, markeredgewidth=0.2,
             marker='', legend=False, markersize=3)
sns.boxplot(x='state', y="synch", data=df_results, width=0.5, showfliers=False,
            palette=['darkgreen','indigo'], ax=panel, linewidth=1)
panel.set_xticks([0,1])
panel.set_xticklabels(['STM','RepL'])
panel.set_ylabel('Synch')
panel.set_ylim(1, 3)
panel.set_yticks([1, 2, 3])
panel.legend_.remove() if panel.legend_ else None
add_stat_annotation(panel, data=df_results, x='state', y='synch',
                    box_pairs=[('STM','RepL')], test='t-test_paired',
                    text_format='star', loc='inside',
                    line_offset_to_box=0.05, text_offset=-0.5, line_offset=0,
                    verbose=1, fontsize=6, linewidth=0.5)
sns.despine(ax=panel)
panel.spines['bottom'].set_visible(False)
panel.tick_params(bottom=True)

# ── Panel d ───────────────────────────────────────────────────────────────
file_name = 'synch_corrdata_final'
df_corr = pd.read_csv(path+file_name+'.csv', index_col=0)

panel = i1
sns.boxplot(data=df_corr, palette=['grey','grey','grey'], ax=panel,
            order=['r_WM_shuff','r_acc_shuff','r_repeat_shuff'],
            saturation=0.6, linewidth=1, width=0.5)

for xi, col in enumerate(['r_WM_shuff','r_acc_shuff','r_repeat_shuff']):
    xA = np.random.normal(xi, 0.2, len(df_corr))
    sns.scatterplot(x=xA, y=col, data=df_corr, alpha=0.9, ax=panel,
                    color='black', legend=False)
    p = stats.ttest_1samp(df_corr[col], 0)[1]
    stars = '***' if p<=0.001 else ('**' if p<=0.01 else ('*' if p<=0.05 else 'ns'))
    panel.text(xi, 1.01, stars, ha='center', va='bottom', fontsize=7,
               transform=panel.get_xaxis_transform())
    print(stats.ttest_1samp(df_corr[col], 0))

panel.hlines(y=0, xmin=-0.5, xmax=2.5, linestyle=':')
panel.set_ylabel('Corr. coef.\n(Synch, X)')
panel.set_xlabel('')
panel.set_xticklabels(['p(STM)','Accuracy','RB'])
sns.despine(ax=panel)
panel.spines['bottom'].set_visible(False)
panel.tick_params(bottom=True)

# ── Panel e ───────────────────────────────────────────────────────────────
panel = j1
file_name = 'auto_corrs_indiv_session'
df = pd.read_csv(path+file_name+'.csv', header=None, index_col=0)
panel.plot(df.index, df[1], color='indigo')
panel.plot(df.index, df[2], color='darkgreen')
panel.hlines(xmin=-1, xmax=1, y=0, linestyle=':')
panel.set_xlim(-1, 1)
panel.set_xticks([-1, 0, 1])
panel.set_xlabel('Time lag (s)')
panel.set_ylabel('Population rate\nAutocorrelogram')
panel.text(0.05, 0.95, 'Session E11_2021-05-12', transform=panel.transAxes,
           fontsize=6, va='top')
sns.despine(ax=panel)
panel.tick_params(bottom=True)

panel = j2
file_name = 'psd_ratio_indiv_session'
df = pd.read_csv(path+file_name+'.csv', header=None, index_col=0)
panel.scatter(x=6.2, y=2.5, color='crimson')
panel.plot(df.index, df[1], color='black')
panel.set_xscale('log')
panel.set_xlim(1.99, 100)
panel.set_ylim(0.8, 3)
panel.set_yticks([1, 2, 3])
panel.set_xticks([2, 10, 100])
panel.set_xticklabels(['2','10','100'])
panel.hlines(xmin=2, xmax=100, y=1, linestyle=':')
panel.set_xlabel('Frequency (Hz)')
panel.set_ylabel('PSD Ratio RepL/STM')
panel.text(0.05, 0.95, 'Session E11_2021-05-12', transform=panel.transAxes,
           fontsize=6, va='top')
sns.despine(ax=panel)
panel.tick_params(bottom=True, which='both')

# ── Panel f ───────────────────────────────────────────────────────────────
panel = k1
file_name = 'avg_PSDs_V2'
df = pd.read_csv(path+file_name+'.csv', header=None, index_col=0)
panel.plot(df.index, df[1], color='indigo')
panel.plot(df.index, df[2], color='darkgreen')
panel.set_yscale('log')
panel.set_xscale('log')
panel.set_xlim(2, 99)
panel.set_ylim(0.01, 0.8)
panel.set_xticks([2, 10, 100])
panel.set_xticklabels(['2','10','100'])
panel.hlines(xmin=3.1, xmax=15.6, y=0.015, linewidth=2, color='indigo')
panel.text(7, 0.018, '[3.1, 15.6]', ha='center', va='bottom', fontsize=5)
panel.set_ylabel('Population rate\nPower Spectral Density')
panel.set_xlabel('Frequency (Hz)')
panel.tick_params(axis='both', which='both', direction='out')
sns.despine(ax=panel)

panel = k2
file_name = 'AVG_psd_ratio_w_band'
df = pd.read_csv(path+file_name+'.csv', header=None, index_col=0)
x, y, y_min, y_max = df.index, df[1], df[3], df[2]
panel.plot(x, y, color='black')
panel.set_xscale('log')
panel.set_xlim(1.99, 100)
panel.set_ylim(0.8, 2)
panel.set_xticks([2, 10, 100])
panel.set_xticklabels(['2','10','100'])
panel.set_yticks([0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
panel.hlines(xmin=2, xmax=100, y=1, linestyle=':')
panel.vlines(ymin=0.8, ymax=1.8, x=4.4, color='crimson')
panel.text(4.6, 1.75, 'freq = 4.4Hz', color='crimson', fontsize=6)
panel.fill_between(x, y_min, y_max, color='gray', alpha=0.3)
panel.set_ylabel('PSD ratio RepL/STM')
panel.set_xlabel('Frequency (Hz)')
panel.tick_params(axis='both', which='both', direction='out')
sns.despine(ax=panel)

# ── Finalise ──────────────────────────────────────────────────────────────
sns.despine()

# ── Post-despine fixes (applied AFTER global despine so they are not overridden) ──

# a: same lightgrey shading, no bottom spine on raster and PSTH
a2.spines['bottom'].set_visible(False)
a1.spines['bottom'].set_visible(False)

# b: no spines on raster panels (g1, f1, h1)
for ax in [g1, f1, h1]:
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(left=False, labelleft=False)

# b: no spines on FR panels except left spine of g2 only
for ax in [f2, h2]:
    for sp in ['top', 'right', 'bottom', 'left']: ax.spines[sp].set_visible(False)
    ax.tick_params(left=False, labelleft=False)
g2.spines['top'].set_visible(False)
g2.spines['right'].set_visible(False)
g2.spines['bottom'].set_visible(False)

# c: no bottom spine, tick marks visible
c1.spines['bottom'].set_visible(False)
c1.tick_params(bottom=True)
c1.set_ylim(1, 3)
c1.set_yticks([1, 2, 3])

# d: no bottom spine, tick marks visible, correct ylabel
i1.spines['bottom'].set_visible(False)
i1.tick_params(bottom=True)
i1.set_ylabel('Corr. coef.\n(Synch, X)')
i1.set_xlabel('')

# e: tick marks visible on bottom
j1.tick_params(bottom=True)
j1.set_xticks([-1, 0, 1])
j1.set_ylabel('Population rate\nAutocorrelogram')
j2.tick_params(bottom=True, which='both')
j2.set_ylabel('PSD Ratio RepL/STM')
j2.set_yticks([1, 2, 3])

# f: ylabels (not titles)
k1.set_ylabel('Population rate\nPower Spectral Density')
k2.set_ylabel('PSD ratio RepL/STM')

# Force redraw to ensure all changes are applied
fig.canvas.draw()

# plt.savefig(save_path+'/fig_6_synch.svg', bbox_inches='tight', dpi=1000)
# plt.savefig(save_path+'/fig_6_synch.pdf', bbox_inches='tight', dpi=1000)

plt.show()