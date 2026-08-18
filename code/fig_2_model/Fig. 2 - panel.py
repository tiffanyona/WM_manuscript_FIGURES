# -*- coding: utf-8 -*-
"""
fig_2_model.py — revised version
Changes vs original:
  - All hlines/vlines baselines: color='black' (was blue default)
  - Panels i/j (b/b2): legend replaced with inline text labels
  - Panel e (x): new purple color palette, violin+boxplot layering, stars
  - Panels b/c (c/c2): new purple color palette, inline text labels, no HMM line
  - Panels f/g (a/a2): ylabel/legend updated, session title inside
  - HMM parameter violins (d panels): indigo -> purple color
"""
COLORLEFT = 'teal'
COLORRIGHT = '#FF8D3F'

import warnings
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
warnings.filterwarnings('ignore', 'Attempting to set identical low and high xlims')
warnings.filterwarnings('ignore', 'FigureCanvasAgg is non-interactive')
from matplotlib.lines import Line2D
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import ROOT, FIGURES_OUT, DATA_DIR

sys.path.insert(0, str(ROOT / 'src'))
from functions import add_stat_annotation, figureplot, compute_window_centered
import functions as plots

save_path = str(FIGURES_OUT) + '/fig_2_model/'
Path(save_path).mkdir(parents=True, exist_ok=True)
path = str(DATA_DIR) + '/fig_2_model/'
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

# Shared color palette for DW models (used in panels b/c and e)
DW_COLORS  = ['lightgrey', '#b8b8d1', '#2d2d7a',  "#6b6b6b"]
DW_MODELS  = ['9', '12', '11', '10']
DW_LABELS  = ['DW', 'DW-L', 'DW-B', 'DW-M']
DW_HEIGHT = [0.46, 0.56, 0.63, 0.75]  # approximate y from target image

fig = plt.figure(figsize=(17*cm, 21*cm))
gs = gridspec.GridSpec(nrows=5, ncols=8, figure=fig)

a  = fig.add_subplot(gs[2, 0:3])
a2 = fig.add_subplot(gs[2, 3:6])
x  = fig.add_subplot(gs[1, 4:8])
x2 = fig.add_subplot(gs[2, 6:8])
b  = fig.add_subplot(gs[3, 0:2])
b2 = fig.add_subplot(gs[3, 2:4])
c  = fig.add_subplot(gs[0, 4:6])
c2 = fig.add_subplot(gs[0, 6:8])
d1 = fig.add_subplot(gs[3, 4:6])
d2 = fig.add_subplot(gs[4, 0:1])
d3 = fig.add_subplot(gs[4, 1:2])
d4 = fig.add_subplot(gs[3, 6:8])
d5 = fig.add_subplot(gs[4, 2:3])
d6 = fig.add_subplot(gs[4, 3:4])
g2 = fig.add_subplot(gs[4, 4:6])
g3 = fig.add_subplot(gs[4, 6:8])

fig.text(0.01, 0.99, 'a', fontsize=10, fontweight='bold', va='top')
fig.text(0.5,  0.99, 'b', fontsize=10, fontweight='bold', va='top')
fig.text(0.75, 0.99, 'c', fontsize=10, fontweight='bold', va='top')
fig.text(0.01, 0.8,  'd', fontsize=10, fontweight='bold', va='top')
fig.text(0.5,  0.8,  'e', fontsize=10, fontweight='bold', va='top')
fig.text(0.01, 0.6,  'f', fontsize=10, fontweight='bold', va='top')
fig.text(0.36, 0.6,  'g', fontsize=10, fontweight='bold', va='top')
fig.text(0.75, 0.6,  'h', fontsize=10, fontweight='bold', va='top')
fig.text(0.01, 0.4,  'i', fontsize=10, fontweight='bold', va='top')
fig.text(0.25, 0.4,  'j', fontsize=10, fontweight='bold', va='top')
fig.text(0.5,  0.4,  'k', fontsize=10, fontweight='bold', va='top')
fig.text(0.75, 0.4,  'l', fontsize=10, fontweight='bold', va='top')
fig.text(0.01, 0.22, 'm', fontsize=10, fontweight='bold', va='top')
fig.text(0.15, 0.22, 'n', fontsize=10, fontweight='bold', va='top')
fig.text(0.3,  0.22, 'o', fontsize=10, fontweight='bold', va='top')
fig.text(0.4,  0.22, 'p', fontsize=10, fontweight='bold', va='top')
fig.text(0.5, 0.22, 'q', fontsize=10, fontweight='bold', va='top')
fig.text(0.75, 0.22, 'r', fontsize=10, fontweight='bold', va='top')

# ---------------------------------------------------------------------------
# Panels i/j (b/b2) — accuracy and repeating bias by delay
# ---------------------------------------------------------------------------
threshold = 0.5
groupings = ['subject', 'delays', 'state']

file = 'panels_fg_ij_behavior_all_trials_with_HMM_state_posteriors'
df = pd.read_csv(path + file + '.csv', low_memory=False)
df['WM_roll'] = compute_window_centered(df, 3, 'WM')
df['state']   = np.where(df.WM_roll > threshold, 1, 0)

file = 'panels_ij_HMM_model_simulated_trials_with_state_posteriors'
df_model = pd.read_csv(path + file + '.csv')
df_model = df_model.loc[df_model.animal_delay == 10]
df       = df.loc[df.animal_delay == 10]

# Real data — by state
df_results = pd.DataFrame()
df_results['hit'] = df.groupby(groupings)['hit'].mean()
df_results['repeat_choice'] = (
    0.5 * df.loc[df['repeat_choice_side']==1].groupby(groupings)['choices'].count() /
    df.loc[df.choices==-1].groupby(groupings)['choices'].count() +
    0.5 * df.loc[df['repeat_choice_side']==2].groupby(groupings)['choices'].count() /
    df.loc[df.choices==1].groupby(groupings)['choices'].count())
df_results.reset_index(inplace=True)
sns.lineplot(x='delays', y='hit', data=df_results, marker='o', hue='state',
             markeredgewidth=0.2, palette=['indigo','darkgreen'], ax=b,
             errorbar=('ci',95), linestyle='', legend=False, err_style='bars')
sns.lineplot(x='delays', y='repeat_choice', data=df_results, hue='state',
             markeredgewidth=0.2, palette=['indigo','darkgreen'], marker='o',
             errorbar=('ci',95), legend=False, linestyle='', ax=b2, err_style='bars')

# Real data — all trials
groupings = ['subject', 'delays']
df_results = pd.DataFrame()
df_results['repeat_choice'] = (
    0.5 * df.loc[df['repeat_choice_side']==1].groupby(groupings)['choices'].count() /
    df.loc[df.choices==-1].groupby(groupings)['choices'].count() +
    0.5 * df.loc[df['repeat_choice_side']==2].groupby(groupings)['choices'].count() /
    df.loc[df.choices==1].groupby(groupings)['choices'].count())
df_results['hit'] = df.groupby(groupings)['hit'].mean()
df_results.reset_index(inplace=True)
sns.lineplot(x='delays', y='hit', data=df_results, marker='o', ax=b,
             markeredgewidth=0.2, color='black', errorbar=('ci',95),
             linestyle='', legend=False, err_style='bars')
sns.lineplot(x='delays', y='repeat_choice', data=df_results, marker='o',
             markeredgewidth=0.2, color='black', errorbar=('ci',95),
             legend=False, linestyle='', ax=b2, err_style='bars')

# HMM model — by state
groupings = ['subject', 'delays', 'state']
df_results = pd.DataFrame()
df_results['hit'] = df_model.groupby(groupings)['hit'].mean()
df_results['repeat_choice'] = (
    0.5 * df_model.loc[df_model['repeat_choice_side']==1].groupby(groupings)['choices'].count() /
    df_model.loc[df_model.choices==-1].groupby(groupings)['choices'].count() +
    0.5 * df_model.loc[df_model['repeat_choice_side']==2].groupby(groupings)['choices'].count() /
    df_model.loc[df_model.choices==1].groupby(groupings)['choices'].count())
df_results.reset_index(inplace=True)
sns.lineplot(x='delays', y='hit', data=df_results, hue='state', marker='',
             ax=b, palette=['indigo','darkgreen'], errorbar=('ci',95),
             legend=False, err_style=None)
sns.lineplot(x='delays', y='repeat_choice', hue='state', data=df_results,
             palette=['indigo','darkgreen'], marker='', errorbar=('ci',95),
             ax=b2, err_style=None, legend=False)
df_results_state = df_results.copy()

b.set_ylim(0.4, 1)
b.hlines(xmin=0, xmax=10, y=0.5, linestyles=':', color='black')
b.set_ylabel('Accuracy')
b.set_xlabel('Delay (s)')
b.locator_params(nbins=3)

# HMM model — all trials
groupings = ['subject', 'delays']
df_results = pd.DataFrame()
df_results['hit'] = df_model.groupby(groupings)['hit'].mean()
df_results['repeat_choice'] = (
    0.5 * df_model.loc[df_model['repeat_choice_side']==1].groupby(groupings)['choices'].count() /
    df_model.loc[df_model.choices==-1].groupby(groupings)['choices'].count() +
    0.5 * df_model.loc[df_model['repeat_choice_side']==2].groupby(groupings)['choices'].count() /
    df_model.loc[df_model.choices==1].groupby(groupings)['choices'].count())
df_results.reset_index(inplace=True)
sns.lineplot(x='delays', y='hit', data=df_results, marker='', ax=b,
             color='black', errorbar=('ci',95), legend=False, err_style=None)
sns.lineplot(x='delays', y='repeat_choice', data=df_results, color='black',
             marker='', errorbar=('ci',95), ax=b2, err_style=None)

b2.set_ylim(0.4, 1)
b2.set_ylabel('Repeating bias')
b2.set_xlabel('Delay (s)')
b2.hlines(y=0.5, xmin=0, xmax=10, linestyle=':', color='black')
b2.locator_params(nbins=3)

if b.get_legend():  b.get_legend().remove()
if b2.get_legend(): b2.get_legend().remove()

# Inline text labels on top of model lines in b2
_lines = [l for l in b2.get_lines() if len(l.get_ydata()) > 1]
_rl_line, _wm_line, _all_line = _lines[-3], _lines[-2], _lines[-1]
for line, label, color, va in [
    (_rl_line,  'RepL trials', 'indigo',    'bottom'),
    (_wm_line,  'STM trials',  'darkgreen', 'top'),
    (_all_line, 'All trials',  'black',     'bottom'),
]:
    _xdata = line.get_xdata()
    _ydata = line.get_ydata()
    _mid   = len(_xdata) // 2
    b2.text(_xdata[_mid], _ydata[_mid], label, color=color,
            fontsize=6, ha='center', va=va)

# ---------------------------------------------------------------------------
# Panel e (x) — model LL comparison
# ---------------------------------------------------------------------------
panel = x
file_name = 'panel_e_per_trial_log_likelihood_HMM_vs_DW_models'
full_fit = pd.read_csv(path + file_name + '.csv', index_col=0)

order_list    = ["all", '12', '9', '10', '11']
color_palette = {"all": 'black', "12": 'lightgrey',
                 "9": '#b8b8d1', "10": "#888889", "11": '#2d2d7a'}

sns.violinplot(x='model', y='substracted', data=full_fit, order=order_list,
               hue='model', palette=color_palette, legend=False,
               linewidth=0, alpha=0.4, width=0.5, ax=panel, zorder=1)
sns.stripplot(x='model', y='substracted', data=full_fit, jitter=0.3, size=3,
              order=order_list, hue='model', palette=color_palette,
              legend=False, edgecolor='none', linewidth=0, ax=panel, zorder=2)
sns.boxplot(x='model', y='substracted', data=full_fit, order=order_list,
            hue='model', palette=color_palette, legend=False,
            width=0.12, showcaps=False, showfliers=False, ax=panel,
            boxprops=dict(zorder=4, linewidth=1, color='black'),
            whiskerprops=dict(zorder=4, linewidth=1, color='black'),
            medianprops=dict(color='white', linewidth=1.5, zorder=5))

panel.hlines(y=0, xmin=-0.5, xmax=4.5, linestyle=':', color='black')
panel.set_xlabel('')
panel.set_ylabel('LL difference (bits/trial)')
panel.set_xticks(range(len(order_list)))
panel.set_xticklabels(["HMM", 'DW', 'DW-L', 'DW-M', 'DW-B'])

# Stars: 1-sample t-test vs 0 for each model
for i, model in enumerate(order_list):
    if model == 'all':
        continue
    vals = full_fit.loc[full_fit.model == model, 'substracted'].values
    _, p = stats.ttest_1samp(vals, 0)
    star = plots.p_to_stars(p)
    y_top = full_fit.loc[full_fit.model == model, 'substracted'].max()
    panel.text(i, y_top + 0.003, star, ha='center', va='bottom', fontsize=7)

# ---------------------------------------------------------------------------
# Panels b/c (c/c2) — model predictions vs real data
# ---------------------------------------------------------------------------
groupings = ['subject', 'delays']

df_results = pd.DataFrame()
df_results['hit'] = df.groupby(groupings)['hit'].mean()
df_results['repeat_choice'] = (
    0.5 * df.loc[df['repeat_choice_side']==1].groupby(groupings)['choices'].count() /
    df.loc[df.choices==-1].groupby(groupings)['choices'].count() +
    0.5 * df.loc[df['repeat_choice_side']==2].groupby(groupings)['choices'].count() /
    df.loc[df.choices==1].groupby(groupings)['choices'].count())
df_results.reset_index(inplace=True)
sns.lineplot(x='delays', y='hit', data=df_results, marker='o',
             markeredgewidth=0.1, ax=c, color='black',
             errorbar=('ci',95), linestyle='', legend=False, err_style='bars')
sns.lineplot(x='delays', y='repeat_choice', data=df_results, marker='o',
             markeredgewidth=0.1, color='black', errorbar=('ci',95),
             legend=False, linestyle='', ax=c2, err_style='bars')

for model, color, label in zip(DW_MODELS, DW_COLORS, DW_LABELS):
    df_results = pd.read_csv(path + 'panels_bc_DW_model' + model + '_predicted_accuracy_and_repeat_bias.csv')
    sns.lineplot(x='delays', y='hit', data=df_results, marker='', ax=c,
                 color=color, errorbar=('ci',95), legend=False, err_style=None)
    sns.lineplot(x='delays', y='repeat_choice', data=df_results, color=color,
                 marker='', errorbar=('ci',95), ax=c2, err_style=None)
    c.set_ylim(0.45, 1)
    c.hlines(xmin=0, xmax=30, y=0.5, linestyles=':', color='black')
    c.set_ylabel('Accuracy')
    c.set_xlabel('Delay (s)')
    c2.set_ylim(0.45, 0.8)
    c2.set_ylabel('Repeating bias')
    c2.set_xlabel('Delay (s)')
    c2.hlines(y=0.5, xmin=0, xmax=30, linestyle=':', color='black')

# Inline text labels: fixed positions at right side matching target layout
# Target order top to bottom: DW-M, DW-B, DW-L, DW
_c2_label_x = 24
for color, label, y_fixed in zip(
    DW_COLORS, DW_LABELS,
    DW_HEIGHT
):
    c2.text(_c2_label_x, y_fixed, label, color=color,
            fontsize=6, ha='left', va='center')

c.locator_params(nbins=3)
c2.locator_params(nbins=3)

# ---------------------------------------------------------------------------
# Panels k-p (d panels) — HMM fitted parameter distributions
# ---------------------------------------------------------------------------
file_name = 'panels_k-p_HMM_fitted_parameters_all_animals'
full_fit  = pd.read_csv(path + file_name + '.csv', index_col=0)
full_fit  = full_fit.loc[full_fit.delay == 10]
full_fit['alfa']  = full_fit['c2'] / 2
full_fit['const'] = 1

_ylabel_override = None
for regressor, panel, color in zip(
    ['P_L', 'P_R', 'alfa', 'mu_b', 'WM', 'RL', 'beta_w', 'beta_bias'],
    [d1, d1, d2, d3, d4, d4, d5, d6],
    ['darkgreen','darkgreen','darkgreen','darkgreen','grey','grey','indigo','indigo']
):
    xA = np.random.normal(0, 0.1, len(full_fit))
    if regressor == 'P_R':
        plot = pd.DataFrame({'PR': full_fit['P_R'], 'PL': full_fit['P_L']})
        sns.violinplot(data=plot, palette=[color, color], ax=panel, alpha=0)
        sns.violinplot(data=plot, palette=[color, color], ax=panel,
                       width=0.5, linewidth=0, zorder=1, alpha=0.2)
        xA = np.random.normal(1, 0.1, len(full_fit))
        panel.set_xlabel('$P_L$                $P_R$')
        _ylabel_override = 'Loading prob.'
    elif regressor == 'RL':
        plot = pd.DataFrame({'p(WM)': full_fit['WM'], 'p(RL)': full_fit['RL']})
        sns.violinplot(data=plot, palette=[color, color], ax=panel,
                       width=0.5, alpha=0)
        sns.violinplot(data=plot, palette=[color, color], ax=panel,
                       width=0.5, linewidth=0, zorder=1, alpha=0.2)
        xA = np.random.normal(1, 0.1, len(full_fit))
        panel.set_xlabel('p(STM)          p(RepL)')
        _ylabel_override = 'Posterior prob.'
    elif regressor in ('P_L', 'WM'):
        pass
    else:
        xA = np.random.normal(0, 0.20, len(full_fit))
        sns.violinplot(x=full_fit['const'].astype(float),
                       y=full_fit[regressor].astype(float),
                       data=full_fit, width=1, color=color, ax=panel,
                       zorder=1, alpha=0.2, linewidth=0)
        sns.violinplot(x=full_fit['const'].astype(float),
                       y=full_fit[regressor].astype(float),
                       data=full_fit, width=1, color=color, ax=panel,
                       legend=False, linewidth=1, alpha=0)

    sns.scatterplot(x=xA, y=regressor, data=full_fit, ax=panel,
                    alpha=0.7, style='delay', color=color, legend=False, size=8)
    panel.set_ylabel(_ylabel_override if _ylabel_override else '')
    _ylabel_override = None
    panel.set_xlim(-0.7, 0.7)

    if regressor == 'alfa':
        panel.set_xticks([])
        panel.set_xlabel('')
        panel.set_ylabel('DW barrier $\\alpha$', fontsize=6)
        panel.set_ylim(1.5, 3.1)
    elif regressor == 'mu_b':
        panel.set_xticks([])
        panel.set_xlabel('')
        panel.set_ylabel('DW asymmetry $\\mu$', fontsize=6)
    elif regressor == 'beta_w':
        panel.set_xticks([])
        panel.set_xlabel('')
        panel.set_ylabel('Repeating strength $\\beta_C$', fontsize=6)
    elif regressor == 'beta_bias':
        panel.set_xticks([])
        panel.set_xlabel('')
        panel.set_ylabel('Choice bias $\\beta_{bias}$', fontsize=6)
    elif regressor == 'P_R':
        panel.set_xticks([0, 1])
        panel.set_xticklabels(['$P_L$', '$P_R$'], fontsize=6)
        panel.set_ylabel('Loading prob.', fontsize=6)
        panel.set_ylim(-0.1, 1.1)
        panel.set_xlim(-0.5, 1.5)
    elif regressor == 'RL':
        panel.set_xticks([0, 1])
        panel.set_xticklabels(['p(STM)', 'p(RepL)'], fontsize=6)
        panel.set_ylabel('Posterior prob.', fontsize=6)
        panel.set_ylim(-0.1, 1.1)
        panel.set_xlim(-0.5, 1.5)
    elif regressor in ('P_L', 'WM'):
        panel.set_xticks([])

    panel.hlines(y=0, xmin=-1, xmax=1.5, linestyle=':', color='black')
    panel.locator_params(axis='y', nbins=5)
    panel.locator_params(nbins=3)

# ---------------------------------------------------------------------------
# Panel h (x2) — posterior p(WM) histogram
# ---------------------------------------------------------------------------
df_summary = pd.read_csv(path + 'panel_h_HMM_state_probability_histogram_all_sessions.csv', index_col=0)
panel  = x2
patches = panel.bar(np.arange(0, 10),
                    df_summary.astype(float).mean(axis=0),
                    yerr=df_summary.sem(axis=0),
                    color='grey', alpha=0.3)
for i in range(5, 10):
    patches[i].set_facecolor('darkgreen')
for i in range(0, 5):
    patches[i].set_facecolor('indigo')
panel.set_ylabel('Probability')
panel.set_xlabel('Posterior p(STM)')
panel.vlines(x=threshold*10 - 0.5,
             ymax=max(df_summary.astype(float).mean(axis=0)), ymin=0,
             linestyle=':', color='black')
panel.set_xticks(range(11))
panel.set_xticklabels([0.1,'',0.3,'',0.5,'',0.7,'',0.9,'',''])
# State labels above the dotted line
_y_top = max(df_summary.astype(float).mean(axis=0))
panel.text(1.5, _y_top * 0.95, 'RepL\n state', fontsize=5, ha='center', color='indigo')
panel.text(7.5, _y_top * 0.95, 'STM\n state',  fontsize=5, ha='center', color='darkgreen')

# ---------------------------------------------------------------------------
# Panels f/g (a/a2) — example session p(STM) and running RB
# ---------------------------------------------------------------------------
file = 'panels_fg_example_session_10s_delay_with_HMM_state_posteriors'
df = pd.read_csv(path + file + '.csv', low_memory=False)
df['WM_roll'] = compute_window_centered(df, 3, 'WM')
df['state']   = np.where(df.WM_roll > threshold, 1, 0)

animal  = 'C38'
session = 8
temp_df = df.loc[(df['session'] == session) & (df.subject == animal)].copy()
temp_df['accuracy']      = compute_window_centered(temp_df, 20, 'hit')
temp_df['repeat_choice'] = compute_window_centered(temp_df, 20, 'repeat')
temp_df['WM_roll']       = compute_window_centered(temp_df, 3, 'WM')
print(temp_df.day.unique())

# Panel g (a2) — Running RB
panel = a2
panel.fill_between(temp_df['trial'], 0, 1, where=temp_df['WM'] <= threshold,
                   facecolor='indigo', alpha=0.3)
panel.fill_between(temp_df['trial'], 0, 1, where=temp_df['WM'] >= threshold,
                   facecolor='darkgreen', alpha=0.3)
sns.lineplot(x='trial', y='repeat_choice', data=temp_df, ax=panel, color='black')
panel.set_ylabel('Running RB')
panel.set_xlim(0, max(temp_df.trial.unique()) - 3)
panel.set_xlabel('Trial index')
panel.hlines(xmin=0, xmax=len(temp_df), y=0.5, linestyle=':', color='black')
# Place "Repeating bias" label in open white space (centre-right, below line)
_x_label_g = int(len(temp_df) * 0.6)
_y_label_g = 0.3  # fixed y in white space below the line
panel.text(_x_label_g, _y_label_g, 'Repeating bias', color='black', fontsize=6,
           ha='center', va='bottom')
panel.text(0.02, 0.04, 'Session C38-2021-07-03', transform=panel.transAxes,
           fontsize=5, color='grey', va='bottom')

# Panel f (a) — Posterior p(STM)
panel = a
panel.fill_between(temp_df['trial'], 0, 1, where=temp_df['WM'] <= threshold,
                   facecolor='indigo', alpha=0.3)
panel.fill_between(temp_df['trial'], 0, 1, where=temp_df['WM'] >= threshold,
                   facecolor='darkgreen', alpha=0.3)
sns.lineplot(x='trial', y='WM_roll', data=temp_df, ax=panel, color='black')
panel.set_ylabel('Posterior p(STM)')
panel.set_xlim(0, max(temp_df.trial.unique()) - 3)
panel.set_xlabel('Trial index')
panel.hlines(xmin=0, xmax=len(temp_df), y=0.5, linestyle=':', color='black')
panel.text(0.02, 0.04, 'Session C38-2021-07-03', transform=panel.transAxes,
           fontsize=5, color='grey', va='bottom')

# ---------------------------------------------------------------------------
# Panels q/r (g2/g3) — individual animal model fits
# ---------------------------------------------------------------------------
new_df_real = pd.read_csv(path + 'panel_q_mouse_N11_10s_delay_behavioral_data.csv', index_col=0)
new_df      = pd.read_csv(path + 'panel_q_mouse_N11_10s_delay_HMM_model_predictions.csv', index_col=0)
figureplot(new_df_real, new_df, g2)
g2.text(0.5, 0.97, 'Mouse C10', transform=g2.transAxes,
        fontsize=7, ha='center', va='top')

new_df_real = pd.read_csv(path + 'panel_r_mouse_C37_10s_delay_behavioral_data.csv', index_col=0)
new_df      = pd.read_csv(path + 'panel_r_mouse_C37_10s_delay_HMM_model_predictions.csv', index_col=0)
figureplot(new_df_real, new_df, g3)
g3.text(0.5, 0.97, 'Mouse C37', transform=g3.transAxes,
        fontsize=7, ha='center', va='top')

# Fix blue dotted baselines in q/r to black
for _ax in [g2, g3]:
    for _line in _ax.get_lines():
        if _line.get_linestyle() in (':', '--'):
            _line.set_color('black')

# ---------------------------------------------------------------------------
# Finalise
# ---------------------------------------------------------------------------
sns.despine()
plt.subplots_adjust(left=0.07, bottom=0.07, right=0.97, top=0.97,
                    wspace=1.5, hspace=0.5)
# plt.savefig(save_path+'/Fig 2_model.svg', bbox_inches='tight', dpi=300)
plt.show()
