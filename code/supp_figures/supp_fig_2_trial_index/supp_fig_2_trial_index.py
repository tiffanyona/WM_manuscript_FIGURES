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
from functions import compute_window, trials_normalized, trials_label

data_path = str(DATA_DIR / 'supp_figures' / 'supp_fig_2_trial_index') + '/'
path = data_path
os.chdir(path)
save_path = str(FIGURES_OUT / 'supp_figures' / 'supp_fig_2_trial_index') + '/'

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

fig = plt.figure(figsize=(17*cm, 15*cm))
gs = gridspec.GridSpec(nrows=4, ncols=8, figure=fig)

a = fig.add_subplot(gs[0, 0:4])
e = fig.add_subplot(gs[0, 4:8])

c = fig.add_subplot(gs[1, 0:4])
d = fig.add_subplot(gs[1, 4:8])

h = fig.add_subplot(gs[2, 4:6])
i = fig.add_subplot(gs[2, 6:8])
j = fig.add_subplot(gs[3, 4:6])
k = fig.add_subplot(gs[3, 6:8])

fig.text(0.01, 1,    'a', fontsize=10, fontweight='bold', va='top')
fig.text(0.5,  1,    'b', fontsize=10, fontweight='bold', va='top')
fig.text(0.01, 0.75, 'c', fontsize=10, fontweight='bold', va='top')
fig.text(0.5,  0.75, 'd', fontsize=10, fontweight='bold', va='top')
fig.text(0.01, 0.5,  'e', fontsize=10, fontweight='bold', va='top')
fig.text(0.5,  0.5,  'f', fontsize=10, fontweight='bold', va='top')
fig.text(0.75, 0.5,  'g', fontsize=10, fontweight='bold', va='top')
fig.text(0.5,  0.25, 'h', fontsize=10, fontweight='bold', va='top')
fig.text(0.75, 0.25, 'i', fontsize=10, fontweight='bold', va='top')

# ── Load behavioral data ──────────────────────────────────────────────────────
file_name = 'global_behavior_10_paper'
df = pd.read_csv(data_path + file_name + '.csv', index_col=0)

df['T'] = df.apply(trials_normalized, axis=1)
df['trial_label'] = df.apply(trials_label, axis=1)
df['running_accuracy'] = compute_window(df, 20, 'hit')
df['running_repeat'] = compute_window(df, 20, 'repeat_choice')

df = df[df.valids == 1]

# ── A Panel ───────────────────────────────────────────────────────────────────
df_results = pd.DataFrame()
df_results['accuracy'] = df.groupby(['subject', 'T', 'trial_label'])['hit'].mean()
df_results.reset_index(inplace=True)

panel = a
sns.lineplot(x='T', y='accuracy', hue='trial_label', ax=panel,
             data=df_results[(df_results['T'] >= 0.1) & (df_results['T'] != 1)],
             ci=95, palette=['lightgrey', 'dimgrey'], legend=False,
             hue_order=['Early', 'Late'])
panel.set_ylim(0.4, 1)
panel.set_xlabel('Normalized trial index')
panel.set_ylabel('Accuracy')
panel.hlines(y=0.5, xmin=0.1, xmax=1, linestyle=':')

# bracket comparing Early vs Late above the plot
y_bracket = 0.98
x1, x2 = 0.1, 0.75
panel.annotate('', xy=(x2, y_bracket), xytext=(x1, y_bracket),
               xycoords='data', textcoords='data',
               arrowprops=dict(arrowstyle='-', color='black', lw=0.8),
               annotation_clip=False)
# left tick
panel.annotate('', xy=(x1, y_bracket - 0.01), xytext=(x1, y_bracket),
               xycoords='data', textcoords='data',
               arrowprops=dict(arrowstyle='-', color='black', lw=0.8),
               annotation_clip=False)
# right tick
panel.annotate('', xy=(x2, y_bracket - 0.01), xytext=(x2, y_bracket),
               xycoords='data', textcoords='data',
               arrowprops=dict(arrowstyle='-', color='black', lw=0.8),
               annotation_clip=False)
panel.text((x1 + x2) / 2, y_bracket + 0.01, '***', ha='center', va='bottom', fontsize=7)

# ── E Panel (repeating bias) ───────────────────────────────────────────────────
grouping = ['subject', 'T', 'hit', 'trial_label']
df_results_rep = pd.DataFrame()
df_results_rep['repeat_choice'] = (
    df.loc[(df['repeat_choice_side'] == 1)].groupby(grouping)['valids'].count() /
    df.loc[(df.vector_answer == 0)].groupby(grouping)['valids'].count() +
    df.loc[(df['repeat_choice_side'] == 2)].groupby(grouping)['valids'].count() /
    df.loc[(df.vector_answer == 1)].groupby(grouping)['valids'].count()
) / 2
df_results_rep.reset_index(inplace=True)

panel = e
sns.lineplot(x='T', y='repeat_choice', hue='trial_label',
             data=df_results_rep[(df_results_rep['T'] >= 0.1) & (df_results_rep['T'] != 1)],
             ci=95, ax=panel, palette=['lightgrey', 'dimgrey'], legend=False,
             hue_order=['Early', 'Late'])
panel.set_ylim(0.4, 0.8)
panel.set_xlabel('Normalized trial index')
panel.set_ylabel('Repeating bias')
panel.hlines(y=0.5, xmin=0.1, xmax=1, linestyle=':')

# bracket comparing Early vs Late above the plot
y_bracket = 0.81
x1, x2 = 0.1, 0.95
panel.annotate('', xy=(x2, y_bracket), xytext=(x1, y_bracket),
               xycoords='data', textcoords='data',
               arrowprops=dict(arrowstyle='-', color='black', lw=0.8),
               annotation_clip=False)
# left tick
panel.annotate('', xy=(x1, y_bracket - 0.01), xytext=(x1, y_bracket),
               xycoords='data', textcoords='data',
               arrowprops=dict(arrowstyle='-', color='black', lw=0.8),
               annotation_clip=False)
# right tick
panel.annotate('', xy=(x2, y_bracket - 0.01), xytext=(x2, y_bracket),
               xycoords='data', textcoords='data',
               arrowprops=dict(arrowstyle='-', color='black', lw=0.8),
               annotation_clip=False)
panel.text((x1 + x2) / 2, y_bracket + 0.005, '***', ha='center', va='bottom', fontsize=7)

# ── C Panel (accuracy by delay — Blues palette) ───────────────────────────────
df_results_c = pd.DataFrame()
df_results_c['accuracy'] = df.groupby(['subject', 'T', 'delay_times'])['hit'].mean()
df_results_c.reset_index(inplace=True)

panel = c
sns.lineplot(x='T', y='accuracy', hue='delay_times', ax=panel,
             palette='Blues', legend=False,
             data=df_results_c[(df_results_c['T'] >= 0.1) & (df_results_c['T'] != 1)],
             ci=None)
panel.set_ylim(0.4, 1)
panel.set_xlabel('Normalized trial index')
panel.set_ylabel('Accuracy')
panel.hlines(y=0.5, xmin=0.1, xmax=1, linestyle=':')

# legend for panel c
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color=sns.color_palette('Blues', 4)[0], label='0s delay'),
    Line2D([0], [0], color=sns.color_palette('Blues', 4)[1], label='1s delay'),
    Line2D([0], [0], color=sns.color_palette('Blues', 4)[2], label='3s delay'),
    Line2D([0], [0], color=sns.color_palette('Blues', 4)[3], label='10s delay'),
]
panel.legend(handles=legend_elements, fontsize=5, frameon=False)

# ── D Panel (violin by trial segment — colored by delay) ──────────────────────
df_results_d = pd.DataFrame()
df_results_d['accuracy'] = df.groupby(['subject', 'trial_label', 'delay_times'])['hit'].mean()
df_results_d.reset_index(inplace=True)
df_results_d['trial_label'] = pd.Categorical(df_results_d['trial_label'],
                                              categories=['Early', 'Late'], ordered=True)

panel = d
sns.violinplot(x='trial_label', y='accuracy', hue='delay_times', ax=panel,
               legend=False, data=df_results_d, inner=None,
               palette='Blues', saturation=0.4, order=['Early', 'Late'], linewidth=0, zorder=1)
sns.boxplot(x='trial_label', y='accuracy', hue='delay_times', data=df_results_d,
            order=['Early', 'Late'], ax=panel,
            width=0.1, showcaps=False, showfliers=False, legend=False,
            boxprops=dict(zorder=4, linewidth=1, alpha=0.5),
            whiskerprops=dict(zorder=4, linewidth=1), color='black',
            medianprops=dict(color='white', linewidth=1.5, zorder=5))

panel.set_ylim(0.4, 1)
panel.set_xlim(-0.5, 1.5)
panel.hlines(y=0.5, xmin=-0.5, xmax=1.5, linestyle=':')
panel.set_xticks([0, 1], ['Early', 'Late'])
panel.set_xlabel('Trial segment')
panel.set_ylabel('Accuracy')
plt.legend(handles='', ncol=2).get_frame().set_linewidth(0.0)

# ── F, G, H, I Panels (GLM weights) ──────────────────────────────────────────
file_name = 'GLMM_final_data'
coef_matrix = pd.read_csv(path + file_name + '.csv', index_col=0)
coef_matrix['const'] = 0

# Original order: f=S·T, g=S·D·T, h=C·T, i=C·D·T
for regressor, panel in zip(['SL:T', 'SR:T', 'SL:D:T', 'SR:D:T', 'exp_C:T', 'D:exp_C:T'],
                             [h,      h,       i,        i,         j,          k]):

    if regressor in ('SR', 'SR:D', 'SR:D:T', 'SR:T'):
        main = COLORRIGHT
        light = 'grey'
    elif regressor in ('exp_C', 'D:exp_C', 'exp_C:T', 'D:exp_C:T'):
        main = 'indigo'
        light = 'grey'
    else:
        main = COLORLEFT
        light = 'grey'

    if regressor == 'SR:T':
        plot = pd.DataFrame({'SL:T': coef_matrix['SL:T'], 'SR:T': coef_matrix['SR:T']})
        sns.violinplot(data=plot, palette=[COLORLEFT, COLORRIGHT], ax=panel,
                       width=0.5, saturation=0.4, linewidth=0, inner=None, zorder=1)
        sns.boxplot(data=plot, ax=panel,
                    width=0.15, showcaps=False, showfliers=False,
                    boxprops=dict(zorder=4, linewidth=1, alpha=0.5),
                    whiskerprops=dict(zorder=4, linewidth=1), color='black',
                    medianprops=dict(color='white', linewidth=1.5, zorder=5))
        xA = np.random.normal(1, 0.1, len(coef_matrix))
        const = 1
        panel.set_xlabel('Left          Right')
        panel.set_title('Stimulus x Trial $S_t·T_t$', fontsize=7)

    elif regressor == 'SR:D:T':
        plot = pd.DataFrame({'SL:D:T': coef_matrix['SL:D:T'], 'SR:D:T': coef_matrix['SR:D:T']})
        sns.violinplot(data=plot, palette=[COLORLEFT, COLORRIGHT], ax=panel,
                       width=0.5, saturation=0.4, linewidth=0, inner=None, zorder=1)
        sns.boxplot(data=plot, ax=panel,
                    width=0.15, showcaps=False, showfliers=False,
                    boxprops=dict(zorder=4, linewidth=1, alpha=0.5),
                    whiskerprops=dict(zorder=4, linewidth=1), color='black',
                    medianprops=dict(color='white', linewidth=1.5, zorder=5))
        xA = np.random.normal(1, 0.1, len(coef_matrix))
        const = 1
        panel.set_xlabel('Left            Right')
        panel.set_title('Stimulus x Delay\nx Trial $S_t·D_t·T_t$', fontsize=7)

    elif regressor in ('SL:T', 'SL:D:T'):
        main = COLORLEFT
        light = 'grey'
        xA = np.random.normal(0, 0.1, len(coef_matrix))
        const = 0

    else:
        const = 0
        xA = np.random.normal(0, 0.15, len(coef_matrix))
        sns.violinplot(x=coef_matrix['const'].astype(float),
                       y=coef_matrix[regressor].astype(float),
                       data=coef_matrix, width=0.75, color=main, ax=panel,
                       saturation=0.4, linewidth=0, inner=None, zorder=1)
        sns.boxplot(x=coef_matrix['const'].astype(float),
                    y=coef_matrix[regressor].astype(float),
                    ax=panel, width=0.15, showcaps=False, showfliers=False,
                    boxprops=dict(zorder=4, linewidth=1, alpha=0.5),
                    whiskerprops=dict(zorder=4, linewidth=1), color='black',
                    medianprops=dict(color='white', linewidth=1.5, zorder=5))
        main = 'indigo'
        if regressor == 'exp_C:T':
            panel.set_title('Prev. Choices x Trial\n $C_{t-k}·T_t$', fontsize=7)
        if regressor == 'D:exp_C:T':
            panel.set_title('Prev. Choices x Delay x Trial\n$C_{t-k}·D_t·T_t$', fontsize=7)
        panel.set_xlabel('')

    panel.hlines(y=0, xmin=-2, xmax=2, linestyle=':')

    try:
        sns.scatterplot(x=xA, y=regressor, data=coef_matrix,
                        hue=coef_matrix[regressor + '_sig'],
                        palette=[light, main], ax=panel, alpha=0.9, legend=False)
    except Exception:
        if coef_matrix[regressor + '_sig'].all() == 1:
            sns.scatterplot(x=xA, y=regressor, data=coef_matrix,
                            color=main, ax=panel, alpha=0.9, legend=False)
        else:
            sns.scatterplot(x=xA, y=regressor, data=coef_matrix,
                            color='grey', ax=panel, alpha=0.9, legend=False)

    panel.set_xticks([])
    panel.set(ylabel=None)
    if regressor in ('SR', 'SR:D', 'SR:T', 'SR:D:T'):
        panel.set_xlim(-0.5, 1.5)
    else:
        panel.set_xlim(-1.5, 1.5)

    if regressor == 'D:exp_C:T':
        panel.set_ylim(-0.1, 0.1)
    panel.set_ylabel('Weights')
    panel.locator_params(axis='y', nbins=3)
    y_min, y_max = panel.get_ylim()

    if stats.ttest_1samp(coef_matrix[regressor], 0)[1] <= 0.001:
        panel.text(const - 0.13, y_max, '***', fontsize=6)
    elif stats.ttest_1samp(coef_matrix[regressor], 0)[1] <= 0.01:
        panel.text(const - 0.1, y_max, '**')
    elif stats.ttest_1samp(coef_matrix[regressor], 0)[1] <= 0.05:
        panel.text(const - 0.05, y_max, '*')
    else:
        panel.text(const - 0.1, y_max, 'ns', fontsize=6)

    plt.gca().tick_params(direction='out')

h.set_ylabel('')

# ── Finalize ──────────────────────────────────────────────────────────────────
sns.despine()
plt.subplots_adjust(left=0.07, bottom=0.07, right=0.97, top=0.97,
                    wspace=2.1, hspace=0.7)
plt.show()