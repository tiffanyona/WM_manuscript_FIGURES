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
import warnings
warnings.filterwarnings('ignore', 'FigureCanvasAgg is non-interactive')
warnings.filterwarnings('ignore', 'Attempting to set identical low and high xlims')
from scipy.optimize import curve_fit
from matplotlib.lines import Line2D
#Import all needed libraries
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import ROOT, FIGURES_OUT, DATA_DIR
sys.path.insert(0, str(ROOT / 'src'))
from functions import add_stat_annotation, exp_decay, figureplot, compute_window_centered

save_path = str(FIGURES_OUT / 'supp_figures' / 'supp_fig_3_extra_results_model')
analysis_path = str(DATA_DIR/ 'supp_figures' / 'supp_fig_3_extra_results_model')

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
fig = plt.figure(figsize=(17*cm, 10*cm))
gs = gridspec.GridSpec(nrows=3, ncols=8, figure=fig)

a = fig.add_subplot(gs[0, 0:1])
b1 = fig.add_subplot(gs[0, 1:3])
b2 = fig.add_subplot(gs[0, 3:5])
c = fig.add_subplot(gs[0, 6:8])

d1 = fig.add_subplot(gs[1, 0:2])
d2 = fig.add_subplot(gs[1, 2:4])
e1 = fig.add_subplot(gs[1, 4:6])
e2 = fig.add_subplot(gs[1, 6:8])

f = fig.add_subplot(gs[2, 0:2])
g = fig.add_subplot(gs[2, 2:4])
h = fig.add_subplot(gs[2, 4:7])

fig.text(0.01, 0.99, 'a', fontsize=10, fontweight='bold', va='top')
fig.text(0.37, 0.99, 'b', fontsize=10, fontweight='bold', va='top')
fig.text(0.75, 0.99, 'c', fontsize=10, fontweight='bold', va='top')

fig.text(0.01, 0.66, 'd', fontsize=10, fontweight='bold', va='top')
fig.text(0.5, 0.66, 'e', fontsize=10, fontweight='bold', va='top')

fig.text(0.01, 0.35, 'f', fontsize=10, fontweight='bold', va='top')
fig.text(0.25, 0.35, 'g', fontsize=10, fontweight='bold', va='top')
fig.text(0.49, 0.357, 'h', fontsize=10, fontweight='bold', va='top')



## ----------------------- Panel A HMM LL (bits/trial) ---------------------------------

panel = a
file_name = '/pertrialLL'
full_fit = pd.read_csv(analysis_path + file_name + '.csv', index_col=0)
xA = np.random.normal(0, 0.05, len(full_fit))

sns.violinplot(x='model', y='LL/trial', data=full_fit.loc[(full_fit.delay == 10)&(full_fit.model == 'all')], legend=False,
               linewidth=0, alpha=0.4, width=0.5, ax=panel, zorder=1, color='lightgrey')
sns.stripplot(x='model', y='LL/trial', data=full_fit.loc[(full_fit.delay == 10)&(full_fit.model == 'all')], jitter=0.3, size=3,   
              legend=False, edgecolor='none', color='black', linewidth=0, ax=panel, zorder=2)
sns.boxplot(x='model', y='LL/trial', data=full_fit.loc[(full_fit.delay == 10)&(full_fit.model == 'all')], legend=False,
            width=0.12, showcaps=False, showfliers=False, ax=panel,
            boxprops=dict(zorder=4, linewidth=1, color='black'),
            whiskerprops=dict(zorder=4, linewidth=1, color='black'),
            medianprops=dict(color='white', linewidth=1.5, zorder=5))

panel.set_xlabel('')
panel.set_ylabel('LL (bits/trial)')

# ----------------------------------------------------------------------------------------------------------------

# ------------------------### PAnel B - LL difference versus Lapse rate -------------------------
from scipy.stats import linregress

merge_df = pd.read_csv(analysis_path+'/difference_vs_lapse.csv')
plot = merge_df.loc[merge_df.model == '11']

panel = b1
# linear regression
res = linregress(plot['lapse'], plot['substracted'], alternative='two-sided')
slope = res.slope
pval = res.pvalue
sns.scatterplot(
    data=plot, x='lapse', y='substracted', legend=False,
    hue='notHMM', palette=['grey', 'black'], ax=panel, s = 15)

panel.hlines(
    y=0,
    xmin=0.,
    xmax=0.3,
    colors='gray',
    linestyles='dashed'
)

sns.regplot(
    data=plot, x='lapse', y='substracted',
    scatter=False, color='black', ci=None, ax=panel
)

panel.text(
    0.05, 0.95,
    f"slope = {slope:.3g}\np = {pval:.2e}",
    transform=panel.transAxes,
    ha='left', va='top', fontsize=6
)

sns.despine(ax=panel)
panel.set_xlabel('Lapse rate')
panel.set_ylabel('Delta - LL \n(bits/trial)')
panel.set_xlim(0, 0.3)

# -----------------############################## H Panel - HMM parameter values #################################
file_name = '/fit_HMM_selected_final'
full_fit = pd.read_csv(analysis_path+file_name+'.csv', index_col=0)

full_fit['const'] = 1
full_fit = full_fit.loc[full_fit.delay == 10]

for regressor, panel, color in zip(['pi', 't11','t22'],[h, h, h], ['darkgreen','grey','grey']):
    if regressor == 'pi':
        xA = np.random.normal(0, 0.1, len(full_fit))
        sns.scatterplot(x=xA,y=regressor,data=full_fit, ax=panel, color=color, legend=False, alpha=0.7, size=5, zorder=2)

    elif regressor == 't11':
        xA = np.random.normal(1, 0.1, len(full_fit))
        sns.scatterplot(x=xA,y=regressor,data=full_fit, ax=panel, color=color, legend=False,alpha=0.7, size=5, zorder=2)

    elif regressor == 't22':
        xA = np.random.normal(2, 0.1, len(full_fit))
        sns.scatterplot(x=xA,y=regressor,data=full_fit, ax=panel, color=color, legend=False, alpha=0.7, size=5, zorder=2)

    if regressor == 'alfa':
        panel.set_xlabel('α')
        panel.set_ylim(1.5,3.1)
    elif regressor == 'mu_b':
        panel.set_xlabel('$m_b$')
    elif regressor == 'beta_w':
        panel.set_xlabel('$β_a$')
    elif regressor == 'beta_bias':
        panel.set_xlabel('$β_{bias}$')
    elif regressor == 'P_L' or regressor == 'P_R' or regressor == 'WM' or regressor == 'RL':
        panel.set_ylim(-0.1,1.1)
        panel.set_xlim(-0.5,1.5)

    panel.hlines(y=0,xmin=-1,xmax=2.5,linestyle=':', color='black')
    panel.locator_params(axis='y', nbins=5)
    y_min, y_max = panel.get_ylim()
    panel.locator_params(nbins=3)

plot = pd.DataFrame({'pi': full_fit['pi'], 't11': full_fit['t11'], 't22': full_fit['t22']})
sns.violinplot(data=plot, palette=['darkgreen','grey','grey'], ax=panel, width=0.5, alpha=0.5, inner='box', linewidth=0, zorder=1)
sns.violinplot(data=plot, palette=['darkgreen','grey','grey'], ax=panel, width=0.5, fill=False, zorder=0)
# panel.set_xlabel(r'$\pi \quad t_{11} \quad t_{22}$')
panel.set_xlim(-0.7, 2.7)

# ----------------------------------------------------------------------------------------------------------------
# -----------------############################## B Panel - X-Y #################################
panel = b2

df_results = pd.read_csv(analysis_path+'/X-Y.csv')
cmap = sns.diverging_palette(15, 250, s=100, l=60, n=len(df_results.loc[df_results.streak != 0].streak.unique()), center="dark")

sns.lineplot(x='delays', y='accuracy_model', hue='streak', data=df_results,errorbar=('ci', 67), legend=False, ax=panel, palette=cmap)
sns.lineplot(x='delays', y='accuracy_data', hue='streak', data=df_results, errorbar=('ci', 67), linestyle='', ax=panel,
             markeredgewidth=0.2, marker='o', err_style="bars", legend=False, palette=cmap)

panel.hlines(xmin=0, xmax=10, y=0.5, linestyles=':', color='black')
panel.set_ylim(0.4, 1)
panel.set_xlabel("Delay (s)")
panel.set_ylabel("Accuracy")
panel.locator_params(nbins=3)

# Legend for repetition and alternation plot
legend_elements = [Line2D([0], [0], color=cmap[-1], label='XXX'),
                   Line2D([0], [0], color=cmap[-2], label='XX'),
                   Line2D([0], [0], color=cmap[-3], label='X'),
                   Line2D([0], [0], color=cmap[2], label='Y'),
                   Line2D([0], [0], color=cmap[1], label='YY'),
                   Line2D([0], [0], color=cmap[0], label='YYY')]
panel.legend(handles=legend_elements, ncol=1, fontsize=6, bbox_to_anchor=(1, 1), borderaxespad=0).get_frame().set_linewidth(0.0)

# ----------------------------------------------------------------------------------------------------------------
# -----------------############################## C Panel - figureplot #################################
animal = '/N27_10'
new_df_real = pd.read_csv(analysis_path + animal + '_data.csv')
new_df = pd.read_csv(analysis_path + animal + '_model.csv')
figureplot(new_df_real, new_df, c)

# ----------------------------------------------------------------------------------------------------------------
# -----------------############################## D Panel - After correct #################################
panel = d1
df_results = pd.read_csv(analysis_path + '/after_correct.csv')
sns.lineplot(x='delays', y='accuracy_data', hue='after_correct',
             data=df_results.loc[df_results.after_correct != -1],
             markeredgewidth=0.2, ax=panel, marker='o', palette=['crimson', 'darkgreen'],
             linestyle='', err_style="bars", legend=False)
sns.lineplot(x='delays', y='accuracy_model', hue='after_correct',
             data=df_results.loc[df_results.after_correct != -1],
             ax=panel, marker='', palette=['crimson', 'darkgreen'], legend=False)
panel.hlines(xmin=0, xmax=10, y=0.5, linestyles=':')
panel.set_xlabel("Delay (s)")
panel.set_ylim(0.45, 1)
panel.set_ylabel("Accuracy")

panel = d2
sns.lineplot(x='delays', y='repeat_data', data=df_results, marker='o', hue='after_correct',
             palette=['crimson', 'darkgreen'], markeredgewidth=0.2, ax=panel, legend=False,
             err_style="bars", linestyle='')
sns.lineplot(x='delays', y='repeat_model', data=df_results, hue='after_correct',
             palette=['crimson', 'darkgreen'], ax=panel, legend=False)
panel.hlines(xmin=0, xmax=10, y=0.5, linestyles=':')
panel.set_ylim(0.45, 1)
legend_elements = [Line2D([0], [0], color='crimson', label='After incorrect'),
                   Line2D([0], [0], color='darkgreen', label='After correct')]
panel.legend(handles=legend_elements, fontsize=6, ncol=1).get_frame().set_linewidth(0.0)
panel.set_xlabel("Delay (s)")
panel.set_ylabel("Repeating bias")
panel.locator_params(nbins=3)

# ----------------------------------------------------------------------------------------------------------------
# -----------------############################## E Panel - Previous correct #################################
panel = e1
df_results = pd.read_csv(analysis_path + '/previous_correct.csv')
sns.lineplot(x='delays', y='accuracy_data', hue='previous_correct',
             data=df_results.loc[df_results.previous_correct != -1],
             markeredgewidth=0.2, ax=panel, marker='o', palette=['crimson', 'darkgreen'],
             linestyle='', err_style="bars", legend=False)
sns.lineplot(x='delays', y='accuracy_model', hue='previous_correct',
             data=df_results.loc[df_results.previous_correct != -1],
             ax=panel, marker='', palette=['crimson', 'darkgreen'], legend=False)
panel.hlines(xmin=0, xmax=10, y=0.5, linestyles=':')
panel.set_xlabel("Delay (s)")
panel.set_ylim(0.45, 1)
panel.set_ylabel("Accuracy")

panel = e2
sns.lineplot(x='delays', y='repeat_data', data=df_results, marker='o', hue='previous_correct',
             palette=['crimson', 'darkgreen'], markeredgewidth=0.2, ax=panel, legend=False,
             err_style="bars", linestyle='')
sns.lineplot(x='delays', y='repeat_model', data=df_results, hue='previous_correct',
             palette=['crimson', 'darkgreen'], ax=panel, legend=False)
panel.hlines(xmin=0, xmax=10, y=0.5, linestyles=':')
panel.set_ylim(0.45, 1)
legend_elements = [Line2D([0], [0], color='crimson', label='Before incorrect trial'),
                   Line2D([0], [0], color='darkgreen', label='Before correct trial')]
panel.legend(handles=legend_elements, fontsize=6, ncol=1).get_frame().set_linewidth(0.0)
panel.set_xlabel("Delay (s)")
panel.set_ylabel("Repeating bias")
panel.locator_params(nbins=3)

# ----------------------------------------------------------------------------------------------------------------
# -----------------############################## F Panel - Hit autocorrelation #################################
cumulative_autocorrelation_hit_model = pd.read_csv(analysis_path+'\\hit_autocorrelation_10_model_V2.csv', index_col=0)
cumulative_autocorrelation_repeat_model = pd.read_csv(analysis_path+'\\repeat_autocorrelation_10_model_V2.csv', index_col=0)

cumulative_autocorrelation_hit_data = pd.read_csv(analysis_path+'\\hit_autocorrelation.csv', index_col=0)
cumulative_autocorrelation_repeat_data = pd.read_csv(analysis_path+'\\repeat_autocorrelation.csv', index_col=0)

panel = f
color = 'darkgreen'
corr = cumulative_autocorrelation_hit_model[:25].mean(axis=1)

panel.plot(np.arange(1,len(corr)+1), corr, marker='.', linestyle='', color='grey')
params, cov = curve_fit(exp_decay, np.arange(len(corr)), corr.values)
print(params[1])

corr = cumulative_autocorrelation_hit_data[:25]
df_lower = pd.DataFrame()
df_upper = pd.DataFrame()

for timepoint in range(len(corr)):
    mean_surr = []
    array = corr.iloc[timepoint].to_numpy()
    array = array[~np.isnan(array)]
    for iteration in range(1000):
        x = np.random.choice(array, size=len(array), replace=True)
        mean_surr.append(np.mean(x))
    df_lower.at[0, timepoint] = np.percentile(mean_surr, 2.5)
    df_upper.at[0, timepoint] = np.percentile(mean_surr, 97.5)

lower = df_lower.iloc[0].values[:25]
upper = df_upper.iloc[0].values[:25]

mean = corr.mean(axis=1)[:25]
lower = corr.quantile(q=0.025, axis=1, numeric_only=True)
upper = corr.quantile(q=0.975, axis=1, numeric_only=True)

panel.plot(np.arange(1,len(corr)+1), mean, marker='', color=color)
panel.hlines(y=0, xmin=0, xmax=25, linestyles=':')
panel.set_xlabel('Trial indexes')
panel.set_xlim(0, 25)
panel.text(10, 0.08, f"HMM", ha='left', va='top', fontsize=6, color='grey')
panel.text(10, 0.07, f"Data", ha='left', va='top', fontsize=6, color='darkgreen')

params, cov = curve_fit(exp_decay, np.arange(len(corr.mean(axis=1))), corr.mean(axis=1).values)
print(params[1])

# ----------------------------------------------------------------------------------------------------------------
# -----------------############################## G Panel - Repeat autocorrelation #################################
panel = g
color = 'indigo'
corr = cumulative_autocorrelation_repeat_model.mean(axis=1)

panel.plot(np.arange(1,len(corr)+1), corr, marker='.', linestyle='', color='grey')
params, cov = curve_fit(exp_decay, np.arange(len(corr)), corr.values)
print(params[1])

corr = cumulative_autocorrelation_repeat_data[:25]
df_lower = pd.DataFrame()
df_upper = pd.DataFrame()

for timepoint in range(len(corr)):
    mean_surr = []
    array = corr.iloc[timepoint].to_numpy()
    array = array[~np.isnan(array)]
    for iteration in range(1000):
        x = np.random.choice(array, size=len(array), replace=True)
        mean_surr.append(np.mean(x))
    df_lower.at[0, timepoint] = np.percentile(mean_surr, 2.5)
    df_upper.at[0, timepoint] = np.percentile(mean_surr, 97.5)

lower = df_lower.iloc[0].values[:25]
upper = df_upper.iloc[0].values[:25]

mean = corr.mean(axis=1)[:25]
lower = corr.quantile(q=0.025, axis=1, numeric_only=True)
upper = corr.quantile(q=0.975, axis=1, numeric_only=True)

panel.plot(np.arange(1,len(corr)+1), mean, marker='', color=color)
panel.hlines(y=0, xmin=0, xmax=25, linestyles=':', color='black')
panel.set_xlabel('Trial indexes')
panel.set_xlim(0, 25)
panel.text(10, 0.08, f"HMM", ha='left', va='top', fontsize=6, color='grey')
panel.text(10, 0.07, f"Data", ha='left', va='top', fontsize=6, color='indigo')

params, cov = curve_fit(exp_decay, np.arange(len(corr.mean(axis=1))), corr.mean(axis=1).values)
print(params[1])

# ----------------------------------------------------------------------------------------------------------------
# Show the figure
sns.despine()
plt.subplots_adjust(left=0.07,
                    bottom=0.07,
                    right=0.97,
                    top=0.97,
                    wspace=1.5,
                    hspace=0.5)

# plt.savefig(save_path+'/Fig. Supp. 3. Extra results from model_v4.svg', bbox_inches='tight',dpi=300)
plt.show()