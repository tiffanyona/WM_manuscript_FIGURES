# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 12:06:44 2022

@author: Tiffany
"""
COLORLEFT = 'teal'
COLORRIGHT = '#FF8D3F'

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import ROOT, FIGURES_OUT, DATA_DIR
sys.path.insert(0, str(ROOT / 'src'))
import functions as plots
from functions import add_stat_annotation, convolveandplot, new_convolve, plotsingledelay

path = str(DATA_DIR / 'supp_figures' / 'supp_fig_17_delay_code_STM_RepL')
save_path = str(FIGURES_OUT / 'supp_figures' / 'supp_fig_17_delay_code_STM_RepL')
Path(save_path).mkdir(parents=True, exist_ok=True)

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
fig = plt.figure(figsize=(14*cm, 16*cm), layout='constrained')
fig.get_layout_engine().set(wspace=0.01, hspace=0.05)
gs = gridspec.GridSpec(nrows=3, ncols=8, figure=fig)

a = fig.add_subplot(gs[0, 0:2])
b = fig.add_subplot(gs[0, 2:6])
c = fig.add_subplot(gs[0, 6:8])
d = fig.add_subplot(gs[1, 0:2])
e = fig.add_subplot(gs[1, 2:6])
f = fig.add_subplot(gs[2, 0:2])
f_res = fig.add_subplot(gs[2, 2:4])
g = fig.add_subplot(gs[2, 4:6])
g_res = fig.add_subplot(gs[2, 6:8])

fig.text(0.01, 1,    'a', fontsize=10, fontweight='bold', va='top')
fig.text(0.25, 1,   'b', fontsize=10, fontweight='bold', va='top')
fig.text(0.75, 1,   'c', fontsize=10, fontweight='bold', va='top')
fig.text(0.01, 0.67,'d', fontsize=10, fontweight='bold', va='top')
fig.text(0.25, 0.67,'e', fontsize=10, fontweight='bold', va='top')
fig.text(0.01, 0.33,'f', fontsize=10, fontweight='bold', va='top')
fig.text(0.5,  0.33,'g', fontsize=10, fontweight='bold', va='top')

# #######################################################################################
# panel b: single delay
# #######################################################################################

file_name = r'\single_delay_WM_roll0.6_delay_0.25_all_V3'
df_cum_sti = pd.read_csv(path+file_name+'.csv', index_col=0)

list_sessions = df_cum_sti.session.unique()

scores = df_cum_sti.groupby('session').score.mean().reset_index()
list_exclude = scores.loc[scores.score<0.55].session.unique()
df_cum_sti = df_cum_sti[~df_cum_sti['session'].isin(list_exclude)]

colors=['crimson', 'pink']
variables = ['WM_roll','RL_roll']
hits = [0,0]
ratios = [0.6,0.4]
variables_combined=[variables[0]+'_'+str(hits[0]),variables[1]+'_'+str(hits[1])]

panel=b
delay=10
plotsingledelay(df_cum_sti, panel, colors, variables_combined, delay, invert_list=[False, False, False, False], baseline=0)
panel.locator_params(axis='y', nbins=4)
panel.set_ylim(-0.1,0.12)
for _coll in list(b.collections):
    if _coll.get_alpha() is not None and _coll.get_alpha() < 0.5:
        _coll.remove()

# Linear fit from 0–10 s, one per curve, in the matching color
for _color, _var in zip(colors, variables_combined):
    _df = df_cum_sti.loc[(df_cum_sti['trial_type'] == _var) & (df_cum_sti['delay'] == delay)]
    _num_cols = _df.columns[_df.columns.to_series().apply(pd.to_numeric, errors='coerce').notna()]
    _mean = np.array(_df.groupby('session').mean(numeric_only=True)[_num_cols].mean())
    _t = _num_cols.astype(float)
    _mask = (_t >= 0.5) & (_t <= 10.25)
    _sl, _ic, _, _, _ = stats.linregress(_t[_mask], _mean[_mask])
    b.plot(_t[_mask], _sl * _t[_mask] + _ic, color=_color, linewidth=1)


# ------######################################## commented out ###################################################-----------

# file_name = 'log_odds_roll0.25_RL1_4bins_summary'

# df_final = pd.read_csv(path+file_name+'.csv', index_col=0)

# scores = df_final.groupby('session').score.mean().reset_index()
# list_exclude = scores.loc[scores.score<0.55].session.unique()
# df_final = df_final[~df_final['session'].isin(list_exclude)]

# # df_results = df_final.loc[(df_final.trial_type=='WM_roll_1')|(df_final.trial_type=='WM_roll_0')].groupby(['session', 'trial_type','epoch']).logs.mean()
# # df_results = df_results.reset_index()

# # plot = pd.DataFrame({'Correct WM early': df_results.loc[(df_results.trial_type == 'WM_roll_1')&(df_results.epoch == 'early')].logs.values,
# #                      'Incorrect WM early':  df_results.loc[(df_results.trial_type == 'WM_roll_0')&(df_results.epoch == 'early')].logs.values,
# #                     'Correct WM late': df_results.loc[(df_results.trial_type == 'WM_roll_1')&(df_results.epoch == 'late')].logs.values,
# #                     'Incorrect WM late':  df_results.loc[(df_results.trial_type == 'WM_roll_0')&(df_results.epoch == 'late')].logs.values})

# panel = d1
# # sns.violinplot(data=plot, palette=['darkgreen', 'crimson', 'indigo', 'purple' ], width=1,saturation=0.6,linewidth=0, ax=panel)
# # sns.violinplot(data=plot, palette=['darkgreen', 'crimson', 'indigo', 'purple' ], width=1,linewidth=1, ax=panel)

# df_results = df_final.groupby(['session', 'trial_type','epoch']).logs.mean()
# df_results = df_results.reset_index()

# sns.boxplot(x='trial_type', y='logs',hue='epoch', order=['WM_roll_0','RL_roll_0'],palette=['crimson','pink'] ,ax = panel, data=df_results, showmeans=True)

# panel.set_ylim(-2.2,2.5)
# panel.hlines(xmin=-0.5, xmax=3.5, y=0, linestyle=':')

# '''
# Against zero early:
# WM incorrec:        Ttest_1sampResult(statistic=-2.7477504654505056, pvalue=0.011462877761353584)
# RL incorrec:        Ttest_1sampResult(statistic=-0.06301202488782576, pvalue=0.9503824394776591)

#     late

# WM incorrec:        Ttest_1sampResult(statistic=3.863943233938716, pvalue=0.0006336180802942003)
# RL incorrec:        Ttest_1sampResult(statistic=3.1384435804116797, pvalue=0.0041947215329948264)

# Mixed models: formula="logs ~ state*epoch + (state+epoch+1|session) + (state+epoch+1|fold)"
#                   Estimate Std. Error  t value  Chisq Df Pr(>Chisq)

# (Intercept)     0.4762594  0.1684837 2.826739

# state           0.2313577  0.1954282 1.183850   10.8273  1   0.001000 **

# epochlate       0.3980418  0.1646347 2.417728  21.2205  1  4.094e-06 ***

# state:epochlate 0.3968953  0.1294036 3.067110  9.4072  1   0.002161 **

# '''

# #######################################################################################
# panel c: slope comparison
# #######################################################################################

file_name = r'\slope_comparison_paired_sessions'
df = pd.read_csv(path+file_name+'.csv', index_col=0)

list_sessions = df.session.unique()
# df = df.loc[df['session'].isin(list_sessions)]

df = df.loc[df.session != 'E13_2021-05-25_16-26-57.csv']

panel = c

melted_data = pd.melt(df)
palette = sns.color_palette(['black'], len(df.session.unique()))

sns.boxplot(data=df, x='variable', y='slope', palette=['crimson', 'pink'], ax=panel,
            showfliers=False, showcaps=False, linewidth=0,
            medianprops=dict(color='white', linewidth=1.5),
            whiskerprops=dict(color='black', linewidth=0.7))
sns.lineplot(x='variable', y='slope', hue='session', palette=palette,legend=False,data=df, linewidth=0.2,ax = panel)

panel.set_xticks([0,1],['WM','RepL'])
panel.legend(loc='lower right', ncol=2)
add_stat_annotation(panel, data=df, x='variable', y='slope',
                    box_pairs=[( 'WM_roll_0','RL_roll_0')],
                    test='t-test_paired', text_format='star', loc='inside', line_offset_to_box=0.05, text_offset=-0.5, line_offset=0, verbose=1, fontsize=6, linewidth=0.5)

# Perform one-sample t-test against 0
t_statistic, p_value = stats.ttest_1samp(df.loc[df.variable == 'RL_roll_0']['slope'].values, 0)

# Print results
print("One-sample t-test results against 0:")
print("t-statistic:", t_statistic)
print("p-value:", p_value)

t_statistic, p_value = stats.ttest_1samp(df.loc[df.variable == 'WM_roll_0']['slope'].values, 0)

# Print results
print("One-sample t-test results against 0:")
print("t-statistic:", t_statistic)
print("p-value:", p_value)

# plt.plot([1, 0], df.T.values, color='black', alpha=0.3, marker='o')

panel.locator_params(axis='y', nbins=4)
panel.set_xlabel('')
panel.set_ylabel('Slope')

# #######################################################################################
# panels d & e: RL→RL cross-decoder
# #######################################################################################

variable = 'RL_roll_1'
file_name = '/trainedRLcorrect_testedRLcorrect_V2'

df_cum_res = pd.read_csv(path+file_name+'_res.csv', index_col=0)
df_cum_sti = pd.read_csv(path+file_name+'_sti.csv', index_col=0)

df_cum_sti = df_cum_sti.loc[df_cum_sti.trial_type == variable]
df_cum_res = df_cum_res.loc[df_cum_res.trial_type == variable]

trained_trials = df_cum_sti.session.unique()
df_cum_res = df_cum_res[df_cum_res['session'].isin(trained_trials)]

file_name = '/trainedRLcorrect_testedRLcorrect_V1_session_shuffle'
df_cum_res_shuffle = pd.read_csv(path+file_name+'_res.csv', index_col=0)
df_cum_sti_shuffle = pd.read_csv(path+file_name+'_sti.csv', index_col=0)

df_cum_sti_shuffle['trial_type'] = 'RL_roll_1'
df_cum_res_shuffle['trial_type'] = 'RL_roll_1'

trained_trials = df_cum_sti_shuffle.session.unique()
df_cum_sti = df_cum_sti[df_cum_sti['session'].isin(trained_trials)]

y_range = [-0.10, 0.3]

plots.plot_results_session_summary(fig, d, df_cum_sti, ['indigo'], [variable], shuffle_df=df_cum_sti_shuffle,
                                  y_range=y_range, x_range=[-2,1.125], baseline=0)

plots.plot_results_session_summary(fig, e, df_cum_res, ['indigo'], [variable], shuffle_df=df_cum_res_shuffle,
                                  y_range=y_range, x_range=[-1,4], baseline=0)

# e.locator_params(axis='x', nbins=6)
d.locator_params(axis='x', nbins=4)
e.locator_params(axis='y', nbins=4)
d.locator_params(axis='y', nbins=4)
d.set_ylabel('Excess decoding accuracy')

# #######################################################################################
# panel a: variance of log odds during delay
# #######################################################################################

df_var = pd.read_csv(path + r'\decoder_results_log_odds_100r_withRL_L2_v2.csv', index_col=0)
df_var['trial_type'] = df_var['trial_type'].replace({'WM_roll': 'WM_roll_1', 'RL_roll': 'RL_roll_1'})

trial_order_var = ['WM_roll_1', 'RL_roll_1']
df_period_var = df_var.loc[
    (df_var['delay'] == 10) & (df_var['times'] >= 0) & (df_var['times'] <= 10)
].copy()

data = (df_period_var.loc[df_period_var.trial_type.isin(trial_order_var)]
            .groupby(['session', 'trial_type'])['log_odds'].var().reset_index())
data = data.groupby(['session', 'trial_type'])['log_odds'].median().reset_index()

x_map_var = {k: i for i, k in enumerate(trial_order_var)}
final_var = data.copy()
rng_var = np.random.default_rng(0)
session_jitter_var = {s: rng_var.uniform(-0.15, 0.15) for s in data['session'].unique()}
final_var['x'] = final_var['trial_type'].map(x_map_var)
final_var['x_jitter'] = final_var['x'] + final_var['session'].map(session_jitter_var)
final_var['trial_type'] = pd.Categorical(final_var['trial_type'], categories=trial_order_var, ordered=True)

sns.boxplot(x='trial_type', y='log_odds', data=final_var, showfliers=False,
            width=0.5, palette=['darkgreen', 'indigo'], ax=a,
            showcaps=False, linewidth=0,
            medianprops=dict(color='white', linewidth=1.5),
            whiskerprops=dict(color='black', linewidth=0.7))

for _, sess_data in final_var.groupby('session'):
    a.plot(sess_data['x_jitter'], sess_data['log_odds'], alpha=0.4, linewidth=1, zorder=2, color='grey')

a.set_xticks(list(x_map_var.values()))
a.set_xticklabels(['STM', 'RepL'])
a.set_xlabel('Trial Type')
a.set_ylabel('Variance of log odds\nduring delay period')
a.locator_params(axis='y', nbins=4)

wide_var = data.pivot(index='session', columns='trial_type', values='log_odds').dropna()
tstat_var, pval_var = stats.ttest_rel(wide_var.iloc[:, 0], wide_var.iloc[:, 1])
print(f"paired t-test: t={tstat_var:.3f}, p={pval_var:.3g}")

stars_var = plots.p_to_stars(pval_var)
y_top_var = data['log_odds'].max()
offset_var = 0.15 * y_top_var
bar_h_var = 0.05 * y_top_var
a.plot([0, 0, 1, 1],
           [y_top_var + offset_var, y_top_var + offset_var + bar_h_var,
            y_top_var + offset_var + bar_h_var, y_top_var + offset_var],
           lw=1, c='k', clip_on=False)
a.text(0.5, y_top_var + offset_var + bar_h_var * 1.1, stars_var, ha='center', va='bottom', fontsize=7)

# #######################################################################################
# panels f & f_res: WM→RL cross-decoder, RL trials
# #######################################################################################

variable = 'RL_roll_1'
file_name = '/trainedWM_testedRL_delay_V6'
df_cum_res = pd.read_csv(path+file_name+'_res.csv', index_col=0)
df_cum_sti = pd.read_csv(path+file_name+'_sti.csv', index_col=0)

df_cum_sti = df_cum_sti.loc[df_cum_sti.trial_type == variable]
df_cum_res = df_cum_res.loc[df_cum_res.trial_type == variable]

# trained_trials = df_cum_sti.session.unique()
# df_cum_res = df_cum_res[df_cum_res['session'].isin(trained_trials)]
y_range = [0.4, 0.8]

file_name = '/trainedWM_testedRL_shufflesession_delay_V3'
df_cum_res_shuffle = pd.read_csv(path+file_name+'_res.csv', index_col=0)
df_cum_sti_shuffle = pd.read_csv(path+file_name+'_sti.csv', index_col=0)

# df_cum_sti = df_cum_sti[df_cum_sti['session'].isin(trained_trials)]
numeric_columns = ['-1.88', '-1.62', '-1.38', '-1.12', '-0.88', '-0.62', '-0.38', '-0.12',
       '0.12', '0.38', '0.62', '0.88', '1.12']
df_cum_sti[numeric_columns] += 0.5

numeric_columns = ['-0.88', '-0.62', '-0.38', '-0.12', '0.12', '0.38', '0.62', '0.88',
       '1.12', '1.38', '1.62', '1.88', '2.12', '2.38', '2.62', '2.88', '3.12', '3.38', '3.62']
df_cum_res[numeric_columns] += 0.5

plots.plot_results_session_summary(fig, f, df_cum_sti, ['black'], [variable], shuffle_df=df_cum_sti_shuffle,
                                  y_range=y_range, x_range=[-2,1.3], baseline=0)

plots.plot_results_session_summary(fig, f_res, df_cum_res, ['black'], [variable], shuffle_df=df_cum_res_shuffle,
                                  y_range=y_range, x_range=[-1,3], baseline=0)

variable = 'RL_roll_1'
file_name = '/trainedWM_testedRL_delay_V6'

df_cum_res = pd.read_csv(path+file_name+'_res.csv', index_col=0)
df_cum_sti = pd.read_csv(path+file_name+'_sti.csv', index_col=0)

df_cum_sti = df_cum_sti.loc[df_cum_sti.trial_type == variable]
df_cum_res = df_cum_res.loc[df_cum_res.trial_type == variable]

panel = f
plots.plot_results_session_summary(fig, panel, df_cum_sti, colors = ['indigo'], variables_combined = [variable],
                                   y_range = y_range, x_range = [-2,1.125], epoch = 'Stimulus_ON', baseline=0.5)

panel = f_res
plots.plot_results_session_summary(fig, panel, df_cum_res, colors = ['indigo'], variables_combined = [variable],
                                   y_range = y_range, x_range =  [-1,4], epoch = 'Delay_OFF', baseline=0.5)

f.locator_params(axis='x', nbins=4)
f.locator_params(axis='y', nbins=4)
f.set_ylabel('Decoding accuracy')

# #######################################################################################
# panels g & g_res: WM→RL cross-decoder, WM trials
# #######################################################################################

variable = 'WM_roll_1'
file_name = '/trainedWM_testedRL_delay_V6'
df_cum_res = pd.read_csv(path+file_name+'_res.csv', index_col=0)
df_cum_sti = pd.read_csv(path+file_name+'_sti.csv', index_col=0)

df_cum_sti = df_cum_sti.loc[df_cum_sti.trial_type == variable]
df_cum_res = df_cum_res.loc[df_cum_res.trial_type == variable]

# trained_trials = df_cum_sti.session.unique()
# df_cum_res = df_cum_res[df_cum_res['session'].isin(trained_trials)]

file_name = '/trainedWM_testedWM_shufflesession_delay_V2'
df_cum_res_shuffle = pd.read_csv(path+file_name+'_res.csv', index_col=0)
df_cum_sti_shuffle = pd.read_csv(path+file_name+'_sti.csv', index_col=0)

# trained_trials = df_cum_sti_shuffle.session.unique()
# df_cum_sti = df_cum_sti[df_cum_sti['session'].isin(trained_trials)]

numeric_columns = ['-1.88', '-1.62', '-1.38', '-1.12', '-0.88', '-0.62', '-0.38', '-0.12',
       '0.12', '0.38', '0.62', '0.88', '1.12']
df_cum_sti[numeric_columns] += 0.5

numeric_columns = ['-0.88', '-0.62', '-0.38', '-0.12', '0.12', '0.38', '0.62', '0.88',
       '1.12', '1.38', '1.62', '1.88', '2.12', '2.38', '2.62', '2.88', '3.12', '3.38', '3.62']
df_cum_res[numeric_columns] += 0.5

plots.plot_results_session_summary(fig, g, df_cum_sti, ['darkgreen'], [variable], shuffle_df=df_cum_sti_shuffle,
                                  y_range=y_range, x_range=[-2,1.3], baseline=0)

plots.plot_results_session_summary(fig, g_res, df_cum_res, ['darkgreen'], [variable], shuffle_df=df_cum_res_shuffle,
                                  y_range=y_range, x_range=[-1,3], baseline=0)

variable = 'WM_roll_1'
file_name = '/trainedWM_testedRL_delay_V6'

df_cum_res = pd.read_csv(path+file_name+'_res.csv', index_col=0)
df_cum_sti = pd.read_csv(path+file_name+'_sti.csv', index_col=0)

df_cum_sti = df_cum_sti.loc[df_cum_sti.trial_type == variable]
df_cum_res = df_cum_res.loc[df_cum_res.trial_type == variable]

panel = g
plots.plot_results_session_summary(fig, panel, df_cum_sti, colors = ['black'], variables_combined = [variable],
                                   y_range = y_range, x_range = [-2,1.125], epoch = 'Stimulus_ON', baseline=0.5)

panel = g_res
plots.plot_results_session_summary(fig, panel, df_cum_res, colors = ['black'], variables_combined = [variable],
                                   y_range = y_range, x_range =  [-1,4], epoch = 'Delay_OFF', baseline=0.5)

g.locator_params(axis='x', nbins=4)
g.locator_params(axis='y', nbins=4)
g.set_ylabel('Decoding accuracy')
for _ax in [e,f_res, g_res]:
    sns.despine(ax=_ax, left=True)
# ---------------------------------------------------                                 -----------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------------

# variable = 'RL_roll_1'
# file_name = '/trainedcorrect_testedRLcorrect_currentcue_V1'

# df_cum_res = pd.read_csv(path+file_name+'_res.csv', index_col=0)
# df_cum_sti = pd.read_csv(path+file_name+'_sti.csv', index_col=0)

# df_cum_sti = df_cum_sti.loc[df_cum_sti.trial_type == variable]
# df_cum_res = df_cum_res.loc[df_cum_res.trial_type == variable]


# panel = f
# plots.plot_results_session_summary(fig, panel, df_cum_sti, colors = ['indigo'], variables_combined = [variable],
#                                    y_range = y_range, x_range = [-2,1.125], epoch = 'Stimulus_ON', baseline=0.5)

# panel = g
# plots.plot_results_session_summary(fig, panel, df_cum_res, colors = ['indigo'], variables_combined = [variable],
#                                    y_range = y_range, x_range =  [-1,4], epoch = 'Delay_OFF', baseline=0.5)

# #####################

# variable = 'WM_roll_1'
# file_name = '/trainedcorrect_testedWMcorrect_currentcue_V1'

# df_cum_res = pd.read_csv(path+file_name+'_res.csv', index_col=0)
# df_cum_sti = pd.read_csv(path+file_name+'_sti.csv', index_col=0)

# df_cum_sti = df_cum_sti.loc[df_cum_sti.trial_type == variable]
# df_cum_res = df_cum_res.loc[df_cum_res.trial_type == variable]

# panel = f
# plots.plot_results_session_summary(fig, panel, df_cum_sti, colors = ['darkgreen'], variables_combined = [variable],
#                                    y_range = y_range, x_range = [-2,1.125], epoch = 'Stimulus_ON', baseline=0.5)

# panel = g
# plots.plot_results_session_summary(fig, panel, df_cum_res, colors = ['darkgreen'], variables_combined = [variable],
#                                    y_range = y_range, x_range =  [-1,4], epoch = 'Delay_OFF', baseline=0.5)

# g.locator_params(axis='x', nbins=4)
# f.locator_params(axis='y', nbins=4)

# ---------------------------------------------------                                 ------------------------------------------------------------

# save_path = 'C:/Users/Tiffany/Google Drive/WORKING_MEMORY/PAPER/Figures/'
# os.chdir(save_path)
# file_name = 'trainedall_testedRL_previous_vector_answer_after_correct_V7'
# df_cum_res = pd.read_csv(file_name+'_res.csv', index_col=0)
# df_cum_sti = pd.read_csv(file_name+'_sti.csv', index_col=0)
# plots.plot_results(df_cum_sti, df_cum_res, ['green'], variables_combined, fig = True, ax1=d, ax2=e, upper_limit=0.4)

# file_name = 'trainedall_testedRL_previous_vector_answer_after_incorrect_V7'
# df_cum_res= pd.read_csv(file_name+'_res.csv', index_col=0)
# df_cum_sti= pd.read_csv(file_name+'_sti.csv', index_col=0)
# plots.plot_results(df_cum_sti, df_cum_res, ['crimson'], variables_combined, fig = True, ax1=d, ax2=e, upper_limit=0.4)

# file_name = 'trainedall_testedWM_previous_vector_answer_after_correct_V8'
# df_cum_res= pd.read_csv(file_name+'_res.csv', index_col=0)
# df_cum_sti = pd.read_csv(file_name+'_sti.csv', index_col=0)
# plots.plot_results(df_cum_sti, df_cum_res, ['green'], variables_combined, fig = True, ax1=f, ax2=g, upper_limit=0.4)

# file_name = 'trainedall_testedWM_previous_vector_answer_after_incorrect_V8'
# df_cum_res= pd.read_csv(file_name+'_res.csv', index_col=0)
# df_cum_sti= pd.read_csv(file_name+'_sti.csv', index_col=0)
# plots.plot_results(df_cum_sti, df_cum_res, ['crimson'], variables_combined, fig = True, ax1=f, ax2=g, upper_limit=0.4)

# d.set_ylim(-0.1,0.4)
# f.set_ylim(-0.1,0.4)
# e.set_xlim(-1,3)
# g.set_xlim(-1,3)

# # ----------------------------------------------------------------------------------------------------------------------------

# save_path = 'C:/Users/Tiffany/Google Drive/WORKING_MEMORY/PAPER/ANALYSIS_Figures/'
# os.chdir(save_path)
# file_name = 'crossdecoder_alltrials_3s_r0.5_previous_choice_aftercorrect'
# df_animal_shuffle = pd.read_csv(file_name+'_shuffle.csv', index_col=0)
# df_animal_sti = pd.read_csv(file_name+'_sti.csv', index_col=0)

# train_value_list = df_animal_sti.train.unique()

# df_new = df_animal_sti.loc[:, df_animal_sti.columns != 'fold'].groupby(['subject','train']).mean()
# df_new.reset_index(inplace=True)
# df_new = df_new.groupby('train').mean()

# df_new = df_new.reindex(index=train_value_list)
# panel=h
# sns.heatmap(df_new, fmt='', center=0.5, ax=panel).invert_yaxis()

# # panel.set_xticklabels(["-2",'',"","","","","","","0",'',"","","","","","","2","","","","","","","","4","","","","","","","","6","","","","","","","","8"])
# # panel.set_yticklabels(["-2",'',"","","0",'',"","","2",'',"","","4",'',"","","6",'',"","","8"])
# # panel.set_yticks([["-2",'',"","","","","0",'',"","","","2",'',"","","4",'',"","","6"]])
# panel.set_xlabel("Testing time from Stim. Onset (s)")
# panel.set_ylabel("Training time from Stim. Onset (s)")

# # ----------------------------------------------------------------------------------------------------------------------------

# # Recover the diagonal for all the animals
# df_temp=pd.DataFrame()
# first=True
# for train_value in train_value_list:
#     real_value = (float(train_value.split('_')[0]) + float(train_value.split('_')[1]))/2
#     if real_value == 7.75:
#         continue
#     try:
#         df_temp = df_animal_sti.loc[df_animal_sti.train==train_value].groupby('session')[[str(real_value)]].mean().reset_index()
#     except:
#         df_temp = df_animal_sti.loc[df_animal_sti.train==train_value].groupby('session')[[real_value]].mean().reset_index()
#     if first:
#         df_diagonal = df_temp
#         first=False
#     else:
#         df_diagonal = pd.merge(df_diagonal, df_temp, on=['session'])

# delay=3

# for panel, df_cum_sti, upper_limit in zip([i,j],[df_animal_sti.loc[df_animal_sti.train =='-6.5_-6.0'],
#                                       df_diagonal],[0.3,0.4]):
#     plots.plot_decoder(panel, df_cum_sti,baseline=0.5,individual_sessions=False, upper_limit=upper_limit)


# # ----------------------------------------------------------------------------------------------------------------------------

# plt.savefig(save_path+'/Fig 6_panel_supp4.svg', bbox_inches='tight',dpi=300)
# plt.savefig(save_path+'/Fig 6_panel_supp4.png', bbox_inches='tight',dpi=300)
plt.show()