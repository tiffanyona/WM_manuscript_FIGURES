# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 12:06:44 2022

@author: Tiffany
"""
COLORLEFT = 'teal'
COLORRIGHT = '#FF8D3F'

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
import os
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
#Import all needed libraries
from neo.core import SpikeTrain
from quantities import ms, s, Hz
from elephant.statistics import mean_firing_rate
from elephant.statistics import time_histogram, instantaneous_rate
from elephant.kernels import GaussianKernel

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import ROOT, ANALYSIS_DATA, FIGURES_OUT, DATA_DIR
sys.path.insert(0, str(ROOT / 'src'))
import functions as plots
from functions import add_stat_annotation, convolveandplot, plot_decoder_single, plot_decoder_shuffle, new_convolve, plotsingledelay

path = str(DATA_DIR / 'supp_figures' / 'Supp. 6.1')
save_path = str(FIGURES_OUT / 'supp_figures' / 'supp_fig_8')

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
fig = plt.figure(figsize=(14*cm, 16*cm))
gs = gridspec.GridSpec(nrows=3, ncols=8, figure=fig)

a1 = fig.add_subplot(gs[0, 0:4])
a2 = fig.add_subplot(gs[0, 4:6])
b = fig.add_subplot(gs[1, 0:2])
c = fig.add_subplot(gs[1, 2:6])
d = fig.add_subplot(gs[2, 0:2])
e = fig.add_subplot(gs[2, 2:4])
f = fig.add_subplot(gs[2, 4:6])
g = fig.add_subplot(gs[2, 6:8])

# h = fig.add_subplot(gs[0:2, 0:2])
# i = fig.add_subplot(gs[3, 4:6])
# j = fig.add_subplot(gs[3, 4:6])

fig.text(0.01, 1, 'a', fontsize=10, fontweight='bold', va='top')
fig.text(0.7, 1, 'b', fontsize=10, fontweight='bold', va='top')
fig.text(0.01, 0.5, 'c', fontsize=10, fontweight='bold', va='top')
fig.text(0.5, 0.5, 'd', fontsize=10, fontweight='bold', va='top')


# -----------------###########################################################################################################-----------------------

file_name = r'\single_delay_WM_roll0.6_delay_0.25_all_V3'
df_cum_sti = pd.read_csv(path+file_name+'.csv', index_col=0)

# list_sessions = df_cum_sti.session.unique()
# list_sessions = ['E04_2021-04-03_16-12-15.csv', 'E13_2021-05-25_16-26-57.csv',
#        'E13_2021-05-26_15-01-42.csv', 'E14_2021-04-02_12-53-42.csv',
#        'E17_2022-01-31_16-30-44.csv', 'E17_2022-02-01_17-02-16.csv',
#        'E17_2022-02-13_17-14-28.csv', 'E20_2022-02-13_15-10-51.csv',
#        'E20_2022-02-15_16-02-28.csv', 'E20_2022-03-01_16-11-01.csv',
#        'E22_2022-01-13_16-34-24.csv', 'E22_2022-01-14_16-50-37.csv',
#        'E22_2022-01-16_18-00-47.csv', 'E22_2022-01-17_18-05-16.csv']
# df_cum_sti = df_cum_sti.loc[df_cum_sti['session'].isin(list_sessions)]

scores = df_cum_sti.groupby('session').score.mean().reset_index()
list_exclude = scores.loc[scores.score<0.55].session.unique()
df_cum_sti = df_cum_sti[~df_cum_sti['session'].isin(list_exclude)]

#Variables for testing
colors=['crimson', 'pink']
variables = ['WM_roll','RL_roll']
hits = [0,0]
ratios = [0.6,0.4]
variables_combined=[variables[0]+'_'+str(hits[0]),variables[1]+'_'+str(hits[1])]

panel=a1
delay=10
plotsingledelay(df_cum_sti, panel, colors, variables_combined, delay, invert_list=[False, False, False, False], baseline=0)
panel.locator_params(axis='y', nbins=4)
# panel.locator_params(axis='x', nbins=5)
# panel.locator_params(axis='y', nbins=10)

# ------######################################## Session difference ###############################-----------------------

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
# ----------------------------------------------------------------------------------------------------------------------------

os.chdir(path)
file_name = r'\slope_comparison_paired_sessions'
df = pd.read_csv(path+file_name+'.csv', index_col=0)

list_sessions = df.session.unique()
# df = df.loc[df['session'].isin(list_sessions)]

df = df.loc[df.session != 'E13_2021-05-25_16-26-57.csv']

panel = a2

melted_data = pd.melt(df)
palette = sns.color_palette(['black'], len(df.session.unique()))

sns.boxplot(data=df, x='variable', y='slope', palette=['crimson', 'pink'],ax = panel)
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

# ------#########################################################################################-----------------------

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

plots.plot_results_session_summary_substract(fig, b, df_cum_sti, df_cum_sti_shuffle,
                                  color = 'indigo', variable = variable, y_range = y_range, x_range = [-2,1.125], baseline=0)

plots.plot_results_session_summary_substract(fig, c, df_cum_res, df_cum_res_shuffle,
                                  color = 'indigo', variable = variable, y_range = y_range, x_range = [-1,4], baseline=0)
# c.locator_params(axis='x', nbins=6)
b.locator_params(axis='x', nbins=4)
c.locator_params(axis='y', nbins=4)
b.locator_params(axis='y', nbins=4)

# ----------------------------------------------------------------------------------------------------------------------------

variable = 'RL_roll_1'
file_name = '/trainedWM_testedRL_delay_V6'

df_cum_res = pd.read_csv(path+file_name+'_res.csv', index_col=0)
df_cum_sti = pd.read_csv(path+file_name+'_sti.csv', index_col=0)

df_cum_sti = df_cum_sti.loc[df_cum_sti.trial_type == variable]
df_cum_res = df_cum_res.loc[df_cum_res.trial_type == variable]

y_range = [0.45, 0.8]

panel = d
plots.plot_results_session_summary(fig, panel, df_cum_sti, colors = ['indigo'], variables_combined = [variable],
                                   y_range = y_range, x_range = [-2,1.125], epoch = 'Stimulus_ON', baseline=0.5)

panel = e
plots.plot_results_session_summary(fig, panel, df_cum_res, colors = ['indigo'], variables_combined = [variable],
                                   y_range = y_range, x_range =  [-1,4], epoch = 'Delay_OFF', baseline=0.5)


#####################

variable = 'WM_roll_1'
file_name = '/trainedWM_testedRL_delay_V6'

df_cum_res = pd.read_csv(path+file_name+'_res.csv', index_col=0)
df_cum_sti = pd.read_csv(path+file_name+'_sti.csv', index_col=0)

df_cum_sti = df_cum_sti.loc[df_cum_sti.trial_type == variable]
df_cum_res = df_cum_res.loc[df_cum_res.trial_type == variable]


panel = d
plots.plot_results_session_summary(fig, panel, df_cum_sti, colors = ['darkgreen'], variables_combined = [variable],
                                   y_range = y_range, x_range = [-2,1.125], epoch = 'Stimulus_ON', baseline=0.5)

panel = e
plots.plot_results_session_summary(fig, panel, df_cum_res, colors = ['darkgreen'], variables_combined = [variable],
                                   y_range = y_range, x_range =  [-1,4], epoch = 'Delay_OFF', baseline=0.5)

d.locator_params(axis='x', nbins=4)
e.locator_params(axis='y', nbins=4)

# ---------------------------------------------------                                 ------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------------

variable = 'RL_roll_1'
file_name = '/trainedcorrect_testedRLcorrect_currentcue_V1'

df_cum_res = pd.read_csv(path+file_name+'_res.csv', index_col=0)
df_cum_sti = pd.read_csv(path+file_name+'_sti.csv', index_col=0)

df_cum_sti = df_cum_sti.loc[df_cum_sti.trial_type == variable]
df_cum_res = df_cum_res.loc[df_cum_res.trial_type == variable]


panel = f
plots.plot_results_session_summary(fig, panel, df_cum_sti, colors = ['indigo'], variables_combined = [variable],
                                   y_range = y_range, x_range = [-2,1.125], epoch = 'Stimulus_ON', baseline=0.5)

panel = g
plots.plot_results_session_summary(fig, panel, df_cum_res, colors = ['indigo'], variables_combined = [variable],
                                   y_range = y_range, x_range =  [-1,4], epoch = 'Delay_OFF', baseline=0.5)

#####################

variable = 'WM_roll_1'
file_name = '/trainedcorrect_testedWMcorrect_currentcue_V1'

df_cum_res = pd.read_csv(path+file_name+'_res.csv', index_col=0)
df_cum_sti = pd.read_csv(path+file_name+'_sti.csv', index_col=0)

df_cum_sti = df_cum_sti.loc[df_cum_sti.trial_type == variable]
df_cum_res = df_cum_res.loc[df_cum_res.trial_type == variable]

panel = f
plots.plot_results_session_summary(fig, panel, df_cum_sti, colors = ['darkgreen'], variables_combined = [variable],
                                   y_range = y_range, x_range = [-2,1.125], epoch = 'Stimulus_ON', baseline=0.5)

panel = g
plots.plot_results_session_summary(fig, panel, df_cum_res, colors = ['darkgreen'], variables_combined = [variable],
                                   y_range = y_range, x_range =  [-1,4], epoch = 'Delay_OFF', baseline=0.5)

g.locator_params(axis='x', nbins=4)
f.locator_params(axis='y', nbins=4)

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

# Show the figure
sns.despine(offset = 5)
plt.subplots_adjust(left=0.07,
                    bottom=0.07,
                    right=0.97,
                    top=0.97,
                    wspace=1.5,
                    hspace=0.5)

# plt.savefig(save_path+'/Fig 6_panel_supp4.svg', bbox_inches='tight',dpi=300)
# plt.savefig(save_path+'/Fig 6_panel_supp4.png', bbox_inches='tight',dpi=300)

plt.show()
