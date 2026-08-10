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
import numpy as np
import seaborn as sns
from scipy import stats
#Import all needed libraries
from neo.core import SpikeTrain
from quantities import ms
from elephant.statistics import time_histogram, instantaneous_rate
from elephant.kernels import GaussianKernel

from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
import sys

pandas2ri.activate()

base     = importr('base')
car      = importr('car')
stats    = importr('stats')
lme4     = importr('lme4')
scales   = importr('scales')
lmerTest = importr('lmerTest')

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import ROOT, FIGURES_OUT, DATA_DIR
sys.path.insert(0, str(ROOT / 'src'))
sys.path.append("G:/Mi unidad/WORKING_MEMORY/EXPERIMENTS/ELECTROPHYSIOLOGY/ANALYSIS/functions/")
from functions import add_stat_annotation, convolveandplot, new_convolve
# sys.path.append("G:/My Drive/WORKING_MEMORY/EXPERIMENTS/ELECTROPHYSIOLOGY/ANALYSIS/functions/")


import functions as plots

save_path = str(FIGURES_OUT / 'fig_5_ephys_repl') + '/'
path = str(DATA_DIR) + '/'

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

# Create a figure with 6 subplots using a GridSpec
fig = plt.figure(figsize=(25*cm, 18  *cm))
gs = gridspec.GridSpec(nrows=5, ncols=5, figure=fig)

# Create the subplots
a1 = fig.add_subplot(gs[0, 0])
a2  = fig.add_subplot(gs[0, 1])

b1 = fig.add_subplot(gs[1, 0])
b2  = fig.add_subplot(gs[1, 1])

h1 = fig.add_subplot(gs[2, 0])
h2  = fig.add_subplot(gs[2, 1])

i1 = fig.add_subplot(gs[3, 1])
j1 = fig.add_subplot(gs[4, 1])

c1 = fig.add_subplot(gs[0, 2])
c2  = fig.add_subplot(gs[1, 2])
c3 = fig.add_subplot(gs[2, 2])

d1 = fig.add_subplot(gs[0, 3])
d2  = fig.add_subplot(gs[1, 3])
d3  = fig.add_subplot(gs[2, 3])

f = fig.add_subplot(gs[3, 0])

e1  = fig.add_subplot(gs[3:4, 2])
e2  = fig.add_subplot(gs[3:4, 3])
e3  = fig.add_subplot(gs[3:4, 4])

# fig.text(0.01, 0.99, 'a', fontsize=10, fontweight='bold', va='top')
# fig.text(0.34, 0.99, 'b', fontsize=10, fontweight='bold', va='top')
# fig.text(0.01, 0.86, 'c', fontsize=10, fontweight='bold', va='top')
# fig.text(0.68, 0.99, 'd', fontsize=10, fontweight='bold', va='top')
# fig.text(0.01, 0.72, 'e', fontsize=10, fontweight='bold', va='top')
# fig.text(0.01, 0.58, 'f', fontsize=10, fontweight='bold', va='top')
# fig.text(0.68, 0.58, 'g', fontsize=10, fontweight='bold', va='top')
# fig.text(0.01, 0.2, 'h', fontsize=10, fontweight='bold', va='top')
# fig.text(0.01, 0.2, 'i', fontsize=10, fontweight='bold', va='top')


# ---------------------------------------------------------------------------
# Panels a1/a2 — cross-decoder: stimulus-aligned (a1) and response-aligned (a2)
# ---------------------------------------------------------------------------

#### ------------------------- Stimulus plot
panel = a1
y_range = [-0.1, 0.45]

os.chdir(path)
variable = 'WM_roll_1'
# file_name = 'trainedWM_testedRL_stimulus_remove_low_accuracy_sessions'
file_name = 'trainedWM_testedRL_stimulus_V4'
df_cum_sti = pd.read_csv(path+file_name+'_sti.csv', index_col=0)

df_cum_sti = df_cum_sti.loc[df_cum_sti.trial_type == variable]

trained_trials = df_cum_sti.session.unique()

file_name = 'trainedWM_testedWM_shufflesession_stimulus'
df_cum_sti_shuffle = pd.read_csv(path+file_name+'_sti.csv', index_col=0)

trained_trials = df_cum_sti_shuffle.session.unique()
df_cum_sti = df_cum_sti[df_cum_sti['session'].isin(trained_trials)]

plots.plot_results_session_summary_substract(fig, panel, df_cum_sti, df_cum_sti_shuffle,
                                  color = 'darkgreen', variable = variable, y_range = y_range, x_range = [-2,1.3], baseline=0)

variable = 'RL_roll_1'
# file_name = 'trainedWM_testedRL_stimulus_remove_low_accuracy_sessions'
file_name = 'trainedWM_testedRL_stimulus_V4'
df_cum_sti = pd.read_csv(path+file_name+'_sti.csv', index_col=0)

df_cum_sti = df_cum_sti.loc[df_cum_sti.trial_type == variable]

# trained_trials = df_cum_sti.session.unique()

file_name = 'trainedWM_testedRL_shufflesession_stimulus'
df_cum_sti_shuffle = pd.read_csv(path+file_name+'_sti.csv', index_col=0)

df_cum_sti_shuffle = df_cum_sti_shuffle[df_cum_sti_shuffle['session'].isin(trained_trials)]

# trained_trials = df_cum_sti_shuffle.session.unique()
# df_cum_sti = df_cum_sti[df_cum_sti['session'].isin(trained_trials)]

plots.plot_results_session_summary_substract(fig, panel, df_cum_sti, df_cum_sti_shuffle,
                                  color = 'indigo', variable = variable, y_range = y_range, x_range = [-2,1.3], baseline=0)

#### ------------------------- Response plot
panel = a2
y_range = [-0.1, 0.45]
variable = 'WM_roll_1'
file_name = 'trainedWM_testedRL_response_V4'
df_cum_res = pd.read_csv(path+file_name+'_res.csv', index_col=0)
df_cum_sti = pd.read_csv(path+file_name+'_sti.csv', index_col=0)

df_cum_sti = df_cum_sti.loc[df_cum_sti.trial_type == variable]
df_cum_res = df_cum_res.loc[df_cum_res.trial_type == variable]

# trained_trials = df_cum_sti.session.unique()
# df_cum_res = df_cum_res[df_cum_res['session'].isin(trained_trials)]

file_name = 'trainedWM_testedRL_shufflesession_response_V2'
df_cum_res_shuffle = pd.read_csv(path+file_name+'_res.csv', index_col=0)
df_cum_sti_shuffle = pd.read_csv(path+file_name+'_sti.csv', index_col=0)

# trained_trials = df_cum_sti_shuffle.session.unique()
# df_cum_sti = df_cum_sti[df_cum_sti['session'].isin(trained_trials)]

plots.plot_results_session_summary_substract(fig, panel, df_cum_res, df_cum_res_shuffle,
                                  color = 'darkgreen', variable = variable, y_range = y_range, x_range = [-1,3], baseline=0)

variable = 'RL_roll_1'
file_name = 'trainedWM_testedRL_response_V4'
df_cum_res = pd.read_csv(path+file_name+'_res.csv', index_col=0)
df_cum_sti = pd.read_csv(path+file_name+'_sti.csv', index_col=0)

df_cum_sti = df_cum_sti.loc[df_cum_sti.trial_type == variable]
df_cum_res = df_cum_res.loc[df_cum_res.trial_type == variable]

# trained_trials = df_cum_sti.session.unique()
# df_cum_res = df_cum_res[df_cum_res['session'].isin(trained_trials)]

file_name = 'trainedWM_testedRL_shufflesession_response'
df_cum_res_shuffle = pd.read_csv(path+file_name+'_res.csv', index_col=0)
df_cum_sti_shuffle = pd.read_csv(path+file_name+'_sti.csv', index_col=0)

# trained_trials = df_cum_sti_shuffle.session.unique()
# df_cum_sti = df_cum_sti[df_cum_sti['session'].isin(trained_trials)]

plots.plot_results_session_summary_substract(fig, panel, df_cum_res, df_cum_res_shuffle,
                                  color = 'indigo', variable = variable, y_range = y_range, x_range = [-1,3], baseline=0)

# ---------------------------------------------------------------------------
# Panels b1/b2 — cross-decoder during delay period (stimulus and response aligned)
# ---------------------------------------------------------------------------

y_range = [-0.1, 0.45]
variable = 'WM_roll_1'
file_name = 'trainedWM_testedRL_delay_V6'
df_cum_res = pd.read_csv(path+file_name+'_res.csv', index_col=0)
df_cum_sti = pd.read_csv(path+file_name+'_sti.csv', index_col=0)

df_cum_sti = df_cum_sti.loc[df_cum_sti.trial_type == variable]
df_cum_res = df_cum_res.loc[df_cum_res.trial_type == variable]

# trained_trials = df_cum_sti.session.unique()
# df_cum_res = df_cum_res[df_cum_res['session'].isin(trained_trials)]

file_name = 'trainedWM_testedWM_shufflesession_delay_V2'
df_cum_res_shuffle = pd.read_csv(path+file_name+'_res.csv', index_col=0)
df_cum_sti_shuffle = pd.read_csv(path+file_name+'_sti.csv', index_col=0)

# trained_trials = df_cum_sti_shuffle.session.unique()
# df_cum_sti = df_cum_sti[df_cum_sti['session'].isin(trained_trials)]

plots.plot_results_session_summary_substract(fig, b1, df_cum_sti, df_cum_sti_shuffle,
                                  color = 'darkgreen', variable = variable, y_range = y_range, x_range = [-2,1.3], baseline=0)

plots.plot_results_session_summary_substract(fig, b2, df_cum_res, df_cum_res_shuffle,
                                  color = 'darkgreen', variable = variable, y_range = y_range, x_range = [-1,3], baseline=0)


# -----------------------------------------------------
variable = 'RL_roll_1'
file_name = 'trainedWM_testedRL_delay_V6'
df_cum_res = pd.read_csv(path+file_name+'_res.csv', index_col=0)
df_cum_sti = pd.read_csv(path+file_name+'_sti.csv', index_col=0)

df_cum_sti = df_cum_sti.loc[df_cum_sti.trial_type == variable]
df_cum_res = df_cum_res.loc[df_cum_res.trial_type == variable]

# trained_trials = df_cum_sti.session.unique()
# df_cum_res = df_cum_res[df_cum_res['session'].isin(trained_trials)]

file_name = 'trainedWM_testedRL_shufflesession_delay_V3'
df_cum_res_shuffle = pd.read_csv(path+file_name+'_res.csv', index_col=0)
df_cum_sti_shuffle = pd.read_csv(path+file_name+'_sti.csv', index_col=0)

# df_cum_sti = df_cum_sti[df_cum_sti['session'].isin(trained_trials)]

plots.plot_results_session_summary_substract(fig, b1, df_cum_sti, df_cum_sti_shuffle,
                                  color = 'indigo', variable = variable, y_range = y_range, x_range = [-2,1.3], baseline=0)

plots.plot_results_session_summary_substract(fig, b2, df_cum_res, df_cum_res_shuffle,
                                  color = 'indigo', variable = variable, y_range = y_range, x_range = [-1,3], baseline=0)


# ---------------------------------------------------------------------------
# Panels h1/h2 — example session cross-decoder (E20 2022-02-26)
# ---------------------------------------------------------------------------

#Variables for testing
colors=['darkgreen','indigo']
variables = ['WM_roll','RL_roll']
hits = [1,1]
ratios = [0.6,0.4]
variables_combined=[variables[0]+'_'+str(hits[0]),variables[1]+'_'+str(hits[1])]

file_name = 'E20_2022-02-26_16-49-05example_RLWM'
df_cum_sti = pd.read_csv(file_name+'_sti.csv', index_col=0)
df_cum_res= pd.read_csv(file_name+'_res.csv', index_col=0)
df_cum_shuffle_sti = pd.read_csv(file_name+'_sti_shuffle.csv', index_col=0)
df_cum_shuffle_res= pd.read_csv(file_name+'_res_shuffle.csv', index_col=0)

for color, variable,left,right in zip(colors,variables_combined,[h1, h1, h1, h1],[h2, h2, h2, h2]):

    # Aligmnent for Stimulus cue

    real = np.array(np.mean(df_cum_sti.loc[(df_cum_sti['trial_type'] == variable)].groupby('session').median().drop(columns=['score','fold'])))
    times = df_cum_sti.loc[(df_cum_sti['trial_type'] == variable)]
    times = np.array(times.drop(columns=['session','fold','score','subject','trial_type'],axis = 1).columns.astype(float))

    df_lower = pd.DataFrame()
    df_upper = pd.DataFrame()


    df_new = pd.DataFrame()
    for iteration in np.arange(1,100):
        try:
            df_new[iteration]= df_cum_shuffle_sti.loc[(df_cum_shuffle_sti.trial_type==variable)].groupby('times').mean()[float(iteration)]
        except:
            df_new[iteration]= df_cum_shuffle_sti.loc[(df_cum_shuffle_sti.trial_type==variable)].groupby('times').mean()[str(float(iteration))]

    y_mean= df_new.mean(axis=1).values
    upper =  df_new.quantile(q=0.975, interpolation='linear',axis=1) - y_mean
    lower =  df_new.quantile(q=0.025, interpolation='linear',axis=1) - y_mean
    x=times

    left.plot(times,real, color=color)
    left.plot(x, lower+real, color=color, linestyle = '',alpha=0.6, linewidth=0)
    left.plot(x, upper+real, color=color, linestyle = '',alpha=0.6, linewidth=0)
    left.fill_between(x, lower+real, upper+real, alpha=0.2, color=color, linewidth=0)
    left.axhline(y=0.0,linestyle=':',color='black')
    left.set_ylim(-0.2,0.5)
    left.fill_betweenx(np.arange(-1.1,3.1,0.1), 0,0.35, color='lightgrey', alpha=1, linewidth=0)

    # -------------------- For Aligment to Go cue
    real = np.array(np.mean(df_cum_res.loc[(df_cum_res['trial_type'] == variable)].groupby('session').median().drop(columns=['score','fold'])))
    times = df_cum_res.loc[(df_cum_res['trial_type'] == variable)]
    times = np.array(times.drop(columns=['session','fold','score','subject','trial_type'],axis = 1).columns.astype(float))

    df_lower = pd.DataFrame()
    df_upper = pd.DataFrame()

    df_new = pd.DataFrame()
    for iteration in np.arange(1,100):
        try:
            df_new[iteration]= df_cum_shuffle_res.loc[(df_cum_shuffle_res.trial_type==variable)].groupby('times').mean()[iteration]
        except:
            df_new[iteration]= df_cum_shuffle_res.loc[(df_cum_shuffle_res.trial_type==variable)].groupby('times').mean()[str(float(iteration))]

    y_mean= df_new.mean(axis=1).values
    upper =  df_new.quantile(q=0.975, interpolation='linear',axis=1) - y_mean
    lower =  df_new.quantile(q=0.025, interpolation='linear',axis=1) - y_mean

    x=times

    # ax2.plot(x, y_mean, color=color)
    right.plot(times,real, color=color)
    right.plot(x, real+lower, color=color, linestyle = '',alpha=0.6, linewidth=0)
    right.plot(x, real+upper, color=color, linestyle = '',alpha=0.6, linewidth=0)
    right.fill_between(x, real+lower, real+upper, alpha=0.2, color=color, linewidth=0)
    right.set_ylim(-0.2,0.5)
    right.axhline(y=0.0,linestyle=':',color='grey')
    right.fill_betweenx(np.arange(-1.1,3.1,0.1), 0,0.2, color='grey', alpha=1, linewidth=0)
    right.set_xlabel('Time from Go cue onset (s)')
    left.set_xlabel('Time from stimulus onset (s)')
    left.set_ylabel('Excess decoding\n accuracy')
    left.set_xlim(-2,1)
    right.set_xlim(-1,2)

    # right.set_title('Mouse E20 2022-02-26')

# ---------------------------------------------------------------------------
# Panel f — log-odds by state and epoch (mixed-effects model)
# ---------------------------------------------------------------------------

file_name = 'logodds_WM1_RL1'

df_final = pd.read_csv(path+file_name+'.csv', index_col=0)


panel=f

df_results = df_final.groupby(['session', 'trial_type','epoch']).log_odds.mean()
df_results = df_results.reset_index()

sns.boxplot(x='trial_type', y='log_odds',hue='epoch', order=['WM_roll_1','RL_roll_1'], palette=['darkgreen','lightgreen', 'darkred', 'lightred'],linewidth=0 ,ax = panel, data=df_results, showmeans=True)
sns.boxplot(x='trial_type', y='log_odds',hue='epoch', order=['WM_roll_1','RL_roll_1'], palette=['darkgreen','lightgreen', 'darkred', 'lightred'],linewidth=1 ,ax = panel, data=df_results, showmeans=True)

df_plots = df_results.loc[(df_results.trial_type == 'WM_roll_1')&(df_results.epoch == 'early')]
xA = np.random.normal(-0.25, 0.08, len(df_plots))
sns.scatterplot(x=xA,y='log_odds',data=df_plots,ax=panel,alpha=0.9,legend=False, color='darkgreen')

df_plots = df_results.loc[(df_results.trial_type == 'WM_roll_1')&(df_results.epoch == 'late')]
xA = np.random.normal(0.2, 0.08, len(df_plots))
sns.scatterplot(x=xA,y='log_odds',data=df_plots,ax=panel,alpha=0.5,legend=False, color='darkgreen')

df_plots = df_results.loc[(df_results.trial_type == 'RL_roll_1')&(df_results.epoch == 'early')]
xA = np.random.normal(.75, 0.08, len(df_plots))
sns.scatterplot(x=xA,y='log_odds',data=df_plots,ax=panel,alpha=0.9,legend=False, color='indigo')

df_plots = df_results.loc[(df_results.trial_type == 'RL_roll_1')&(df_results.epoch == 'late')]
xA = np.random.normal(1.2, 0.08, len(df_plots))
sns.scatterplot(x=xA,y='log_odds',data=df_plots,ax=panel,alpha=0.5,legend=False, color='indigo')

panel.set_ylim(-1,9)
panel.hlines(xmin=-0.5, xmax=1.5, y=0, linestyle=':')
panel.set_ylabel('Log odds')

formula="log_odds ~ state*epoch + (1|session:fold)"
r_df  = ro.conversion.py2ri(df_final)
model = lme4.lmer(formula, data=r_df)

for i, v in enumerate(list(base.summary(model).names)):
    if v in ['coefficients']:
        print (base.summary(model).rx2(v))
print(base.summary(model))
print(car.Anova(model))

# ---------------------------------------------------------------------------
# Panels c1/c2/c3 — single WM trial population activity (E17, trial 223)
# ---------------------------------------------------------------------------

T=223
filename = 'E17_2022-01-31_16-30-44.csv'

file_name = 'decoder_'+str(T)+'_'+filename
df_decoder = pd.read_csv(path+file_name, index_col=0)

file_name = 'df_'+str(T)+'_'+filename
df = pd.read_csv(path+file_name, index_col=0)

file_name = 'convolve_'+str(T)+'_'+filename
big_data = pd.read_csv(path+file_name, index_col=0)

plots.single_trial_with_decoder(df, df_decoder, big_data, filename, T, panels = [c1,c2,c3])
c1.set_title('Correct WM trial (Right stimulus)', fontsize=8)

# Panels d1/d2/d3 — single RepL trial (E17, trial 21)
T=21
filename = 'E17_2022-01-31_16-30-44.csv'

file_name = 'decoder_'+str(T)+'_'+filename
df_decoder =pd.read_csv(path+file_name, index_col=0)

file_name = 'df_'+str(T)+'_'+filename
df = pd.read_csv(path+file_name, index_col=0)

file_name = 'convolve_'+str(T)+'_'+filename
big_data = pd.read_csv(path+file_name, index_col=0)

plots.single_trial_with_decoder(df, df_decoder, big_data, filename, T, panels = [d1,d2,d3])
d3.set_ylim(-10,10)
c3.set_ylim(-10,10)

d1.set_title('Correct RepL trial (Right stimulus)', fontsize=8)

# ---------------------------------------------------------------------------
# Panels i1/j1 — example neuron PSTH: WM vs HB trials
# ---------------------------------------------------------------------------

# session: E22_2022-01-13_16-34-24
file_name = 'WMvsHB_example_138'
df = pd.read_csv(path+file_name+'.csv')

delay = 10
colors = [COLORRIGHT,COLORLEFT]
labels = ['Right stimulus','Left stimulus']
align = 'Stimulus_ON'

# temp_df = df.loc[(df.hit ==1)&(df.cluster_id == 138)]
temp_df = df.loc[(df.cluster_id == 138)]
temp_df['state'] = np.where(temp_df['WM_roll'] > 0.6, 1, 0)

for cluster_id in temp_df.cluster_id.unique():
    print(cluster_id)
    j=1
    j = convolveandplot(temp_df.loc[(temp_df.vector_answer == 0)&(temp_df.hit == 1)], i1, j1, variable='state', cluster_id = cluster_id, delay = delay, j=j,
                       labels=['WM left','HB left'], colors=['darkgreen', 'indigo'], kernel=200)

    # j = convolveandplot(temp_df.loc[(temp_df.vector_answer == 0)&(temp_df.hit == 0)], i1, j1, variable='state', cluster_id = cluster_id, delay = delay, j=j,
    #                    labels=['WM left','HB left'], colors=['lightgrey', 'lightgrey'], kernel=200)

    i1.set_ylim(0,19)


# ---------------------------------------------------------------------------
# Panels e1/e2/e3 — previous trial decoder (current cue aligned)
# ---------------------------------------------------------------------------

y_range = [-0.1, 0.45]

variable = 'WM_roll_1'
file_name = 'trainedcorrect_testedWMcorrect_currentcue_V1'
df_cum_res = pd.read_csv(path+file_name+'_res.csv', index_col=0)
df_cum_sti = pd.read_csv(path+file_name+'_sti.csv', index_col=0)

trained_trials = df_cum_sti.session.unique()
df_cum_res = df_cum_res[df_cum_res['session'].isin(trained_trials)]

file_name = 'trainedcorrect_testedWMcorrect_gocueprevious_shufflesession'
df_cum_res_shuffle = pd.read_csv(path+file_name+'_res.csv', index_col=0)
df_cum_sti_shuffle = pd.read_csv(path+file_name+'_sti.csv', index_col=0)

df_cum_sti_shuffle['trial_type'] = 'WM_roll_1'
df_cum_res_shuffle['trial_type'] = 'WM_roll_1'

trained_trials = df_cum_sti_shuffle.session.unique()
df_cum_sti = df_cum_sti[df_cum_sti['session'].isin(trained_trials)]

plots.plot_results_session_summary_substract(fig, e2, df_cum_sti, df_cum_sti_shuffle,
                                  color = 'darkgreen', variable = variable, y_range = y_range, x_range = [-4,1], baseline=0)

plots.plot_results_session_summary_substract(fig, e3, df_cum_res, df_cum_res_shuffle,
                                  color = 'darkgreen', variable = variable, y_range = y_range, x_range = [-1,3], baseline=0)


# -----------------------------------------------------

variable = 'RL_roll_1'
file_name = 'trainedcorrect_testedRLcorrect_currentcue_V1'
df_cum_res = pd.read_csv(path+file_name+'_res.csv', index_col=0)
df_cum_sti = pd.read_csv(path+file_name+'_sti.csv', index_col=0)

df_cum_res = df_cum_res[df_cum_res['session'].isin(trained_trials)]
file_name = 'trainedcorrect_testedRLcorrect_gocueprevious_shufflesession'

df_cum_res_shuffle = pd.read_csv(path+file_name+'_res.csv', index_col=0)
df_cum_sti_shuffle = pd.read_csv(path+file_name+'_sti.csv', index_col=0)
df_cum_sti_shuffle['trial_type'] = variable
df_cum_res_shuffle['trial_type'] = variable
df_cum_sti = df_cum_sti[df_cum_sti['session'].isin(trained_trials)]

plots.plot_results_session_summary_substract(fig, e2, df_cum_sti, df_cum_sti_shuffle,
                                  color = 'indigo', variable = variable, y_range = y_range, x_range = [-4,1], baseline=0)

plots.plot_results_session_summary_substract(fig, e3, df_cum_res, df_cum_res_shuffle,
                                  color = 'indigo', variable = variable, y_range = y_range, x_range = [-1,3], baseline=0)

x_range=[-1,4]
file_name = 'trainedcorrect_testedWMcorrect_gocueprevious_substracted'
df_cum_res = pd.read_csv(path+file_name+'_res.csv', index_col=0)
df_cum_sti = pd.read_csv(path+file_name+'_sti.csv', index_col=0)

# df_cum_res = df_cum_res[df_cum_res['session'].isin(trained_trials)]

# file_name = ''
# df_cum_res_shuffle = pd.read_csv(path+file_name+'_res.csv', index_col=0)
# df_cum_sti_shuffle = pd.read_csv(path+file_name+'_sti.csv', index_col=0)

# df_cum_sti = df_cum_sti[df_cum_sti['session'].isin(trained_trials)]

# y_range = [0.45, 0.8]

# plot_results_session_summary_substract(fig, ax2, df_cum_res, df_cum_res_shuffle,
#                                   color = 'indigo', variable = variable, y_range = y_range, x_range = [-1,4], baseline=0)

plots.plot_results_session_summary(fig, e1, df_cum_res, ['darkgreen'], df_cum_sti.trial_type.unique(), epoch = 'Delay_OFF', x_range=x_range, y_range=y_range, baseline=0.5)

file_name = 'trainedcorrect_testedRLcorrect_gocueprevious_substracted'
df_cum_res = pd.read_csv(path+file_name+'_res.csv', index_col=0)
df_cum_sti = pd.read_csv(path+file_name+'_sti.csv', index_col=0)

# df_cum_res = df_cum_res[df_cum_res['session'].isin(trained_trials)]
# print(len(df_cum_res.session.unique()))

# file_name = ''
# df_cum_res_shuffle = pd.read_csv(path+file_name+'_res.csv', index_col=0)
# df_cum_sti_shuffle = pd.read_csv(path+file_name+'_sti.csv', index_col=0)

# df_cum_sti = df_cum_sti[df_cum_sti['session'].isin(trained_trials)]

# y_range = [0.45, 0.8]

# plot_results_session_summary_substract(fig, ax2, df_cum_res, df_cum_res_shuffle,
#                                   color = 'indigo', variable = variable, y_range = y_range, x_range = [-1,4], baseline=0)

plots.plot_results_session_summary(fig, e1, df_cum_res, ['indigo'], df_cum_res.trial_type.unique(), epoch = 'Delay_OFF', x_range=x_range, y_range=y_range, baseline=0.0)


# ----------------------------------------------------------------------------------------------------------------------------

# Show the figure
sns.despine()
# Show the figure
plt.subplots_adjust(left=0.03,
                    bottom=0.03,
                    right=0.97,
                    top=0.97,
                    wspace=.5,
                    hspace=0.5)

# plt.savefig(save_path+'/Fig. 6. Ephys HB_V5.svg', bbox_inches='tight',dpi=300)
# plt.savefig(save_path+'/Fig. 6. Ephys HB_V4.png', bbox_inches='tight',dpi=300)

plt.show()
