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
from matplotlib.backends.backend_pdf import PdfPages

from neo.core import SpikeTrain
from quantities import ms
from elephant.statistics import time_histogram, instantaneous_rate
from elephant.kernels import GaussianKernel

from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

base     = importr('base')
car      = importr('car')
Rstats   = importr('stats')
lme4     = importr('lme4')
scales   = importr('scales')
lmerTest = importr('lmerTest')

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import ROOT, FIGURES_OUT, DATA_DIR
sys.path.insert(0, str(ROOT / 'src'))
from functions import add_stat_annotation, convolveandplot, plot_decoder, plot_decoder_single, plot_decoder_shuffle, new_convolve, plotsingledelay
import functions as plots

save_path = str(FIGURES_OUT / 'fig_5_ephys_repl') + '/'
path = str(DATA_DIR / 'fig_5_ephys_repl') + '/'

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
fig = plt.figure(figsize=(15*cm, 21*cm))
gs = gridspec.GridSpec(nrows=7, ncols=3, figure=fig)

# Create the subplots
a1 = fig.add_subplot(gs[0, 0])
a2  = fig.add_subplot(gs[0, 1])

b1 = fig.add_subplot(gs[1, 0])
b2  = fig.add_subplot(gs[1, 1])

h1 = fig.add_subplot(gs[2, 0])
h2  = fig.add_subplot(gs[2, 1])

i1 = fig.add_subplot(gs[3, 2])
j1 = fig.add_subplot(gs[4, 2])

c1 = fig.add_subplot(gs[3, 0])
c2  = fig.add_subplot(gs[4, 0])
c3 = fig.add_subplot(gs[5, 0])

d1 = fig.add_subplot(gs[3, 1])
d2  = fig.add_subplot(gs[4, 1])
d3  = fig.add_subplot(gs[5, 1])

f = fig.add_subplot(gs[0, 2])

e1  = fig.add_subplot(gs[6, 0])
e2  = fig.add_subplot(gs[6, 1])

fig.text(0.01, 0.99, 'a', fontsize=10, fontweight='bold', va='top')
fig.text(0.34, 0.99, 'b', fontsize=10, fontweight='bold', va='top')
fig.text(0.01, 0.86, 'c', fontsize=10, fontweight='bold', va='top')
fig.text(0.68, 0.99, 'd', fontsize=10, fontweight='bold', va='top')
fig.text(0.01, 0.72, 'e', fontsize=10, fontweight='bold', va='top')
fig.text(0.01, 0.58, 'f', fontsize=10, fontweight='bold', va='top')
fig.text(0.68, 0.58, 'g', fontsize=10, fontweight='bold', va='top')
fig.text(0.01, 0.2, 'h', fontsize=10, fontweight='bold', va='top')
fig.text(0.01, 0.2, 'i', fontsize=10, fontweight='bold', va='top')

# ---------------------------------------------------------------------------
# Panels a/b — cross-decoder: stimulus-aligned (a1) and response-aligned (a2)
# ---------------------------------------------------------------------------

os.chdir(path)
# file_name = 'RLandWM_roll0_stimulusaligned_L2_lbfgs'
file_name = 'RLandWM_roll0.6_stimulus_L1_definitive_correct'

df_cum_sti = pd.read_csv(path+file_name+'_sti.csv', index_col=0)

scores = df_cum_sti.groupby('session').score.mean().reset_index()
list_exclude = scores.loc[scores.score<0.55].session.unique()
df_cum_sti = df_cum_sti[~df_cum_sti['session'].isin(list_exclude)]

#Variables for testing
# colors=['darkgreen','crimson','indigo', 'purple']
# variables = ['WM_roll','WM_roll','RL_roll','RL_roll']
# hits = [1,0,1,0]
# ratios = [0.6,0.6,0.4,0.4]
# variables_combined=[variables[0]+'_'+str(hits[0]),variables[1]+'_'+str(hits[1]),variables[2]+'_'+str(hits[2]),
#                    variables[3]+'_'+str(hits[3])]

#Variables for testing
colors=['darkgreen','indigo']
variables = ['WM_roll','RL_roll']
hits = [1,1]
ratios = [0.6,0.4]
variables_combined=[variables[0]+'_'+str(hits[0]),variables[1]+'_'+str(hits[1])]

plot_decoder([a1,a1,a1,a1], df_cum_sti,baseline=0.0,upper_limit = 0.4,  show_axis=True, individual_sessions=False, align='Stimulus_ON', colors=colors, variables_combined=variables_combined)
a1.set_xlim(-1.5,1)
a1.locator_params(axis='x', nbins=4)

os.chdir(path)
# file_name = 'RLandWM_roll0_delayoffaligned_L2_lbfgs'
# file_name = 'RLandWM_roll0.6_response_L1_definitive_correct'
file_name = 'RLandWM_roll0.6_response_V9'
# file_name = 'RLandWM_roll0.6_delay_V2'

df_cum_res= pd.read_csv(file_name+'_res.csv', index_col=0)

scores = df_cum_res.groupby('session').score.mean().reset_index()
list_exclude = scores.loc[scores.score<0.6].session.unique()
df_cum_res = df_cum_res[~df_cum_res['session'].isin(list_exclude)]

plot_decoder([a2,a2,a2,a2], df_cum_res,baseline=0.0, upper_limit = 0.4, show_axis=False, individual_sessions=False, align='Delay_OFF', colors=colors, variables_combined=variables_combined)
a2.set_xlim(-1,2.)
a2.locator_params(axis='x', nbins=4)



# ---------------------------------------------------------------------------
# Panel c — cross-decoder during delay period (b1: stimulus, b2: response)
# ---------------------------------------------------------------------------

# save_path = 'C:/Users/Tiffany/Google Drive/WORKING_MEMORY/PAPER/Figures/'
# os.chdir(save_path)
# file_name = 'RLandWM_roll0_stimulus_L2_lbfgs_2'
# df_cum_sti = pd.read_csv(file_name+'_sti.csv', index_col=0)
# df_cum_res= pd.read_csv(file_name+'_res.csv', index_col=0)

# #Variables for testing
# colors=['darkgreen','crimson','indigo', 'purple']
# variables = ['WM_roll','WM_roll','RL_roll','RL_roll']
# hits = [1,0,1,0]
# ratios = [0.6,0.6,0.4,0.4]
# variables_combined=[variables[0]+'_'+str(hits[0]),variables[1]+'_'+str(hits[1]),variables[2]+'_'+str(hits[2]),
#                    variables[3]+'_'+str(hits[3])]

# plot_decoder_shuffle([d1,d1,d1,d1], df_cum_sti,baseline=0.5,upper_limit=0.3, align='Stimulus_ON', colors=colors, variables_combined=variables_combined)
# plot_decoder_shuffle([d2,d2,d2,d2], df_cum_res,baseline=0.5,upper_limit=0.3, align='Stimulus_ON', colors=colors, variables_combined=variables_combined)

# ---------------------------------------------------------------------------
# Panel c cont. — delay decoder data loading
# ---------------------------------------------------------------------------

score='score'
y_lower = 0
y_upper = 0

os.chdir(path)
file_name = 'RLandWM_roll0.6_delay_V5'

scores = df_cum_sti.groupby('session').score.mean().reset_index()
list_exclude = scores.loc[scores.score<0.55].session.unique()
df_cum_sti = df_cum_sti[~df_cum_sti['session'].isin(list_exclude)]

scores = df_cum_res.groupby('session').score.mean().reset_index()
list_exclude = scores.loc[scores.score<0.55].session.unique()
df_cum_res = df_cum_res[~df_cum_res['session'].isin(list_exclude)]

df_cum_sti = pd.read_csv(file_name+'_sti.csv', index_col=0)
df_cum_res= pd.read_csv(file_name+'_res.csv', index_col=0)

#Variables for testing
colors=['darkgreen','indigo']
variables = ['WM_roll','RL_roll']
hits = [1,1]
ratios = [0.6,0.4]
variables_combined=[variables[0]+'_'+str(hits[0]),variables[1]+'_'+str(hits[1])]

plot_decoder([b1,b1,b1,b1], df_cum_sti,baseline=0.0,upper_limit=0.3, align='Stimulus_ON', colors=colors, variables_combined=variables_combined)
plot_decoder([b2,b2,b2,b2], df_cum_res,baseline=0.0,upper_limit=0.3, show_axis = False, align='Delay_OFF', colors=colors, variables_combined=variables_combined)

b1.set_xlim(-1.5,1)
b2.set_xlim(-1,2)

# ---------------------------------------------------------------------------
# Panel e — example session cross-decoder (E20 2022-02-26; h1: stim, h2: response)
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
    right.set_xlabel('Time from response onset (s)')
    left.set_xlabel('Time from stimulus onset (s)')
    left.set_ylabel('Decoding\n accuracy')
    left.set_xlim(-1.5,1)
    right.set_xlim(-1,2)

    right.set_title('Mouse E20 2022-02-26')

# ---------------------------------------------------------------------------
# Panel d — log-odds by state and epoch (subplot f; mixed-effects model)
# ---------------------------------------------------------------------------

file_name = 'logodds_WM1_RL1'

df_final = pd.read_csv(path+file_name+'.csv', index_col=0)

# scores = df_final.groupby('session').score.mean().reset_index()
# list_exclude = scores.loc[scores.score<0.55].session.unique()
# df_final = df_final[~df_final['session'].isin(list_exclude)]

# df_results = df_final.loc[(df_final.trial_type=='WM_roll_1')|(df_final.trial_type=='WM_roll_0')].groupby(['session', 'trial_type','epoch']).logs.mean()
# df_results = df_results.reset_index()

# plot = pd.DataFrame({'Correct WM early': df_results.loc[(df_results.trial_type == 'WM_roll_1')&(df_results.epoch == 'early')].logs.values,
#                      'Incorrect WM early':  df_results.loc[(df_results.trial_type == 'WM_roll_0')&(df_results.epoch == 'early')].logs.values,
#                     'Correct WM late': df_results.loc[(df_results.trial_type == 'WM_roll_1')&(df_results.epoch == 'late')].logs.values,
#                     'Incorrect WM late':  df_results.loc[(df_results.trial_type == 'WM_roll_0')&(df_results.epoch == 'late')].logs.values})

panel=f
# sns.violinplot(data=plot, palette=['darkgreen', 'crimson', 'indigo', 'purple' ], width=1,saturation=0.6,linewidth=0, ax=panel)
# sns.violinplot(data=plot, palette=['darkgreen', 'crimson', 'indigo', 'purple' ], width=1,linewidth=1, ax=panel)

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

panel.set_ylim(-1,3)
panel.hlines(xmin=-0.5, xmax=1.5, y=0, linestyle=':')
panel.set_ylabel('Log odds')

with localconverter(ro.default_converter + pandas2ri.converter):
    r_df = ro.conversion.py2rpy(df_final)

# formula="logs ~ state*epoch + (1|session)"
# formula="logs ~ state*epoch + (1|session)+ (1|fold)"
# formula="logs ~ state*epoch + (state+epoch+1|session) + (state+epoch+1|fold)"
formula="log_odds ~ state*epoch + (state+epoch+1|session:fold)"

model = lme4.lmer(formula, data=r_df)

for i, v in enumerate(list(base.summary(model).names)):
    if v in ['coefficients']:
        print (base.summary(model).rx2(v))
# print(base.summary(model))
print(car.Anova(model))


# ---------------------------------------------------------------------------
# Panel f — single trial population activity (E17); WM trial: c1/c2/c3
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
# c1.set_subtitle('Mouse E17 2022-01-31', fontsize=8)

# Panel f cont. — RepL trial (d1/d2/d3)
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
# Panel g — example neuron PSTH: WM vs HB trials (i1: raster, j1: PSTH)
# ---------------------------------------------------------------------------


file_name = 'WMvsHB_example_138'
df = pd.read_csv(path+file_name+'.csv')

delay = 10
colors = [COLORRIGHT,COLORLEFT]
labels = ['Right stimulus','Left stimulus']
align = 'Stimulus_ON'

# temp_df = df.loc[(df.hit ==1)&(df.cluster_id == 138)]
temp_df = df.loc[(df.cluster_id == 138)]
temp_df['state'] = np.where(temp_df['WM_roll'] > 0.6, 1, 0)

with PdfPages(path +  filename[:-4]+'_population.pdf') as pdf:
    for cluster_id in temp_df.cluster_id.unique():
        print(cluster_id)
        j=1
        j = convolveandplot(temp_df.loc[(temp_df.vector_answer == 0)&(temp_df.hit == 1)], i1, j1, variable='state', cluster_id = cluster_id, delay = delay, j=j,
                           labels=['WM left','HB left'], colors=['darkgreen', 'indigo'], kernel=200, add_state=True)

        # j = convolveandplot(temp_df.loc[(temp_df.vector_answer == 0)&(temp_df.hit == 0)], i1, j1, variable='state', cluster_id = cluster_id, delay = delay, j=j,
        #                    labels=['WM left','HB left'], colors=['lightgrey', 'lightgrey'], kernel=200)

        i1.set_ylim(0,19)


# ---------------------------------------------------------------------------
# Panels h/i — previous trial decoder (e1: stimulus, e2: response)
# ---------------------------------------------------------------------------

#Variables for testing
colors=['darkgreen','indigo']
variables = ['WM_roll','RL_roll']
hits = [1,1]
ratios = [0.6,0.4]
variables_combined=[variables[0]+'_'+str(hits[0]),variables[1]+'_'+str(hits[1])]

file_name = 'trainedall_testedRL_previous_vector_answer_after_correct_V11'
df_cum_res= pd.read_csv(path+file_name+'_res.csv', index_col=0)
df_cum_sti= pd.read_csv(path+file_name+'_sti.csv', index_col=0)
trained_trials = df_cum_sti.session.unique()

df_cum_res = df_cum_res[df_cum_res['session'].isin(trained_trials[:-1])]
df_cum_sti = df_cum_sti[df_cum_sti['session'].isin(trained_trials[:-1])]

file_name = 'trainedall_testedRL_previous_vector_answer_shuffle_V11'
df_cum_res_shuffle = pd.read_csv(path+file_name+'_res.csv', index_col=0)
df_cum_sti_shuffle = pd.read_csv(path+file_name+'_sti.csv', index_col=0)

df_cum_res_shuffle = df_cum_res_shuffle[df_cum_res_shuffle['session'].isin(trained_trials)]
df_cum_sti_shuffle = df_cum_sti_shuffle[df_cum_sti_shuffle['session'].isin(trained_trials)]

plots.plot_results_shuffle_substraction(df_cum_sti, df_cum_res, df_cum_sti_shuffle, df_cum_res_shuffle, ['indigo'], variables_combined, fig = True, ax1=e1, ax2=e2)

file_name = 'trainedall_testedWM_previous_vector_answer_shuffle_V11'
df_cum_res_shuffle = pd.read_csv(path+file_name+'_res.csv', index_col=0)
df_cum_sti_shuffle = pd.read_csv(path+file_name+'_sti.csv', index_col=0)

df_cum_res_shuffle = df_cum_res_shuffle[df_cum_res_shuffle['session'].isin(trained_trials)]
df_cum_sti_shuffle = df_cum_sti_shuffle[df_cum_sti_shuffle['session'].isin(trained_trials)]

file_name = 'trainedall_testedWM_previous_vector_answer_after_correct_V11'
df_cum_res= pd.read_csv(path+file_name+'_res.csv', index_col=0)
df_cum_sti= pd.read_csv(path+file_name+'_sti.csv', index_col=0)

df_cum_res = df_cum_res[df_cum_res['session'].isin(trained_trials)]
df_cum_sti = df_cum_sti[df_cum_sti['session'].isin(trained_trials)]

plots.plot_results_shuffle_substraction(df_cum_sti, df_cum_res, df_cum_sti_shuffle, df_cum_res_shuffle, ['darkgreen'], variables_combined, fig = True, ax1=e1, ax2=e2)

e1.set_ylim(-0.1,0.4)
e2.set_ylim(-0.1,0.4)

# ----------------------------------------------------------------------------------------------------------------------------

# Show the figure
sns.despine()
plt.subplots_adjust(left=0.1,
                    bottom=0.07,
                    right=0.95,
                    top=0.97,
                    wspace=0.5,
                    hspace=0.5)

save_path = str(FIGURES_OUT / 'fig_5_ephys_repl') + '/'
# plt.savefig(save_path+'/Fig. 5. Ephys HB.svg', bbox_inches='tight',dpi=300)
# plt.savefig(save_path+'/Fig. 5. Ephys HB.png', bbox_inches='tight',dpi=300)

plt.show()
