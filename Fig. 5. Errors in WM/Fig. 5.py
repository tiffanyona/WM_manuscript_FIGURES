# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 12:06:44 2022

@author: Tiffany
"""
COLORLEFT = 'teal'
COLORRIGHT = '#FF8D3F'

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.anova import AnovaRM
from statsmodels.graphics.factorplots import interaction_plot
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib as mpl
import os
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from scipy import special
import json 
from sklearn.linear_model import LogisticRegression
from scipy.optimize import curve_fit
#Import all needed libraries
from matplotlib.lines import Line2D
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from matplotlib.backends.backend_pdf import PdfPages
from statannot import add_stat_annotation
from neo.core import SpikeTrain
from quantities import ms, s, Hz
from elephant.statistics import mean_firing_rate
from elephant.statistics import time_histogram, instantaneous_rate
from elephant.kernels import GaussianKernel
from elephant.statistics import mean_firing_rate
from cycler import cycler
import scipy.io

from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
import statsmodels.formula.api as smf

pandas2ri.activate()

base     = importr('base')
car      = importr('car')
Rstats    = importr('stats')
lme4     = importr('lme4')
scales   = importr('scales')
lmerTest = importr('lmerTest')


# utilities = 'C:/Users/Tiffany/Google Drive/WORKING_MEMORY/PAPER/WM_manuscript_FIGURES/'
# os.chdir(utilities)
# import functions as plots

utilities = 'G:/Mi unidad/WORKING_MEMORY/PAPER/WM_manuscript_FIGURES/'
os.chdir(utilities)
import functions as plots

save_path = 'G:/Mi unidad/WORKING_MEMORY/PAPER/WM_manuscript_FIGURES/Fig. 5. Errors in WM/'
path = 'G:/Mi unidad/WORKING_MEMORY/PAPER/ANALYSIS_figures/'
os.chdir(path)

cm = 1/2.54
sns.set_context('paper', rc={'axes.labelsize': 7,
                            'lines.linewidth': 1, 
                            'lines.markersize': 2, 
                            'legend.fontsize': 7,  
                            'xtick.major.size': 1,
                            'xtick.labelsize': 6, 
                            'ytick.major.size': 1, 
                            'ytick.labelsize': 6,
                            'xtick.major.pad': 0,
                            'ytick.major.pad': 0,
                            'xlabel.labelpad': -10})

# Create a figure with 6 subplots using a GridSpec
fig = plt.figure(figsize=(12*cm, 18*cm))
gs = gridspec.GridSpec(nrows=6, ncols=2, figure=fig)

# Create the subplots
a1 = fig.add_subplot(gs[3, 1:2])
a2 = fig.add_subplot(gs[4, 1:2])
a3 = fig.add_subplot(gs[5, 1:2])

b1 = fig.add_subplot(gs[3, 0:1])
b2 = fig.add_subplot(gs[4, 0:1])
b3 = fig.add_subplot(gs[5, 0:1])

# Trained weights plots
g2 = fig.add_subplot(gs[0, 1:2])

j0 = fig.add_subplot(gs[2, 0:1])
j1 = fig.add_subplot(gs[0, 0:1])
j2 = fig.add_subplot(gs[1, 0:1])

j3 = fig.add_subplot(gs[1, 1:2])
k = fig.add_subplot(gs[2, 1:2])


fig.text(0.01, 0.99, 'a', fontsize=10, fontweight='bold', va='top')
fig.text(0.5, 0.99, 'd', fontsize=10, fontweight='bold', va='top')
fig.text(0.01, 0.82, 'b', fontsize=10, fontweight='bold', va='top')
fig.text(0.01, 0.67, 'c', fontsize=10, fontweight='bold', va='top')
fig.text(0.5, 0.82, 'e', fontsize=10, fontweight='bold', va='top')
fig.text(0.5, 0.67, 'f', fontsize=10, fontweight='bold', va='top')
fig.text(0.01, 0.52, 'g', fontsize=10, fontweight='bold', va='top')
fig.text(0.5, 0.52, 'h', fontsize=10, fontweight='bold', va='top')


# ------############################## J Panel -- WM error trials individual example for stimulus #################################-----------------------
# path = 'C:/Users/Tiffany/Google Drive/WORKING_MEMORY/PAPER/ANALYSIS_figures/'
os.chdir(path)
colors=['crimson','darkgreen']
variables = ['WM_roll','WM_roll']
hits = [0,1]
variables_combined=[variables[0]+'_'+str(hits[0]),variables[1]+'_'+str(hits[1])]
file_name = '/single_delay_WM_roll0.6_stimulus_0.25_V1'
df_cum_sti = pd.read_csv(path+file_name+'.csv', index_col=0)

scores = df_cum_sti.loc[df_cum_sti.delay == 10].groupby('session').score.mean().reset_index()
list_exclude = scores.loc[scores.score<0.60].session.unique()
df_cum_sti = df_cum_sti.loc[df_cum_sti.delay == 10][~df_cum_sti['session'].isin(list_exclude)] 


delay=10
panel=j1
plots.plotsingledelay(df_cum_sti, panel, colors, variables_combined, delay, baseline=0, invert_list=[True, False])
panel.set_xlim(-2,14)

# ------################################################################################### ######-----------------------
# ------############################## K Panel -- WM error trials  for response #################################-----------------------

os.chdir(path)
file_name = '/single_delay_WM_roll0.6_lick_0.25_V1'
df_cum_sti = pd.read_csv(path+file_name+'.csv', index_col=0)

scores = df_cum_sti.loc[df_cum_sti.delay == 10].groupby('session').score.mean().reset_index()
list_exclude = scores.loc[scores.score<0.60].session.unique()
df_cum_sti = df_cum_sti.loc[df_cum_sti.delay == 10][~df_cum_sti['session'].isin(list_exclude)] 

panel=j2
plots.plotsingledelay(df_cum_sti, panel, colors, variables_combined, delay, baseline=0)
panel.set_xlim(-2,14)

# ------#########################################################################################-----------------------
# ------############################## K Panel -- WM error trials for delay #################################-----------------------

os.chdir(path)
file_name = '/final_WMcorrect_WMincorrect'
df_cum_sti = pd.read_csv(path+file_name+'.csv', index_col=0)

panel=j0
plots.plotsingledelay(df_cum_sti, panel, colors, variables_combined, delay, baseline=0)
panel.set_xlim(-2,14)

# ------#########################################################################################-----------------------

# ------############################## J Panel -- WM error trials individual example #################################-----------------------

# delays = [10]
# path = 'C:/Users/Tiffany/Google Drive/WORKING_MEMORY/PAPER/Figures/'
# file_name = 'single_delay_WM_roll0.6_0.5_folded5_overlap0.25'
# df_cum_sti = pd.read_csv(path+file_name+'.csv', index_col=0)
# df_cum_shuffle = pd.read_csv(path+file_name+'_shuffle.csv', index_col=0)

# panel = j3
# for delay in delays:
#     # filename='E20_2022-02-27_17-02-17.csv'
#     filename="E20_2022-02-14_16-01-30.csv"
#     df_sti = df_cum_sti.loc[(df_cum_sti.session ==filename)&(df_cum_sti.delay == delay)]
#     df_iter = df_cum_shuffle.loc[(df_cum_shuffle.session ==filename)&(df_cum_shuffle.delay == delay)]

#     for color, variable in zip(colors,variables_combined):
#         real = np.array(np.mean(df_sti.loc[(df_sti['trial_type'] == variable)].groupby('session').median().drop(columns=['fold','delay','score']).dropna(axis=1)))
#         times = df_sti.loc[(df_sti['trial_type'] == variable)].dropna(axis=1)
#         times = np.array(times.drop(columns=['fold','score','trial_type', 'delay','session'],axis = 1).columns.astype(float))

#         df_new = df_iter.loc[(df_iter.trial_type==variable)].drop(columns=['fold', 'delay','session']).groupby('times').mean()
#         y_mean = df_new.mean(axis=1).values
#         lower =  df_new.quantile(q=0.975, interpolation='linear',axis=1).values-y_mean
#         upper =  df_new.quantile(q=0.025, interpolation='linear',axis=1).values-y_mean
#         x = times

#         panel.set_xlabel('Time to stimulus onset (s)')
#         try:
#             panel.plot(times,real, color=color)
#             panel.plot(x, lower+real, color=color, linestyle = '',alpha=0.6, linewidth=0)
#             panel.plot(x, upper+real, color=color, linestyle = '',alpha=0.6, linewidth=0)
#             panel.fill_between(x, lower+real, upper+real, alpha=0.2, color=color, linewidth=0)
#             panel.set_ylim(0.1,1)
#             panel.axhline(y=0.5,linestyle=':',color='black')
#             panel.fill_betweenx(np.arange(0.1,1.15,0.1), 0,0.4, color='lightgrey', alpha=1, linewidth=0)
#             panel.fill_betweenx(np.arange(0.1,1.1,0.1), delay+0.3,delay+0.5, color='grey', alpha=.5, linewidth=0)
#         except:
#             print('not this condition for this delay')
#             continue
# panel.set_xlim(-2,14)
# # 

delays = [10]
# path = 'C:/Users/Tiffany/Google Drive/WORKING_MEMORY/PAPER/ANALYSIS_figures/'
file_name = '\WMcorrectincorrect_10s_session_example'
df_cum_sti = pd.read_csv(path+file_name+'.csv', index_col=0)
df_cum_shuffle = pd.read_csv(path+file_name+'_shuffle.csv', index_col=0)

panel = j3
for delay in delays:
    # filename='E20_2022-02-27_17-02-17.csv'
    df_sti = df_cum_sti.loc[(df_cum_sti.delay == delay)]
    df_iter = df_cum_shuffle.loc[(df_cum_shuffle.delay == delay)]

    for color, variable in zip(colors,variables_combined):
        real = np.array(np.mean(df_sti.loc[(df_sti['trial_type'] == variable)].groupby('session').median().drop(columns=['fold','delay','score']).dropna(axis=1)))
        times = df_sti.loc[(df_sti['trial_type'] == variable)].dropna(axis=1)
        times = np.array(times.drop(columns=['fold','score','trial_type', 'delay','session'],axis = 1).columns.astype(float))

        df_new = pd.DataFrame()
        for iteration in np.arange(1,50):
            df_new[iteration]= df_iter.loc[(df_iter.trial_type==variable)].groupby('times').mean()[str(float(iteration))]
        
        y_mean = df_new.mean(axis=1).values
        lower =  df_new.quantile(q=0.975, interpolation='linear',axis=1).values-y_mean
        upper =  df_new.quantile(q=0.025, interpolation='linear',axis=1).values-y_mean
        x = times

        panel.set_xlabel('Testing time from stimulus onset (s)')
        try:
            panel.plot(times,real, color=color)
            panel.plot(x, lower+real, color=color, linestyle = '',alpha=0.6, linewidth=0)
            panel.plot(x, upper+real, color=color, linestyle = '',alpha=0.6, linewidth=0)
            panel.fill_between(x, lower+real, upper+real, alpha=0.2, color=color, linewidth=0)
            panel.set_ylim(0.1,1)
            panel.axhline(y=0.5,linestyle=':',color='black')
            panel.fill_betweenx(np.arange(0.1,1.15,0.1), 0,0.4, color='lightgrey', alpha=1, linewidth=0)
            panel.fill_betweenx(np.arange(0.1,1.1,0.1), delay+0.3,delay+0.5, color='grey', alpha=.5, linewidth=0)
            panel.set_xlim(-2,14)
        except:
            print('not this condition for this delay')
            continue

# ------####################################### Example neuron for correct and incorrect trials ##################################-----------------------


file_name = '\single_neuron_10s'
df = pd.read_csv(path+file_name+'.csv', index_col=0)

delay = 10
cluster_id = 153

colors = [COLORRIGHT,COLORLEFT]
# colors = ['darkgreen','crimson']
labels = ['Right stimulus','Left stimulus']
# labels = ['Correct','Incorrect']

align = 'Stimulus_ON'

print('First correct')
j=1
temp_df = df.loc[(df.WM_roll >0.6)&(df.hit ==1)]
j = plots.convolveandplot(temp_df, k, k, variable='reward_side', cluster_id = cluster_id, delay = delay, j=j, alpha=0.3,colors = ['grey','grey'], spikes = False)

temp_df = df.loc[(df.WM_roll >0.6)&(df.hit ==0)]
j = plots.convolveandplot(temp_df, k, k, variable='reward_side', cluster_id = cluster_id, delay = delay, j=j, 
                   labels=['Incorrect right stimulus','Incorrect left stimulus'], spikes = False, kernel=100)
k.set_xlim(-2,14)

# ------#####################################   Comparing early delay with alte one for correct and incorrect WM    ####################################################-----------------------
            
file_name = 'logodds_WM1_WM0'

df_final = pd.read_csv(path+file_name+'.csv', index_col = 0)
# df_final = df_final[~df_final['session'].isin(list_exclude)] 


# df_results = df_final.loc[(df_final.trial_type=='WM_roll_1')|(df_final.trial_type=='WM_roll_0')].groupby(['session', 'trial_type','epoch']).logs.mean()
# df_results = df_results.reset_index()

# plot = pd.DataFrame({'Correct WM early': df_results.loc[(df_results.trial_type == 'WM_roll_1')&(df_results.epoch == 'early')].logs.values, 
#                      'Incorrect WM early':  df_results.loc[(df_results.trial_type == 'WM_roll_0')&(df_results.epoch == 'early')].logs.values,
#                     'Correct WM late': df_results.loc[(df_results.trial_type == 'WM_roll_1')&(df_results.epoch == 'late')].logs.values, 
#                     'Incorrect WM late':  df_results.loc[(df_results.trial_type == 'WM_roll_0')&(df_results.epoch == 'late')].logs.values})

panel=g2
df_results = df_final.groupby(['session', 'trial_type','epoch']).log_odds.mean()
df_results = df_results.reset_index()

sns.boxplot(x='trial_type', y='log_odds',hue='epoch', order=['WM_roll_1','WM_roll_0'], showcaps=False, showfliers=False, 
            palette=['darkgreen','lightgreen', 'darkred', 'lightred'], linewidth=0 , ax = panel, data=df_results, width = 0.)
sns.boxplot(x='trial_type', y='log_odds',hue='epoch', order=['WM_roll_1','WM_roll_0'], showcaps=False, showfliers=False,
            palette=['darkgreen','lightgreen', 'darkred', 'lightred'], medianprops=dict(color="white"),
            linewidth=1, ax = panel, data=df_results, showmeans=True, width = 0.5)

df_plots = df_results.loc[(df_results.trial_type == 'WM_roll_1')&(df_results.epoch == 'early')]
xA = np.random.normal(-0.25,  0.05, len(df_plots))
sns.scatterplot(x=xA,y='log_odds',data=df_plots,ax=panel,alpha=0.9,legend=False, color='darkgreen')

df_plots = df_results.loc[(df_results.trial_type == 'WM_roll_1')&(df_results.epoch == 'late')]
xA = np.random.normal(0.25,  0.05, len(df_plots))
sns.scatterplot(x=xA,y='log_odds',data=df_plots,ax=panel,alpha=0.5,legend=False, color='darkgreen')

df_plots = df_results.loc[(df_results.trial_type == 'WM_roll_0')&(df_results.epoch == 'early')]
xA = np.random.normal(.75,  0.05, len(df_plots))
sns.scatterplot(x=xA,y='log_odds',data=df_plots,ax=panel,alpha=0.9,legend=False, color='crimson')

df_plots = df_results.loc[(df_results.trial_type == 'WM_roll_0')&(df_results.epoch == 'late')]
xA = np.random.normal(1.25, 0.05, len(df_plots))
sns.scatterplot(x=xA,y='log_odds',data=df_plots,ax=panel,alpha=0.5,legend=False, color='crimson')

panel.set_ylim(-3,5)
panel.hlines(xmin=-0.5, xmax=1.5, y=0, linestyle=':')
panel.set_ylabel('Log odds')


r_df  = ro.conversion.py2ri(df_final)

# formula="logs ~ state*epoch + (1|session)"
# formula="logs ~ state*epoch + (1|session)+ (1|fold)"
# formula="logs ~ state*epoch + (state+epoch+1|session) + (state+epoch+1|fold)"
formula="log_odds ~ hit*epoch + (hit+epoch+1|session:fold)"

model = lme4.lmer(formula, data=r_df)

for i, v in enumerate(list(base.summary(model).names)):
    if v in ['coefficients']:
        print (base.summary(model).rx2(v))
# print(base.summary(model))
print(car.Anova(model))

data_group1 = df_final.loc[(df_final.trial_type == 'WM_roll_0')&(df_final.epoch == 'early')].groupby('session').log_odds.mean().values
data_group2 = df_final.loc[(df_final.trial_type == 'WM_roll_0')&(df_final.epoch == 'late')].groupby('session').log_odds.mean().values
print(stats.ttest_1samp(data_group1, 0))
print(stats.ttest_1samp(data_group2, 0))

print(scipy.stats.wilcoxon(df_final.loc[(df_final.trial_type == 'WM_roll_0')&(df_final.epoch == 'early')].groupby('session')['log_odds'].mean(), alternative='less'))
scipy.stats.wilcoxon(df_final.loc[(df_final.trial_type == 'WM_roll_0')&(df_final.epoch == 'late')].groupby('session')['log_odds'].mean())

print(scipy.stats.wilcoxon(df_final.loc[(df_final.trial_type == 'WM_roll_1')&(df_final.epoch == 'early')].groupby('session')['log_odds'].mean(), alternative='less'))
scipy.stats.wilcoxon(df_final.loc[(df_final.trial_type == 'WM_roll_1')&(df_final.epoch == 'late')].groupby('session')['log_odds'].mean())

# scipy.stats.ttest_rel(df_final.loc[(df_final.trial_type == 'WM_roll_1')&(df_final.epoch == 'early')].groupby('session')['log_odds'].mean().values,
# df_final.loc[(df_final.trial_type == 'WM_roll_1')&(df_final.epoch == 'late')].groupby('session')['log_odds'].mean().values)

# print(scipy.stats.wilcoxon(df_final.loc[(df_final.trial_type == 'WM_roll_0')&(df_final.epoch == 'early')].groupby('session')['log_odds'].mean(),
# df_final.loc[(df_final.trial_type == 'WM_roll_0')&(df_final.epoch == 'late')].groupby('session')['log_odds'].mean()))

# -----------------############################## Example trial correct  #######################-----------------------

T=83
filename = 'E17_2022-02-02_17-13-06.csv'

file_name = 'decoder_'+str(T)+'_'+filename
df_decoder =pd.read_csv(path+file_name, index_col = 0)

file_name = 'df_'+str(T)+'_'+filename
df = pd.read_csv(path+file_name, index_col = 0)

file_name = '/convolve_'+str(T)+'_'+filename
big_data = pd.read_csv(path+file_name, index_col = 0)

plots.single_trial_with_decoder(df, df_decoder, big_data, filename, T, panels = [a1,a2,a3])

# ------#########################################################################################-----------------------

T=185

file_name = 'decoder_'+str(T)+'_'+filename
df_decoder =pd.read_csv(path+file_name, index_col = 0)

file_name = 'df_'+str(T)+'_'+filename
df = pd.read_csv(path+file_name, index_col = 0)
file_name = 'convolve_'+str(T)+'_'+filename
big_data = pd.read_csv(path+file_name, index_col = 0)

plots.single_trial_with_decoder(df, df_decoder, big_data, filename, T, panels = [b1,b2,b3], show_y = True)
# a1.set_ylim(0,11.5)
# b2.set_ylim(-0,15)
# a2.set_ylim(-0,15)
# b3.set_ylim(-10,10)
# a3.set_ylim(-10,10)

# ------#########################################################################################-----------------------


# ------#########################################################################################-----------------------

# Show the figure
plt.subplots_adjust(left=0.07,
                    bottom=0.07,
                    right=0.97,
                    top=0.97,
                    wspace=0.5,
                    hspace=0.5)
sns.despine()
# plt.savefig(save_path+'/Fig 5_WM errors_V4.svg', bbox_inches='tight',dpi=300)
plt.show()