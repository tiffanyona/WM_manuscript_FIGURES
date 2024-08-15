# # -*- coding: utf-8 -*-
# """
# Created on Mon Jan  9 17:27:02 2023

# @author: Tiffany
# """

# # -*- coding: utf-8 -*-
# """
# Created on Mon Jan  9 17:17:53 2023

# @author: Tiffany
# """

COLORLEFT = 'teal'
COLORRIGHT = '#FF8D3F'
# COLORLEFT = 'crimson'
# COLORRIGHT = 'darkgreen'

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
from datahandler import Utils
from neo.core import SpikeTrain
from quantities import ms, s, Hz
from elephant.statistics import mean_firing_rate
from elephant.statistics import time_histogram, instantaneous_rate
from elephant.kernels import GaussianKernel
from elephant.statistics import mean_firing_rate
from cycler import cycler


# y_lower = 0
# y_upper=0

# colors=['crimson','darkgreen']
# variables = ['WM_roll','WM_roll']
# hits = [0,1]
# ratios = [0.65,0.65]
# variables_combined=[variables[0]+'_'+str(hits[0]),variables[1]+'_'+str(hits[1])]

# save_path = 'C:/Users/Tiffany/Google Drive/WORKING_MEMORY/PAPER/Figures/'
# os.chdir(save_path)
# # file_name = 'single_delay_WM_roll0.65_0.5_4'
# file_name = 'single_delay_WM_roll0.6_0.5_2_folded3'
# df_cum_sti = pd.read_csv(file_name+'.csv', index_col=0)

# # delays = [0.1,1,3,10]
# delays = [1,3]

# scores = df_cum_sti.groupby('session').score.mean().reset_index()
# list_exclude = scores.loc[scores.score<0.55].session.unique()
# df_cum_sti = df_cum_sti[~df_cum_sti['session'].isin(list_exclude)] 

# for delay in delays:
#     fig, ax1 = plt.subplots(1,1, figsize=(10, 4), sharey=True)
                
#     for color, variable in zip(colors,variables_combined):
#         print(variable)

#         # Aligmnent for Stimulus cue
#         real = np.array(df_cum_sti.loc[(df_cum_sti['trial_type'] == variable)&(df_cum_sti['delay'] == delay)].drop(columns=['delay', 'score','fold']).mean(axis=0))
        
#     #     times = np.array(df_cum_sti.loc[:, df_cum_sti.columns != 'trial_type' and df_cum_sti.columns != 'trial_type'].columns).astype(float)
#         times = df_cum_sti.loc[(df_cum_sti['trial_type'] == variable)&(df_cum_sti['delay'] == delay)]
#         times = np.array(times.drop(columns=['trial_type', 'delay','session','delay', 'score','fold'],axis = 1).columns.astype(float))

#         if color=='crimson':
#             ax1.set_xlabel('Time (s) to Cue')
#         sns.despine()

#         df_lower = pd.DataFrame()
#         df_upper = pd.DataFrame()

#         for timepoint in times:
#             mean_surr = []

#             # recover the values for that specific timepoint
#             array = df_cum_sti.loc[(df_cum_sti.trial_type ==variable)&(df_cum_sti['delay'] == delay)].drop(columns='delay')[str(timepoint)].to_numpy()

#             # iterate several times with resampling: chose X time among the same list of values
#             for iteration in range(1000):
    
#                 x = np.random.choice(array, size=len(array), replace=True)
#                 # recover the mean of that new distribution
#                 mean_surr.append(np.mean(x))

#             df_lower.at[0,timepoint] = np.percentile(mean_surr, 2.5)
#             df_upper.at[0,timepoint] = np.percentile(mean_surr, 97.5)

#         lower =  df_lower.iloc[0].values
#         upper =  df_upper.iloc[0].values
#         x=times

#         ax1.plot(times,real, color=color)
#         ax1.plot(x, lower, color=color, linestyle = '',alpha=0.6)
#         ax1.plot(x, upper, color=color, linestyle = '',alpha=0.6)
#         ax1.fill_between(x, lower, upper, alpha=0.2, color=color)
#         if max(upper)>y_upper:
#             y_upper = max(upper)
#         if  min(lower)<y_lower:
#             y_lower = min(lower)
#         ax1.set_ylim(y_lower-0.1,y_upper+0.1)
#         ax1.axhline(y=0.0,linestyle=':',color='black')
#         ax1.fill_betweenx(np.arange(-1,1.15,0.1), 0,0.4, color='grey', alpha=.4)
#         ax1.fill_betweenx(np.arange(-1.1,1.1,0.1), delay+0.4,delay+0.6, color='beige', alpha=.8)
#         if color=='crimson':
#             ax1.set_xlabel('Time (s) to Go')

#         sns.despine()
#         plt.tight_layout()
#         plt.show()

def new_convolve(nx,df):
    '''
    nx = already selected cluster
    df = dataframe from the session with all trials
    '''

    errors_=[] ##indexes and neurons without enough spikes to make a spiketrain
    frames=[]
   
    # Iterate for each trial in that session
    for T in df.trial.unique():
        if T > nx.iloc[0].trial_start+1 and T < nx.iloc[0].trial_end+1:
            # Take the spike times for that trial
            nxt = nx.loc[nx['trial']==T]['fixed_times']

            # !!!! IMPORTANT use the main df that has all the trials. If you use the filtered nx, some trials may not appear if they were no spikes there. 
            dft = df.loc[df['trial']==T]

            # try: 
            ############################################################ Get the times of the spikes
            times_spikes = nxt
            times_spikes = times_spikes*1000 #transform to ms

            ############################################################ Set the strat and end time of the train
            stop_time =  (dft.END.unique()[0])*1000*ms ## End of the trial in ms
            try:
                start_time = (dft.START_adjusted.unique()[0]-0.1)*1000*ms ## Start of the trial in ms    
            except:
                start_time = dft.START.unique()[0]*1000*ms ## Start of the trial in ms   
                    
            ############################################################ Spiketrain
            spiketrain = SpikeTrain(times_spikes, units=ms, t_stop=stop_time, t_start=start_time) 

            ############################################################ Convoluted firing rate
            histogram_rate = time_histogram([spiketrain], 5*ms, output='rate')
            gaus_rate = instantaneous_rate(spiketrain, sampling_period=5*ms, kernel=GaussianKernel(50*ms)) #s.d of Suzuki & Gottlieb 
            times_ = gaus_rate.times.rescale(ms)
            firing = gaus_rate.rescale(histogram_rate.dimensionality).magnitude.flatten()

            ############################################################ Dataframe 
            df_trial = pd.DataFrame({'times':times_, 'firing':firing}) #dataframe con times y firing
            df_trial['trial']=T
            df_trial['Delay_OFF']= dft.Delay_OFF.unique()[0]*1000
            df_trial['Stimulus_ON']= dft.Stimulus_ON.unique()[0]*1000
            df_trial['delay']=dft.delay.unique()[0]
            df_trial['vector_answer']=dft.vector_answer.unique()[0]
            df_trial['reward_side']=dft.reward_side.unique()[0]
            # df_trial['miss']=dft.miss.unique()[0]
            df_trial['hit']=dft.hit.unique()[0]
            df_trial['state']=dft.state.unique()[0]
            
            frames.append(df_trial)
            # except ValueError:
            #     errors_.append([N,T])
            #     print (N, T)
            # except IndexError:
            #     print('Index error, missing trial ' + str(T))
            #     errors_.append([N,T])
        else:
            continue
    neuron = pd.concat(frames)
    return neuron



fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10, 8), sharex=True)

# session = pd.read_csv('C:/Users/Tiffany/Documents/Ephys/summary_complete/E20_2022-02-14_16-01-30.csv', sep=',',index_col=0)
# session['trial_start'] = session.trial.min()
# session['trial_end'] = session.trial.max()

# path = 'C:/Users/Tiffany/Google Drive/WORKING_MEMORY/PAPER/Figures/'
# file_name = 'single_neuron_10s'
# df = pd.read_csv(path+file_name+'.csv', index_col=0)

# align = 'Stimulus_ON'
# delay = 1
# cue_on=0
# cue_off=0.35
# start=-3
# stop=16

# df = df.loc[(df.WM_roll >= 0.6)&(df.hit==1)&(df.delay!=0.1)]
# df['state'] = np.where(df.WM_roll > 0.6, 1, 0)

# variable = 'reward_side'
# neuron = new_convolve(df, df)
# plt.title('WM incorrect')
# conditions = ['Left','Right']
# # conditions = ['WM','RL']
# # neuron = neuron.loc[neuron.delay != 0.1] # Remove trials with no delay if the studied segment is that one. 

# # Align the data to the targeted align
# neuron['time_centered'] = neuron['times'] - neuron[align] 
# neuron['time_centered'] = np.round(neuron.time_centered/1000, 2) #### estos es importante!!
# neuron['firing_'] = neuron['firing']*1000

# df_results = pd.DataFrame(dtype=float)
# # df_results['firing'] = neuron.loc[(neuron.time_centered <= stop)&(neuron.delay == delay)].groupby(['time_centered',variable])['firing_'].mean()
# # df_results['error'] = neuron.loc[(neuron.time_centered <= stop)&(neuron.delay == delay)].groupby(['time_centered',variable])['firing_'].std()
# df_results['firing'] = neuron.loc[(neuron.time_centered <= stop)].groupby(['time_centered',variable])['firing_'].mean()
# df_results['error'] = neuron.loc[(neuron.time_centered <= stop)].groupby(['time_centered',variable])['firing_'].std()

# df_results.reset_index(inplace=True)

# for condition,color,name in zip([1,0],[COLORRIGHT,COLORLEFT],conditions):
#     y_mean = df_results[df_results[variable]==condition].firing
#     error = 0.5 * df_results[df_results[variable]==condition].error
#     lower = y_mean - error
#     upper = y_mean + error
#     x = df_results[df_results[variable]==condition].time_centered

#     panel = ax2
#     panel.plot(x, y_mean, label=name,color=color)
#     panel.plot(x, lower, color=color, alpha=0.1)
#     panel.plot(x, upper, color=color, alpha=0.1)
#     panel.fill_between(x, lower, upper, alpha=0.2, color=color)

# panel.set_xlim(start,stop)  

# panel.set_xlabel('Time (s) from response onset')
# y = np.arange(0,60,0.1)     
# panel.fill_betweenx(y, cue_on,cue_off, color='grey', alpha=.4)  
# panel.fill_betweenx(y, cue_off+delay,cue_off+delay+.2, color='beige', alpha=.6)
# panel.set_ylim(0,60)  

# # axis labels and legend
# panel.legend(frameon=False, bbox_to_anchor=(1.05, 1))    
# panel.set_xlabel('Time (s) from stimulus onset')
# panel.set_ylabel('Firing rate (spikes/s)')
        
# # Plot the rater plot
# # SpikesRight = (df.loc[(df[variable] == 1)&(df.delay == delay)])
# # SpikesLeft = (df.loc[(df[variable] == 0)&(df.delay == delay)])
# SpikesRight = (df.loc[(df[variable] == 1)])
# SpikesLeft = (df.loc[(df[variable] == 0)])

# SpikesRight['a_'+align] = SpikesRight['fixed_times'] - SpikesRight['Stimulus_ON'] 
# SpikesLeft['a_'+align] = SpikesLeft['fixed_times'] - SpikesLeft['Stimulus_ON'] 

# panel = ax1
# trial=1
# j=1
# spikes = []
# trial_repeat = []
# for i in range(len(SpikesRight)):
#     # Plot for licks for left trials
#     if SpikesRight.trial.iloc[i] != trial:
#         panel.plot(spikes,trial_repeat, '|', markersize=1, color=COLORRIGHT, zorder=1)
#         spikes = []
#         trial_repeat = []
#         trial = SpikesRight.trial.iloc[i]
#         j+=1
#     if SpikesRight['a_'+align].iloc[i] > start and SpikesRight['a_'+align].iloc[i] < stop:
#         spikes.append(SpikesRight['a_'+align].iloc[i])
#         trial_repeat.append(j)
#     else:
#         continue

# trial=1
# spikes = []
# trial_repeat = []
# for i in range(len(SpikesLeft)):
#     # Plot for licks for left trials
#     if SpikesLeft.trial.iloc[i] != trial:
#         panel.plot(spikes,trial_repeat, '|', markersize=1, color=COLORLEFT, zorder=1)
#         spikes = []
#         trial_repeat = []
#         trial = SpikesLeft.trial.iloc[i]
#         j+=1
#     if SpikesLeft['a_'+align].iloc[i] > start and SpikesLeft['a_'+align].iloc[i] < stop:
#         spikes.append(SpikesLeft['a_'+align].iloc[i])
#         trial_repeat.append(j)
#     else:
#         continue
            
# panel.set_ylabel('Trials (n)')
# panel.set_ylim(0,j)
# panel.set_xlim(start,stop) 

# y = np.arange(0,j+1,0.1)
# panel.fill_betweenx(y, cue_on,cue_off, color='grey', alpha=.4)
# panel.fill_betweenx(y, cue_off+delay,cue_off+delay+.2, color='beige', alpha=.8)
# panel.locator_params(nbins=5) 
# sns.despine()

fig, ax1 = plt.subplots(1,1, figsize=(8, 4), sharex=True)
y_upper=0
y_lower=0
colors=['crimson','darkgreen']
variables = ['WM_roll','WM_roll']
hits = [0,1]
variables_combined=[variables[0]+'_'+str(hits[0]),variables[1]+'_'+str(hits[1])]

delays = [10]
save_path = 'C:/Users/Tiffany/Google Drive/WORKING_MEMORY/PAPER/Figures/'
os.chdir(save_path)
file_name = 'single_delay_WM_roll0.6_0.5_folded5_sti_all'
df_cum_sti = pd.read_csv(file_name+'.csv', index_col=0)

# delays = [0.1,1,3,10]
scores = df_cum_sti.groupby('session').score.mean().reset_index()
list_exclude = scores.loc[scores.score<0.6].session.unique()
df_cum_sti = df_cum_sti[~df_cum_sti['session'].isin(list_exclude)] 

delays = [10]
panel=ax1
for delay in delays:
                
    for color, variable in zip(colors,variables_combined):
        print(variable)

        # Aligmnent for Stimulus cue
        real = np.array(df_cum_sti.loc[(df_cum_sti['trial_type'] == variable)&(df_cum_sti['delay'] == delay)].drop(columns=['trial_type', 'delay','session','fold','score']).mean(axis=0))
        times = df_cum_sti.loc[(df_cum_sti['trial_type'] == variable)&(df_cum_sti['delay'] == delay)]
        times = np.array(times.drop(columns=['trial_type', 'delay','session','fold','score'],axis = 1).columns.astype(float))

        df_lower = pd.DataFrame()
        df_upper = pd.DataFrame()

        for timepoint in times:
            mean_surr = []

            # recover the values for that specific timepoint
            try:
                array = df_cum_sti.loc[(df_cum_sti.trial_type ==variable)&(df_cum_sti['delay'] == delay)].drop(columns='delay').groupby('session').mean()[str(timepoint)].to_numpy()
            except:
                array = df_cum_sti.loc[(df_cum_sti.trial_type ==variable)&(df_cum_sti['delay'] == delay)].drop(columns='delay').groupby('session').mean()[timepoint].to_numpy()

            # iterate several times with resampling: chose X time among the same list of values
            for iteration in range(1000):
    
                x = np.random.choice(array, size=len(array), replace=True)
                # recover the mean of that new distribution
                mean_surr.append(np.mean(x))

            df_lower.at[0,timepoint] = np.percentile(mean_surr, 2.5)
            df_upper.at[0,timepoint] = np.percentile(mean_surr, 97.5)

        lower =  df_lower.iloc[0].values
        upper =  df_upper.iloc[0].values
        x=times

        panel.plot(times,real, color=color)
        panel.plot(x, lower, color=color, linestyle = '',alpha=0.6)
        panel.plot(x, upper, color=color, linestyle = '',alpha=0.6)
        panel.fill_between(x, lower, upper, alpha=0.2, color=color)
        if max(upper)>y_upper:
            y_upper = max(upper)
        if  min(lower)<y_lower:
            y_lower = min(lower)
        panel.set_ylim(y_lower-0.1,y_upper+0.1)
        panel.axhline(y=0.0,linestyle=':',color='black')
        panel.fill_betweenx(np.arange(-1,1.15,0.1), 0,0.4, color='grey', alpha=.4)
        panel.fill_betweenx(np.arange(-1.1,1.1,0.1), delay+0.4,delay+0.6, color='beige', alpha=.8)
        if color=='crimson':
            panel.set_xlabel('Time (s) to Go')