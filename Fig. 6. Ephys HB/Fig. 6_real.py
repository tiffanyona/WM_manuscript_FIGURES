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

from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
import statsmodels.formula.api as smf

pandas2ri.activate()

base     = importr('base')
car      = importr('car')
stats    = importr('stats')
lme4     = importr('lme4')
scales   = importr('scales')
lmerTest = importr('lmerTest')


utilities = 'G:/Mi unidad/WORKING_MEMORY/PAPER/WM_manuscript_FIGURES/'
os.chdir(utilities)
import functions as plots

save_path = 'G:/Mi unidad/WORKING_MEMORY/PAPER/WM_manuscript_FIGURES/Fig. 2 Model/'
path = 'G:/Mi unidad/WORKING_MEMORY/PAPER/ANALYSIS_figures/'

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

# -----------------############################## A Panel #################################-----------------------
def convolveandplot(df, upper_plot, lower_plot, variable='reward_side', cluster_id = 153, delay = 10, labels=['Correct right stimulus','Correct left stimulus'],
                   colors=[COLORRIGHT,COLORLEFT], align = 'Stimulus_ON', j=1, alpha=1, spikes=True, kernel=50):
    cue_on=0
    cue_off=0.35
    start=-2.5
    stop= 5 + delay
    
    neuron = new_convolve(df.loc[df.cluster_id==cluster_id], df, kernel)

    # neuron = neuron.loc[neuron.delay != 0.1] # Remove trials with no delay if the studied segment is that one. 

    # Align the data to the targeted align
    neuron['time_centered'] = neuron['times'] - neuron[align] 
    neuron['time_centered'] = np.round(neuron.time_centered/1000, 2) #### estos es importante!!
    neuron['firing_'] = neuron['firing']*1000

    df_results = pd.DataFrame(dtype=float)
    df_results['firing'] = neuron.loc[(neuron.time_centered <= stop)&(neuron.delay == delay)].groupby(['time_centered',variable])['firing_'].mean()
    df_results['error'] = neuron.loc[(neuron.time_centered <= stop)&(neuron.delay == delay)].groupby(['time_centered',variable])['firing_'].std()
    df_results.reset_index(inplace=True)
    
    panel = lower_plot
    for condition,color,name in zip([1,0],colors,labels):
        y_mean= df_results[df_results[variable]==condition].firing
        error = 0.5*df_results[df_results[variable]==condition].error
        lower = y_mean - error
        upper = y_mean + error
        x=df_results[df_results[variable]==condition].time_centered

        panel.plot(x, y_mean, label=name,color=color, alpha=alpha)
        panel.plot(x, lower, color=color, alpha=0.0, linewidth=0)
        panel.plot(x, upper, color=color, alpha=0.0, linewidth=0)
        panel.fill_between(x, lower, upper, alpha=0.2, color=color, linewidth=0)

    panel.set_xlim(start,stop)  
    panel.set_ylim(0,50)  
    y = np.arange(0,50,0.1)     
    panel.fill_betweenx(y, cue_on,cue_off, color='lightgrey', alpha=1, linewidth=0)  
    panel.fill_betweenx(y, cue_off+delay,cue_off+delay+.2, color='grey', alpha=1, linewidth=0)
    
    # axis labels and legend
    lower_plot.legend(frameon=False)  
    panel.set_ylabel('Firing rate (spks/s)')
    panel.locator_params(nbins=4) 

    if spikes:
        pass
    else:
        return j
    
    panel = upper_plot
    SpikesRight = (df.loc[(df[variable] == 1)&(df.cluster_id == cluster_id)&(df.delay == delay)])
    SpikesLeft = (df.loc[(df[variable] == 0)&(df.cluster_id == cluster_id)&(df.delay == delay)])

    SpikesRight['a_'+align] = SpikesRight['fixed_times'] - SpikesRight['Stimulus_ON'] 
    SpikesLeft['a_'+align] = SpikesLeft['fixed_times'] - SpikesLeft['Stimulus_ON'] 

    trial=1
    spikes = []
    trial_repeat = []
    for i in range(len(SpikesRight)):
        # Plot for licks for left trials
        if SpikesRight.trial.iloc[i] != trial:
            panel.plot(spikes,trial_repeat, '|', markersize=3, color=colors[0], zorder=1)
            spikes = []
            trial_repeat = []
            trial = SpikesRight.trial.iloc[i]
            j+=1
        if SpikesRight['a_'+align].iloc[i] > start and SpikesRight['a_'+align].iloc[i] < stop:
            spikes.append(SpikesRight['a_'+align].iloc[i])
            trial_repeat.append(j)
        else:
            continue

    trial=1
    spikes = []
    trial_repeat = []
    for i in range(len(SpikesLeft)):
        # Plot for licks for left trials
        if SpikesLeft.trial.iloc[i] != trial:
            panel.plot(spikes,trial_repeat, '|', markersize=3, color=colors[1], zorder=1)
            spikes = []
            trial_repeat = []
            trial = SpikesLeft.trial.iloc[i]
            j+=1
        if SpikesLeft['a_'+align].iloc[i] > start and SpikesLeft['a_'+align].iloc[i] < stop:
            spikes.append(SpikesLeft['a_'+align].iloc[i])
            trial_repeat.append(j)
        else:
            continue

    panel.set_ylabel('Trials (n)')
    panel.set_ylim(0,j)
    panel.set_xlim(start,stop)  

    y = np.arange(0,j+1,0.1)
    panel.fill_betweenx(y, cue_on,cue_off, color='grey', alpha=1, linewidth=0)
    panel.fill_betweenx(y, cue_off+delay,cue_off+delay+.2, color='darkgrey', alpha=1, linewidth=0)
    
    panel.locator_params(nbins=5) 
    panel.axes.get_xaxis().set_visible(False)
    
    return j

def plot_decoder_single(left, df_cum_sti,baseline=0.5,individual_sessions=False, align='Stimulus_ON', colors=['black'], upper_limit=0.2, alpha=1, variables_combined=['WM_roll_1']):
    for color, variable,left in zip(colors,variables_combined,left):
        if individual_sessions == True:
            # Aligmnent for Stimulus cue - sessions separately
            real = df_cum_sti.loc[(df_cum_sti['trial_type'] == variable)].groupby('session').mean().drop(columns=['fold','score']).reset_index()
            times = df_cum_sti.loc[(df_cum_sti['trial_type'] == variable)]

            try:
                times = np.array(times.drop(columns=['trial_type','session','fold','score'],axis = 1).columns.astype(float))
            except:
                times = np.array(times.drop(columns=['trial_type','session','fold','score','subject'],axis = 1).columns.astype(float))
        
            x=times
            for i in range(len(real)):
                left.plot(times,real.iloc[i][1:-1], color=color,alpha=0.1)

        try:
            times = df_cum_sti.loc[(df_cum_sti['trial_type'] == variable)]
            times = np.array(times.drop(columns=['trial_type','session','fold','score'],axis = 1).columns.astype(float))
            real =  np.array(np.mean(df_cum_sti.loc[(df_cum_sti['trial_type'] ==variable)]
                                     .groupby('session').mean().drop(columns = ['fold','score'])))

        except:
            try:
                times = df_cum_sti.loc[(df_cum_sti['trial_type'] == variable)]
                times = np.array(times.drop(columns=['trial_type','session', 'score_type'], axis = 1).columns.astype(float))
                real =  np.array(np.mean(df_cum_sti.loc[(df_cum_sti['trial_type'] ==variable)]
                                         .groupby('session').mean()))
            except:
                times = df_cum_sti.loc[(df_cum_sti['trial_type'] == variable)]
                times = np.array(times.drop(columns=['subject', 'trial_type', 'session', 'fold', 'score'], axis = 1).columns.astype(float))
                real =  np.array(np.mean(df_cum_sti.loc[(df_cum_sti['trial_type'] ==variable)]
                                         .groupby('session').mean().drop(columns = ['fold','score'])))                
        # df_new = pd.DataFrame()
        # for iteration in np.arange(1,100):
        #     df_new[iteration]= df_cum_shuffle_sti.loc[(df_cum_shuffle_sti.trial_type==variable)].groupby('times').mean()[iteration]
    
        # y_mean= df_new.mean(axis=1).values
        # upper =  df_new.quantile(q=0.975, interpolation='linear',axis=1) - y_mean
        # lower =  df_new.quantile(q=0.025, interpolation='linear',axis=1) - y_mean
        
        # left.plot(x, lower, color=color, linestyle = '', linewidth=0)
        # left.plot(x, upper, color=color, linestyle = '', linewidth=0)
        # left.fill_between(x, lower, upper, alpha=0.2, color=color, linewidth=0)
        
        # lower =  real - 2*df_for_boots.std()
        # upper =  real + 2*df_for_boots.std()
        # lower =  df_cum_sti.quantile(0.025)
        # upper =  df_cum_sti.quantile(0.975)
    
        left.plot(times,real, color=color, alpha=alpha)
            
        left.set_ylim(baseline-0.1,upper_limit+baseline)
        left.axhline(y=baseline,linestyle=':',color='black')
        left.set_ylabel('Decoding\n accuracy')

        
def plot_decoder(left, df_cum_sti,baseline=0.5,individual_sessions=False, show_axis=True, align='Stimulus_ON', colors=['black'], upper_limit=0.2, alpha=1, variables_combined=['WM_roll_1']):
    for color, variable,left in zip(colors,variables_combined,left):
        if individual_sessions == True:
            # Aligmnent for Stimulus cue - sessions separately
            real = df_cum_sti.loc[(df_cum_sti['trial_type'] == variable)].groupby('session').mean().drop(columns=['fold','score']).reset_index()
            times = df_cum_sti.loc[(df_cum_sti['trial_type'] == variable)]

            try:
                times = np.array(times.drop(columns=['trial_type','session','fold','score'],axis = 1).columns.astype(float))
            except:
                times = np.array(times.drop(columns=['trial_type','session','fold','score','subject'],axis = 1).columns.astype(float))
        
            x=times
            for i in range(len(real)):
                left.plot(times,real.iloc[i][1:-1], color=color,alpha=0.1)

        try:
            times = df_cum_sti.loc[(df_cum_sti['trial_type'] == variable)]
            times = np.array(times.drop(columns=['trial_type','session','fold','score'],axis = 1).columns.astype(float))
            real =  np.array(np.mean(df_cum_sti.loc[(df_cum_sti['trial_type'] ==variable)]
                                     .groupby('session').mean().drop(columns = ['fold','score'])))

        except:
            try:
                times = df_cum_sti.loc[(df_cum_sti['trial_type'] == variable)]
                times = np.array(times.drop(columns=['trial_type','session', 'score_type'], axis = 1).columns.astype(float))
                real =  np.array(np.mean(df_cum_sti.loc[(df_cum_sti['trial_type'] ==variable)]
                                         .groupby('session').mean()))
            except:
                times = df_cum_sti.loc[(df_cum_sti['trial_type'] == variable)]
                times = np.array(times.drop(columns=['subject', 'trial_type', 'session', 'fold', 'score'], axis = 1).columns.astype(float))
                real =  np.array(np.mean(df_cum_sti.loc[(df_cum_sti['trial_type'] ==variable)]
                                         .groupby('session').mean().drop(columns = ['fold','score'])))                
        mean_surr = []
        df_lower = pd.DataFrame()
        df_upper = pd.DataFrame()
    
        for timepoint in times:
            mean_surr = []
    
            # recover the values for that specific timepoint
            try:
                array = (df_cum_sti.loc[(df_cum_sti.trial_type ==variable)]
                                    .drop(columns='fold').groupby('session').mean()[str(timepoint)].to_numpy())
            except:
                array = (df_cum_sti.loc[(df_cum_sti.trial_type ==variable)].groupby('session').mean()[str(timepoint)].to_numpy())
                
            # iterate several times with resampling: chose X time among the same list of values
            for iteration in range(1000):
                x = np.random.choice(array, size=len(array), replace=True)
                # recover the mean of that new distribution
                mean_surr.append(np.mean(x))
    
            df_lower.at[0,timepoint] = np.percentile(mean_surr, 2.5)
            df_upper.at[0,timepoint] = np.percentile(mean_surr, 97.5)
        
        x=times
        lower =  df_lower.iloc[0].values
        upper =  df_upper.iloc[0].values
        
        left.plot(x, lower, color=color, linestyle = '', linewidth=0)
        left.plot(x, upper, color=color, linestyle = '', linewidth=0)
        left.fill_between(x, lower, upper, alpha=0.2, color=color, linewidth=0)
        
        # lower =  real - 2*df_for_boots.std()
        # upper =  real + 2*df_for_boots.std()
        # lower =  df_cum_sti.quantile(0.025)
        # upper =  df_cum_sti.quantile(0.975)
    
        left.plot(times,real, color=color, alpha=alpha)
            
        left.set_ylim(baseline-0.05,upper_limit+baseline)
        left.axhline(y=baseline,linestyle=':',color='black')

        left.set_ylabel('Decoding\n accuracy')
        
        y = np.arange(0,upper_limit+baseline+1,0.1)
        if align == 'Stimulus_ON':
            left.fill_betweenx(y, 0,.35, color='lightgrey', alpha=1, linewidth=0)
        elif align == 'Delay_OFF':
            left.fill_betweenx(y, 0,0.2, color='grey', alpha=1, linewidth=0)  
    
        if show_axis==False:
            left.spines['left'].set_visible(False)

def new_convolve(nx,df, kernel=50):
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
            gaus_rate = instantaneous_rate(spiketrain, sampling_period=5*ms, kernel=GaussianKernel(kernel*ms)) #s.d of Suzuki & Gottlieb 
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

def plotsingledelay(df_cum_sti, panel, colors, variables_combined, delay, invert_list = [False, False, False]):
    baseline = 0.5
    y_upper=baseline
    y_lower=baseline

    for color, variable, invert in zip(colors,variables_combined, invert_list):
    
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
                array = (df_cum_sti.loc[(df_cum_sti.trial_type ==variable)&(df_cum_sti['delay'] == delay)]
                                    .drop(columns='delay').groupby('session').mean()[str(timepoint)].to_numpy())
            except:
                array = (df_cum_sti.loc[(df_cum_sti.trial_type ==variable)&(df_cum_sti['delay'] == delay)]
                                    .drop(columns='delay').groupby('session').mean()[timepoint].to_numpy())
            
            # iterate several times with resampling: chose X time among the same list of values
            for iteration in range(1000):
                x = np.random.choice(array, size=len(array), replace=True)
                # recover the mean of that new distribution
                mean_surr.append(np.mean(x))
    
            df_lower.at[0,timepoint] = np.percentile(mean_surr, 2.5)
            df_upper.at[0,timepoint] = np.percentile(mean_surr, 97.5)
        print(len(array))
        
        lower =  df_lower.iloc[0].values
        upper =  df_upper.iloc[0].values
        x=times
        if invert:
            real = -real +1
            lower = -lower +1
            upper = -upper +1

        panel.plot(times,real, color=color)
        panel.plot(x, lower, color=color, linestyle = '',alpha=0.6, linewidth=0)
        panel.plot(x, upper, color=color, linestyle = '',alpha=0.6, linewidth=0)
        panel.fill_between(x, lower, upper, alpha=0.2, color=color, linewidth=0)
        if max(upper)>y_upper:
            y_upper = max(upper)
        if  min(lower)<y_lower:
            y_lower = min(lower)
        panel.set_ylabel('Accuracy')
        panel.axhline(y=baseline,linestyle=':',color='black')
        panel.fill_betweenx(np.arange(-baseline-0.1,baseline+.45,0.1), 0,0.35, color='lightgrey', alpha=1, linewidth=0)
        panel.fill_betweenx(np.arange(-baseline-0.1,baseline+.45,0.1), delay+.35,delay+.55, color='grey', alpha=1, linewidth=0)
        panel.set_ylim(y_lower-0.05,y_upper+0.05)
        panel.set_xlim(-2,delay+3)
        if panel=='crimson':
            panel.set_xlabel('Time to Cue onset (s)')
            
# ----------------------------------------------------------------------------------------------------------------

# -----------------##################################################    Stimulus and Choice  #########################################################-----------------------

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



# -----------------###########################################################################################################-----------------------

            
 # -----------------############################## Delay code in WM and RL correct - both halves  #######################-----------------------

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
       
# plot_decoder([d1,d1,d1,d1], df_cum_sti,baseline=0.5,upper_limit=0.3, align='Stimulus_ON', colors=colors, variables_combined=variables_combined)
# plot_decoder([d2,d2,d2,d2], df_cum_res,baseline=0.5,upper_limit=0.3, align='Stimulus_ON', colors=colors, variables_combined=variables_combined)
    
  # -----------------############################## Delay code in WM and RL correct - both halves  #######################-----------------------

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

# -----------------############################## Delay code in WM and RL correct - one example session  #######################-----------------------

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
    
# -----------------###########################################################################################################-----------------------

# ------######################################## Session difference ###############################-----------------------

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

r_df  = ro.conversion.py2ri(df_final)

# formula="logs ~ state*epoch + (1|session)"
# formula="logs ~ state*epoch + (1|session)+ (1|fold)"
# formula="logs ~ state*epoch + (state+epoch+1|session) + (state+epoch+1|fold)"
formula="log_odds ~ state*epoch + state+epoch+1|session:fold)"

model = lme4.lmer(formula, data=r_df)

for i, v in enumerate(list(base.summary(model).names)):
    if v in ['coefficients']:
        print (base.summary(model).rx2(v))
# print(base.summary(model))
print(car.Anova(model))


# ------######################################## Single trial example of population activity during WM and RL ###############################-----------------------


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

# ------#########################################################################################-----------------------


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

# ------#########################################################################################-----------------------


# -----------------############################## PSTH for an example neuron with WM and Hb trials #################################-----------------------


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
                           labels=['WM left','HB left'], colors=['darkgreen', 'indigo'], kernel=200)    
 
        # j = convolveandplot(temp_df.loc[(temp_df.vector_answer == 0)&(temp_df.hit == 0)], i1, j1, variable='state', cluster_id = cluster_id, delay = delay, j=j,
        #                    labels=['WM left','HB left'], colors=['lightgrey', 'lightgrey'], kernel=200) 
        
        i1.set_ylim(0,19)


  # -----------------############################## Delay code in WM and RL correct - both halves  #######################-----------------------

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

save_path = 'C:/Users/Tiffany/Google Drive/WORKING_MEMORY/PAPER/Panel figures/Fig. 6. Ephys HB/'
# plt.savefig(save_path+'/Fig. 6. Ephys HB_V2.svg', bbox_inches='tight',dpi=300)
# plt.savefig(save_path+'/Fig. 6. Ephys HB_V2.png', bbox_inches='tight',dpi=300)

plt.show()