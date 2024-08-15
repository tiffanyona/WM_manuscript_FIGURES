# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 15:14:01 2023

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
# from datahandler import Utils
from neo.core import SpikeTrain
from quantities import ms, s, Hz
from elephant.statistics import mean_firing_rate
from elephant.statistics import time_histogram, instantaneous_rate
from elephant.kernels import GaussianKernel
from elephant.statistics import mean_firing_rate
from cycler import cycler


def plot_results_shuffle(df_cum_sti, df_cum_res, colors, variables_combined, fig = False, ax1=False, ax2=False, upper_limit=0.4, baseline=0):
    if fig == False:
        fig, ([ax1,ax2])  = plt.subplots(1,2, figsize=(14, 4), sharey=True)
        
    for color, variable,left,right in zip(colors,variables_combined,[ax1],[ax2]):

        # Aligmnent for Stimulus cue
        real = np.array(np.mean(df_cum_sti.groupby('session').mean().drop(columns=['session_shuffle']))) 
        # real = np.array(np.mean(df_cum_sti.groupby('session').median()))
        times = df_cum_sti
        times = np.array(times.drop(columns=['session','session_shuffle','subject'],axis = 1).columns.astype(float))

        mean_surr = []
        df_lower = pd.DataFrame()
        df_upper = pd.DataFrame()

        df_for_boots = df_cum_sti.groupby('session').mean()
        for timepoint in times:
            mean_surr = []

            # recover the values for that specific timepoint
            try:
                array = df_for_boots[timepoint].to_numpy()
            except:
                array = df_for_boots[str(timepoint)].to_numpy()

            # iterate several times with resampling: chose X time among the same list of values
            for iteration in range(1000):
                x = np.random.choice(array, size=len(array), replace=True)
                # recover the mean of that new distribution
                mean_surr.append(np.mean(x))

            df_lower.at[0,timepoint] = np.percentile(mean_surr, 2.5)
            df_upper.at[0,timepoint] = np.percentile(mean_surr, 97.5)

        lower =  df_lower.iloc[0].values
        upper =  df_upper.iloc[0].values
        # lower =  real - 2*df_for_boots.std()
        # upper =  real + 2*df_for_boots.std()
        # lower =  df_cum_sti.quantile(0.025)
        # upper =  df_cum_sti.quantile(0.975)
        x=times

        # plt.plot(x, y_mean, label='Non-prefered Stimulus', color=color)
        left.plot(times,real, color=color)
        left.plot(x, lower, color=color, linestyle = '',alpha=0.6)
        left.plot(x, upper, color=color, linestyle = '',alpha=0.6)
        left.fill_between(x, lower, upper, alpha=0.2, color=color)
        left.set_ylim(baseline-0.1,baseline+upper_limit)
        left.axhline(y=0.0,linestyle=':',color='black')
        left.fill_betweenx(np.arange(-1,1.15,0.1), 0,0.4, color='lightlightgrey', alpha=.4)
        sns.despine()

        # -------------------- For Aligment to Go cue
        real = np.array(np.mean(df_cum_res.groupby('session').mean().drop(columns=['session_shuffle']))) 
        times = df_cum_res
        times = np.array(times.drop(columns=['session','session_shuffle','subject'],axis = 1).columns.astype(float))

        mean_surr = []
        df_lower = pd.DataFrame()
        df_upper = pd.DataFrame()

        # df_for_boots = df_cum_res.loc[:, df_cum_res.columns != 'session']
        df_for_boots = df_cum_res.groupby('session').mean()
        for timepoint in times:
            mean_surr = []

            # recover the values for that specific timepoint
            try:
                array = df_for_boots[timepoint].to_numpy()
            except:
                array = df_for_boots[str(timepoint)].to_numpy()

            # iterate several times with resampling: chose X time among the same list of values
            for iteration in range(1000):
                # print(array)
                x = np.random.choice(array, size=len(array), replace=True)
                # recover the mean of that new distribution
                mean_surr.append(np.mean(x))

            df_lower.at[0,timepoint] = np.percentile(mean_surr, 2.5)
            df_upper.at[0,timepoint] = np.percentile(mean_surr, 97.5)

        lower =  df_lower.iloc[0].values
        upper =  df_upper.iloc[0].values
        # lower =  real - 2*df_for_boots.std()
        # upper =  real + 2*df_for_boots.std()
        # lower =  df_cum_res.quantile(0.025)
        # upper =  df_cum_res.quantile(0.975)

        # lower = -np.mean(df_cum_res.loc[df_cum_sti['trial_type'] ==variable].groupby(['session','fold']).sem(),axis=0)*2
        # upper = np.mean(df_cum_res.loc[df_cum_sti['trial_type'] ==variable].groupby(['session','fold']).sem(),axis=0)*2
        x=times

        # ax2.plot(x, y_mean, color=color)
        right.plot(times,real, color=color)
        right.plot(x, lower, color=color, linestyle = '',alpha=0.6)
        right.plot(x, upper, color=color, linestyle = '',alpha=0.6)
        right.fill_between(x, lower, upper, alpha=0.2, color=color)
        right.set_ylim(baseline-0.1,baseline+upper_limit)
        right.axhline(y=0,linestyle=':',color='black')
        right.fill_betweenx(np.arange(-1.1,1.1,0.1), 0,0.2, color='lightgrey', alpha=.8)
        right.set_xlabel('Time from go cue (s)')
        
def single_trial_with_decoder(df, df_decoder, big_data, filename, T, panels = [], threshold = 0.4, 
                              align = 'Stimulus_ON', show_y = True ):
    delay = df.loc[df.trial==T].delay.unique()[0]
    cue_on=0
    cue_off=0.38
    start=-2.5
    stop=3+delay
    # stop=12

    # stop= max(df.loc[df.trial==T]['a_'+align])
    endrange = max(df.loc[df.trial==T]['a_'+align])
    
    ### ------ Filter neurons with substantial weight in the decoder
    window='Delay_OFF--0.5-0'
    path = 'G:/Mi unidad/WORKING_MEMORY/PAPER/ANALYSIS_figures/'
    file_weights = 'weights for the modelling_complete.csv'
    # file_weights = 'weights for the modelling_3and10_V8'

    df_weights = pd.read_csv(path+file_weights, index_col=0)
    df_weights = df_weights.loc[df_weights.session == filename]
    
    significant_neurons = df_weights.loc[(df_weights[window] > threshold)|(df_weights[window] < -threshold)].neuron.unique()
    df = df[df['cluster_id'].isin(significant_neurons)]
    
    # Align to specific epoch, in this case Stimulus 
    big_data['time_centered'] = big_data['times'] - big_data['Stimulus_ON'] 
    big_data['time_centered'] = np.round(big_data.time_centered/1000, 2) #### estos es importante!!
    big_data['firing_'] = big_data['firing']*1000
    
    df_results = pd.DataFrame(dtype=float)
    df_results['firing'] = big_data.loc[(big_data.time_centered <= stop)].groupby(['time_centered','neuron'])['firing_'].mean()
    df_results['error'] = big_data.loc[(big_data.time_centered <= stop)].groupby(['time_centered','neuron'])['firing_'].std()
    df_results.reset_index(inplace=True)
    
    # This piece of code is to filter neurons with lower than 1 Hz firing across the session
    filter_neuron = df_results.groupby('neuron').firing.mean().reset_index()
    filter_neuron = (df_results.loc[(df_results.time_centered>0)&(df_results.time_centered<delay+0.2)]
                     .groupby(['neuron']).firing.mean().reset_index())
    # filter_neuron = filter_neuron.loc[filter_neuron.firing > 1]
    neurons = filter_neuron.neuron.unique()
    df_results = df_results.loc[df_results.neuron.isin(neurons)]
    
    significant_left_neurons = df_weights.loc[(df_weights[window] < -threshold)].neuron.unique()
    significant_right_neurons = df_weights.loc[(df_weights[window] > threshold)].neuron.unique()
    
    left = df_results[df_results['neuron'].isin(significant_right_neurons)].groupby('time_centered').firing.mean().values
    right = df_results[df_results['neuron'].isin(significant_left_neurons)].groupby('time_centered').firing.mean().values
    
    panel = panels[1]
    x = df_results[df_results['neuron'].isin(significant_right_neurons)].groupby('time_centered').firing.mean().index
    panel.plot(x,  df_results[df_results['neuron'].isin(significant_right_neurons)].groupby('time_centered').firing.mean().values, color=COLORRIGHT)
    
    x = df_results[df_results['neuron'].isin(significant_left_neurons)].groupby('time_centered').firing.mean().index
    panel.plot(x,  df_results[df_results['neuron'].isin(significant_left_neurons)].groupby('time_centered').firing.mean().values, color=COLORLEFT)
    
    panel.set_xlim(start,stop)  
    panel.set_ylabel('Firing rate (spks/s)')
    
    y = np.arange(0,75,0.1)     
    panel.fill_betweenx(y, cue_on,cue_off, color='lightgrey', alpha=.8)  
    panel.fill_betweenx(y, cue_off+delay,cue_off+delay+.2, color='lightgrey', alpha=.8)
    panel.set_ylim(0,max(df_results.groupby('time_centered').firing.mean().values)+5)
    panel.xaxis.set_tick_params(labelbottom=False)

# ---------------  Raster -------------   Organized by cluster_id and corrected for FR ____________________________
    cluster_id=[]
    FR_mean=[]
    weights = []
    dft = df.loc[df.trial ==T]
    dft = dft.loc[dft.cluster_id.isin(neurons)]
    start_FR = 0
    stop_FR = delay
    df['a_Stimulus_ON'] = df['fixed_times'] - df['Stimulus_ON']
    
    for N in dft.cluster_id.unique():
        # spikes = dft.loc[(dft.cluster_id==N)]['a_'+align].values
        spikes = dft.loc[(dft.cluster_id==N)&(dft['a_'+align] > start_FR)&(dft['a_'+align]<stop_FR)].fixed_times.values
        FR_mean.append(len(spikes)/abs(stop_FR-start_FR))
        weights.append(df_weights.loc[df_weights.neuron == N][window].values)
        cluster_id.append(N)
    
    df_spikes = pd.DataFrame(list(zip(cluster_id,FR_mean, weights)), columns =['cluster_id','FR','weights'])
    df_spikes = df_spikes.sort_values('FR')
    # df_spikes = df_spikes.sort_values('weights')
    df_spikes['new_order'] = np.arange(len(df_spikes))
    
    dft = pd.merge(df_spikes, dft, on=['cluster_id'])
    
    right = 1
    left = len(cluster_id)
    panel = panels[0]
    for N in reversed(dft.new_order.unique()):
        cluster_id = dft.loc[dft.new_order==N].cluster_id.iloc[0]
        # if df_weights.loc[df_weights.neuron == cluster_id]["Delay_OFF--0.5-0"].iloc[0] >0:
        if df_weights.loc[df_weights.neuron == cluster_id][window].iloc[0] >threshold:
            color_selectivity=COLORRIGHT
            j=right
            right+=1
        elif df_weights.loc[df_weights.neuron == cluster_id][window].iloc[0] <-threshold:
        # elif df_weights.loc[df_weights.neuron == cluster_id]["Delay_OFF--0.5-0"].iloc[0] <0:
            color_selectivity=COLORLEFT
            j=left
            left-=1
        spikes = dft.loc[dft.new_order==N]['a_'+align].values
        panel.plot(spikes,np.repeat(j, len(spikes)), '|', markersize=3, color=color_selectivity, zorder=3)
    
    panel.set_ylabel('Single units')
    panel.set_ylim(0,len(dft.new_order.unique())+1)
    panel.set_ylabel('Neurons')

    y = np.arange(0,len(dft.new_order.unique())+1,0.1)
    panel.fill_betweenx(y, cue_on,cue_off, color='lightgrey', alpha=1)
    panel.fill_betweenx(y, cue_off+delay,cue_off+delay+.2, color='lightgrey', alpha=1)    
    panel.set_xlim(start,stop)
    panel.xaxis.set_tick_params(labelbottom=False)
    
    # axis labels and legend
    # if T == 21:
    #     panel.set_title('Trial: ' + str(T) + '; WM_roll: '+str(0.1)+'; Hit: '+str(dft.hit.unique()[0])+ '; Side: '+str(dft.reward_side.unique()[0]))
    # else:
    #     panel.set_title('Trial: ' + str(T) + '; WM_roll: '+str(np.round(dft.WM_roll.unique()[0],2))+'; Hit: '+str(dft.hit.unique()[0])+ '; Side: '+str(dft.reward_side.unique()[0]))

##  --------------------------Decoder plot
    panel = panels[2]
    
    panel.plot(df_decoder['times'],df_decoder['real'], color='grey')
    if abs(min(df_decoder['real'])) > max(df_decoder['real']):
        max_true=abs(min(df_decoder['real']))+0.5
    else:
        max_true=abs(max(df_decoder['real']))
    panel.set_ylim(-max_true,max_true)
    panel.axhline(y=0.0,linestyle=':',color='black')
    panel.fill_betweenx(np.arange(-max_true,max_true,0.1), 0,0.4, color='lightgrey', alpha=.8)
    panel.fill_betweenx(np.arange(-max_true,max_true,0.1), delay+0.4,delay+0.6, color='lightgrey', alpha=.8)
    panel.set_xlabel('Time from stimulus onset (s)')
    panel.set_ylabel('Log odds')
    panel.set_xlim(start,stop)
    
    if show_y == False:
        panel[0].yaxis.set_tick_params(labelbottom=False)
        panel[1].yaxis.set_tick_params(labelbottom=False)
        panel[2].yaxis.set_tick_params(labelbottom=False)

def convolveandplot(df, upper_plot, lower_plot, variable='reward_side', cluster_id = 153, delay = 10, labels=['Correct right stimulus','Correct left stimulus'],
                   colors=[COLORRIGHT,COLORLEFT], align = 'Stimulus_ON', j=1, alpha=1, spikes=True, kernel=50):
    cue_on=0
    cue_off=0.4
    start=-2
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
    panel.set_xlabel('Time from stimulus onset (s)')
    y = np.arange(0,50,0.1)     
    panel.fill_betweenx(y, cue_on,cue_off, color='lightgrey', alpha=1, linewidth=0)  
    panel.fill_betweenx(y, cue_off+delay,cue_off+delay+.2, color='lightgrey', alpha=1, linewidth=0)
    
    # axis labels and legend
    lower_plot.legend(frameon=False)  
    panel.set_xlabel('Time from stimulus onset (s)')
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
            panel.plot(spikes,trial_repeat, '|', markersize=0.5, color=colors[0], zorder=1)
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
            panel.plot(spikes,trial_repeat, '|', markersize=0.5, color=colors[1], zorder=1)
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
    panel.fill_betweenx(y, cue_on,cue_off, color='lightgrey', alpha=1, linewidth=0)
    panel.fill_betweenx(y, cue_off+delay,cue_off+delay+.2, color='lightgrey', alpha=1, linewidth=0)
    
    panel.locator_params(nbins=5) 
    panel.axes.get_xaxis().set_visible(False)
    
    return j


def plot_decoder(left, df,baseline=0.5,individual_sessions=False, align='Stimulus_ON', show_axis=True,colors=['black'], upper_limit=0.2, variables_combined=['WM_roll_1']):
    for color, variable,left in zip(colors,variables_combined,left):
        if individual_sessions == True:
            # Aligmnent for Stimulus cue - sessions separately
            real = df.groupby('session').median().reset_index()
            try:
                times = np.array(df.columns[:-4]).astype(float)
            except:
                times = np.array(df.columns[1:]).astype(float)
        
            left.set_xlabel('Time (s) to Cue')
        
            x=times
            for i in range(len(real)):
                left.plot(times,real.iloc[i][1:-1], color=color,alpha=0.1)
                
        try:
            df_loop = df.loc[(df['trial_type'] == variable)]
        except:
            df_loop = df

        # Select only columns where the column name is a number or can be transformed to a number
        numeric_columns = df_loop.columns[df_loop.columns.to_series().apply(pd.to_numeric, errors='coerce').notna()]

        real = np.array(np.mean(df_loop.groupby('session').mean()[numeric_columns])) 
        times = df_loop[numeric_columns].columns.astype(float)


        df_results = pd.DataFrame()
        df_results['times'] = times
        df_results['real'] = real
        df_results = df_results.sort_values(by='times')

        mean_surr = []
        df_lower = pd.DataFrame()
        df_upper = pd.DataFrame()
    
        df_for_boots = df_loop.groupby('session').mean()[numeric_columns]

        for timepoint in df_results['times']:
            mean_surr = []
    
            # recover the values for that specific timepoint
            array = df_for_boots[str(timepoint)].to_numpy()

            # iterate several times with resampling: chose X time among the same list of values
            for iteration in range(1000):
                x = np.random.choice(array, size=len(array), replace=True)
                # recover the mean of that new distribution
                mean_surr.append(np.mean(x))
    
            df_lower.at[0,timepoint] = np.percentile(mean_surr, 0.5)
            df_upper.at[0,timepoint] = np.percentile(mean_surr, 99.5)
        
        x=times
        lower =  df_lower.iloc[0].values
        upper =  df_upper.iloc[0].values
        left.plot(x, lower, color=color, linestyle = '',alpha=0.6, linewidth=0)
        left.plot(x, upper, color=color, linestyle = '',alpha=0.6, linewidth=0)
        left.fill_between(x, lower, upper, alpha=0.2, color=color, linewidth=0)
    
        left.plot(times,real, color=color)

        left.fill_betweenx(np.arange(-baseline-0.1,baseline+.5,0.1), 0,0.35, color='lightgrey', alpha=1, linewidth=0)
        left.fill_betweenx(np.arange(-baseline-0.1,baseline+.5,0.1), 3.35,3.55, color='lightgrey', alpha=1, linewidth=0)
        left.set_ylim(baseline-0.1,upper_limit+baseline)
        left.axhline(y=baseline,linestyle=':',color='black')
        left.set_xlabel('Time from stimulus onset (s)')
        left.set_ylabel('Excess decoding\n accuracy')
        
        y = np.arange(-1,1.15,0.1)
        if align == 'Stimulus_ON':
            left.fill_betweenx(y, 0,.35, color='lightgrey', alpha=1, linewidth=0)
        elif align == 'Delay_OFF':
            left.fill_betweenx(y, 0,0.2, color='lightgrey', alpha=1, linewidth=0)  
    
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
            histogram_rate = time_histogram([spiketrain], 20*ms, output='rate')
            gaus_rate = instantaneous_rate(spiketrain, sampling_period=20*ms, kernel=GaussianKernel(kernel*ms)) #s.d of Suzuki & Gottlieb 
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

def plotsingledelay(df_cum_sti, panel, colors, variables_combined, delay, baseline = 0.5, invert_list=[False, False]):

    y_upper=baseline
    y_lower=baseline

    for color, variable, invert in zip(colors,variables_combined, invert_list):
    
        # Aligmnent for Stimulus cue
        # try:
        # real = np.array(df_cum_sti.loc[(df_cum_sti['trial_type'] == variable)&(df_cum_sti['delay'] == delay)].drop(columns=['trial_type', 'delay','session','fold','score']).mean(axis=0))
        # times = df_cum_sti.loc[(df_cum_sti['trial_type'] == variable)&(df_cum_sti['delay'] == delay)]
        # times = np.array(times.drop(columns=['trial_type', 'delay','session','fold'],axis = 1).columns.astype(float))
        # except:
        #     real = np.array(df_cum_sti.loc[(df_cum_sti['trial_type'] == variable)&(df_cum_sti['delay'] == delay)].drop(columns=['trial_type', 'delay','session','fold']).mean(axis=0))
        #     times = df_cum_sti.loc[(df_cum_sti['trial_type'] == variable)&(df_cum_sti['delay'] == delay)]
        #     times = np.array(times.drop(columns=['trial_type', 'delay','session','fold'],axis = 1).columns.astype(float))
        
        df_loop = df_cum_sti.loc[(df_cum_sti['trial_type'] == variable)&(df_cum_sti['delay'] == delay)]

        # Select only columns where the column name is a number or can be transformed to a number
        numeric_columns = df_loop.columns[df_loop.columns.to_series().apply(pd.to_numeric, errors='coerce').notna()]

        real = np.array(np.mean(df_loop.groupby('session').mean()[numeric_columns])) 
        times = df_loop[numeric_columns].columns.astype(float)
        
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
        
        if invert:
            real = -real +baseline
            lower = -lower +baseline
            upper = -upper +baseline
            
        panel.plot(times,real, color=color)
        panel.plot(x, lower, color=color, linestyle = '',alpha=0.6, linewidth=0)
        panel.plot(x, upper, color=color, linestyle = '',alpha=0.6, linewidth=0)
        panel.fill_between(x, lower, upper, alpha=0.2, color=color, linewidth=0)
        if max(upper)>y_upper:
            y_upper = max(upper)
        if  min(lower)<y_lower:
            y_lower = min(lower)
        panel.set_ylabel('Excess decoding\n accuracy')
        panel.axhline(y=baseline,linestyle=':',color='black')
        panel.fill_betweenx(np.arange(-1.1,3.1,0.1), 0,0.4, color='lightgrey', alpha=1, linewidth=0)
        panel.fill_betweenx(np.arange(-1.1,3.1,0.1), delay+.4,delay+.6, color='lightgrey', alpha=1, linewidth=0)
        panel.set_ylim(y_lower-0.045,y_upper+0.01)
        if panel=='crimson':
            panel.set_xlabel('Time from stimulus onset (s)')
            
def plot_results(df_cum_sti, df_cum_res, colors, variables_combined, fig = False, ax1=False, ax2=False, upper_limit=0.3):
    if fig == False:
        fig, ([ax1,ax2])  = plt.subplots(1,2, figsize=(14, 4), sharey=True)
    for color, variable,left,right in zip(colors,variables_combined,[ax1],[ax2]):

        # Aligmnent for Stimulus cue
        real = np.array(np.mean(df_cum_sti.groupby('session').mean())) 
        # real = np.array(np.mean(df_cum_sti.groupby('session').median()))
        times = df_cum_sti
        times = np.array(times.drop(columns=['session','subject'],axis = 1).columns.astype(float))

        mean_surr = []
        df_lower = pd.DataFrame()
        df_upper = pd.DataFrame()

        df_for_boots = df_cum_sti.groupby('session').mean()
        for timepoint in times:
            mean_surr = []

            # recover the values for that specific timepoint
            try:
                array = df_for_boots[timepoint].to_numpy()
            except:
                array = df_for_boots[str(timepoint)].to_numpy()

            # iterate several times with resampling: chose X time among the same list of values
            for iteration in range(1000):
                x = np.random.choice(array, size=len(array), replace=True)
                # recover the mean of that new distribution
                mean_surr.append(np.mean(x))

            df_lower.at[0,timepoint] = np.percentile(mean_surr, 2.5)
            df_upper.at[0,timepoint] = np.percentile(mean_surr, 97.5)

        lower =  df_lower.iloc[0].values
        upper =  df_upper.iloc[0].values
        # lower =  real - 2*df_for_boots.std()
        # upper =  real + 2*df_for_boots.std()
        # lower =  df_cum_sti.quantile(0.025)
        # upper =  df_cum_sti.quantile(0.975)
        x=times

        # plt.plot(x, y_mean, label='Non-prefered Stimulus', color=color)
        left.plot(times,real, color=color)
        left.plot(x, lower, color=color, linestyle = '',alpha=0.6)
        left.plot(x, upper, color=color, linestyle = '',alpha=0.6)
        left.fill_between(x, lower, upper, alpha=0.2, color=color)
        left.axhline(y=0.0,linestyle=':',color='black')
        left.fill_betweenx(np.arange(-1,1.15,0.1), 0,0.35, color='lightgrey', alpha=.4)
        left.set_xlabel('Time from stimulus onset (s)')

        sns.despine()

        # -------------------- For Aligment to Go cue
        real = np.array(np.mean(df_cum_res.groupby('session').mean()))
        times = df_cum_res
        times = np.array(times.drop(columns=['session','subject'],axis = 1).columns.astype(float))

        mean_surr = []
        df_lower = pd.DataFrame()
        df_upper = pd.DataFrame()

        # df_for_boots = df_cum_res.loc[:, df_cum_res.columns != 'session']
        df_for_boots = df_cum_res.groupby('session').mean()
        for timepoint in times:
            mean_surr = []

            # recover the values for that specific timepoint
            try:
                array = df_for_boots[timepoint].to_numpy()
            except:
                array = df_for_boots[str(timepoint)].to_numpy()

            # iterate several times with resampling: chose X time among the same list of values
            for iteration in range(1000):
                # print(array)
                x = np.random.choice(array, size=len(array), replace=True)
                # recover the mean of that new distribution
                mean_surr.append(np.mean(x))

            df_lower.at[0,timepoint] = np.percentile(mean_surr, 2.5)
            df_upper.at[0,timepoint] = np.percentile(mean_surr, 97.5)

        lower =  df_lower.iloc[0].values
        upper =  df_upper.iloc[0].values
        # lower =  real - 2*df_for_boots.std()
        # upper =  real + 2*df_for_boots.std()
        # lower =  df_cum_res.quantile(0.025)
        # upper =  df_cum_res.quantile(0.975)

        # lower = -np.mean(df_cum_res.loc[df_cum_sti['trial_type'] ==variable].groupby(['session','fold']).sem(),axis=0)*2
        # upper = np.mean(df_cum_res.loc[df_cum_sti['trial_type'] ==variable].groupby(['session','fold']).sem(),axis=0)*2
        x=times

        # ax2.plot(x, y_mean, color=color)
        right.plot(times,real, color=color)
        right.plot(x, lower, color=color, linestyle = '',alpha=0.6)
        right.plot(x, upper, color=color, linestyle = '',alpha=0.6)
        right.fill_between(x, lower, upper, alpha=0.2, color=color)
        right.set_ylim(-0.1,upper_limit)
        right.axhline(y=0,linestyle=':',color='black')
        right.fill_betweenx(np.arange(-1.1,1.1,0.1), 0,0.2, color='lightgrey', alpha=.8)
        right.set_xlabel('Time from go cue (s)')

        sns.despine()
    #     plt.ylim(0.4,0.8)

def plot_results_shuffle_substraction(df_cum_sti, df_cum_res, df_cum_sti_shuffle, df_cum_res_shuffle, colors, variables_combined, 
                                      fig = False, ax1=False, ax2=False, upper_limit=0.4, lower_limit = -0.1, baseline=0):
    if fig == False:
        fig, ([ax1,ax2])  = plt.subplots(1,2, figsize=(14, 4), sharey=True)
        
    for color, variable,left,right in zip(colors,variables_combined,[ax1],[ax2]):
        
        df_grouped = df_cum_sti.groupby('session').mean()-df_cum_sti_shuffle.drop(columns=['session_shuffle']).groupby('session').mean()
        real = np.array(np.mean(df_grouped))
        times = np.array(np.mean(df_grouped).index).astype(float)

        mean_surr = []
        df_lower = pd.DataFrame()
        df_upper = pd.DataFrame()
        df_for_boots = df_grouped
        
        for timepoint in times:
            mean_surr = []

            # recover the values for that specific timepoint
            try:
                array = df_for_boots[timepoint].to_numpy()
            except:
                array = df_for_boots[str(timepoint)].to_numpy()

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

        # plt.plot(x, y_mean, label='Non-prefered Stimulus', color=color)
        left.plot(times,real, color=color)
        left.plot(x, lower, color=color, linestyle = '',alpha=0.6, linewidth=0)
        left.plot(x, upper, color=color, linestyle = '',alpha=0.6, linewidth=0)
        left.fill_between(x, lower, upper, alpha=0.2, color=color, linewidth=0)
        left.set_ylim(lower_limit-0.05,upper_limit+0.1)
        left.axhline(y=0.0,linestyle=':',color='black')
        left.fill_betweenx(np.arange(lower_limit,upper_limit,0.1), 0,0.45, color='lightgrey', alpha=.8)
        left.set_xlabel('Time from stimulus onset (s)')
        

        # -------------------- For Aligment to Go cue
        df_grouped = df_cum_res.groupby('session').mean()-df_cum_res_shuffle.drop(columns=['session_shuffle']).groupby('session').mean()
        real = np.array(np.mean(df_grouped))
        times = np.array(np.mean(df_grouped).index).astype(float)

        mean_surr = []
        df_lower = pd.DataFrame()
        df_upper = pd.DataFrame()
        # df_for_boots = df_cum_res.loc[:, df_cum_res.columns != 'session']
        df_for_boots = df_grouped
        for timepoint in times:
            mean_surr = []

            # recover the values for that specific timepoint
            try:
                array = df_for_boots[timepoint].to_numpy()
            except:
                array = df_for_boots[str(timepoint)].to_numpy()

            # iterate several times with resampling: chose X time among the same list of values
            for iteration in range(1000):
                # print(array)
                x = np.random.choice(array, size=len(array), replace=True)
                # recover the mean of that new distribution
                mean_surr.append(np.mean(x))

            df_lower.at[0,timepoint] = np.percentile(mean_surr, 2.5)
            df_upper.at[0,timepoint] = np.percentile(mean_surr, 97.5)

        lower =  df_lower.iloc[0].values
        upper =  df_upper.iloc[0].values
        # lower =  real - 2*df_for_boots.std()
        # upper =  real + 2*df_for_boots.std()
        # lower =  df_cum_res.quantile(0.025)
        # upper =  df_cum_res.quantile(0.975)

        # lower = -np.mean(df_cum_res.loc[df_cum_sti['trial_type'] ==variable].groupby(['session','fold']).sem(),axis=0)*2
        # upper = np.mean(df_cum_res.loc[df_cum_sti['trial_type'] ==variable].groupby(['session','fold']).sem(),axis=0)*2
        x=times

        # ax2.plot(x, y_mean, color=color)
        right.plot(times,real, color=color)
        right.plot(x, lower, color=color, linestyle = '',alpha=0.6, linewidth=0)
        right.plot(x, upper, color=color, linestyle = '',alpha=0.6, linewidth=0)
        right.fill_between(x, lower, upper, alpha=0.2, color=color, linewidth=0)
        right.set_ylim(lower_limit,upper_limit)
        right.axhline(y=0,linestyle=':',color='black')
        right.fill_betweenx(np.arange(lower_limit-0.05,upper_limit+0.1,0.1), 0,0.2, color='lightgrey', alpha=0.8)
        right.set_xlabel('Time from go cue (s)')
        
def plot_results_session_summary_substract(fig, plot, df, df_shuffle, color, variable= 'WM_roll_1', 
                                 y_range = [-0.05, 0.3], x_range = None, epoch = 'Stimulus_ON', baseline=0.5):
    
        df_loop = (df.loc[(df['trial_type'] == variable)].groupby('session').mean()
                    - df_shuffle.loc[(df_shuffle['trial_type'] == variable)].groupby('session').mean())
        df_loop.fillna(0)
        # df_loop = df.loc[(df['trial_type'] == variable)].groupby('session').mean()
                    
        # Select only columns where the column name is a number or can be transformed to a number
        numeric_columns = df_loop.columns[df_loop.columns.to_series().apply(pd.to_numeric, errors='coerce').notna()]

        real = np.array(np.mean(df_loop.groupby('session').mean()[numeric_columns])) 
        times = np.array(np.mean(df_loop[numeric_columns]).index).astype(float)

        df_results = pd.DataFrame()
        df_results['times'] = times
        df_results['real'] = real
        df_results = df_results.sort_values(by='times')

        if x_range == None:
            x_range = [min(times), max(times)]
            
        mean_surr = []
        df_lower = pd.DataFrame()
        df_upper = pd.DataFrame()

        df_for_boots = df_loop.groupby('session').mean()
        df_for_boots = df_for_boots.dropna(how='all')
        
        for timepoint in df_results['times']:
            mean_surr = []

            # recover the values for that specific timepoint
            try:
                array = df_for_boots[timepoint].to_numpy()
            except:
                array = df_for_boots[str(timepoint)].to_numpy()
                

            # iterate several times with resampling: chose X time among the same list of values
            for iteration in range(1000):
                x = np.random.choice(array, size=len(array), replace=True)
                # recover the mean of that new distribution
                mean_surr.append(np.mean(x))

            df_lower.at[0, timepoint] = np.percentile(mean_surr, 2.5)
            df_upper.at[0, timepoint] = np.percentile(mean_surr, 97.5)

        lower =  df_lower.iloc[0].values
        upper =  df_upper.iloc[0].values

        plot.plot(df_results.times,df_results.real, color=color)
        plot.fill_between(df_results.times, lower, upper,  alpha=0.2, color=color, linewidth=0)
        plot.axhline(y=baseline,linestyle=':',color='black')
        plot.set_ylim(y_range)
        plot.set_xlim(x_range)

        if epoch == 'Stimulus_ON':
            plot.set_xlabel('Time to stimulus onset (s)')
            plot.fill_betweenx(np.arange(-1,1.15,0.1), 0,0.4, color='lightgrey', alpha=1)
        else:
            plot.set_xlabel('Time to go cue (s)')
            plot.fill_betweenx(np.arange(-1,1.15,0.1), 0,0.2, color='lightgrey', alpha=1)

        sns.despine()

def plot_results_session_summary(fig, plot, df, colors, variables_combined = ['WM_roll_1', 'RL_roll_1'], 
                                 y_range = [], x_range = None, epoch = 'Stimulus_ON', baseline=0.5):
    
    for color, variable, ax in zip(colors, variables_combined, np.repeat(plot, len(variables_combined))):
        try:
            df_loop = df.loc[(df['trial_type'] == variable)]
        except:
            df_loop = df

        # Select only columns where the column name is a number or can be transformed to a number
        numeric_columns = df_loop.columns[df_loop.columns.to_series().apply(pd.to_numeric, errors='coerce').notna()]

        real = np.array(np.mean(df_loop.groupby('session').mean()[numeric_columns])) 
        times = df_loop[numeric_columns].columns.astype(float)


        df_results = pd.DataFrame()
        df_results['times'] = times
        df_results['real'] = real
        df_results = df_results.sort_values(by='times')
        

        if x_range == None:
            x_range = [min(times), max(times)]
            
        mean_surr = []
        df_lower = pd.DataFrame()
        df_upper = pd.DataFrame()

        df_for_boots = df_loop.groupby('session').mean()
        for timepoint in df_results['times'].values:
            mean_surr = []

            # recover the values for that specific timepoint
            try:
                array = df_for_boots[timepoint].to_numpy()
            except:
                array = df_for_boots[str(timepoint)].to_numpy()

            # iterate several times with resampling: chose X time among the same list of values
            for iteration in range(1000):
                x = np.random.choice(array, size=len(array), replace=True)
                # recover the mean of that new distribution
                mean_surr.append(np.mean(x))

            df_lower.at[0, timepoint] = np.percentile(mean_surr, 2.5)
            df_upper.at[0, timepoint] = np.percentile(mean_surr, 97.5)

        lower =  df_lower.iloc[0].values
        upper =  df_upper.iloc[0].values

        ax.plot(df_results.times,df_results.real, color=color)
        ax.fill_between(df_results.times, lower, upper, alpha=0.2, color=color, linewidth=0)
        ax.axhline(y=baseline,linestyle=':',color='black')
        ax.set_ylim(y_range)
        ax.set_xlim(x_range)

        if epoch == 'Stimulus_ON':
            ax.set_xlabel('Time to stimulus onset (s)')
            ax.fill_betweenx(np.arange(-1,1.15,0.1), 0,0.4, color='grey', alpha=.4)
        else:
            ax.set_xlabel('Time to go cue (s)')
            ax.fill_betweenx(np.arange(-1,1.15,0.1), 0,0.2, color='grey', alpha=.4)

        sns.despine()
