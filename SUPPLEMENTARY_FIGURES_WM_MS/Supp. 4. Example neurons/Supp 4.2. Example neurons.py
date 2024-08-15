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

save_path = 'G:/Mi unidad/WORKING_MEMORY/PAPER/WM_manuscript_FIGURES/SUPPLEMENTARY_FIGURES_WM_MS/Supp. 4.2. Example neurons'
path = 'G:/Mi unidad/WORKING_MEMORY/PAPER/ANALYSIS_figures/'

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
fig = plt.figure(figsize=(18*cm, 15*cm))
gs = gridspec.GridSpec(nrows=8, ncols=8, figure=fig)

# Create the subplots
a1 = fig.add_subplot(gs[0, 0:2])
a2 = fig.add_subplot(gs[1, 0:2])

b1 = fig.add_subplot(gs[0, 2:4])
b2 = fig.add_subplot(gs[1, 2:4])

c1 = fig.add_subplot(gs[0, 4:6])
c2 = fig.add_subplot(gs[1, 4:6])

d1 = fig.add_subplot(gs[0, 6:8])
d2 = fig.add_subplot(gs[1, 6:8])

e1 = fig.add_subplot(gs[2, 0:2])
e2 = fig.add_subplot(gs[3, 0:2])

f1 = fig.add_subplot(gs[2, 2:4])
f2 = fig.add_subplot(gs[3, 2:4])

g1 = fig.add_subplot(gs[2, 4:6])
g2 = fig.add_subplot(gs[3, 4:6])

h1 = fig.add_subplot(gs[2, 6:8])
h2 = fig.add_subplot(gs[3, 6:8])

i1 = fig.add_subplot(gs[4, 0:2])
i2 = fig.add_subplot(gs[5, 0:2])

j1 = fig.add_subplot(gs[4, 2:4])
j2 = fig.add_subplot(gs[5, 2:4])

k1 = fig.add_subplot(gs[4, 4:6])
k2 = fig.add_subplot(gs[5, 4:6])

l1 = fig.add_subplot(gs[4, 6:8])
l2 = fig.add_subplot(gs[5, 6:8])

m1 = fig.add_subplot(gs[6, 0:2])
m2 = fig.add_subplot(gs[7, 0:2])

n1 = fig.add_subplot(gs[6, 2:4])
n2 = fig.add_subplot(gs[7, 2:4])

o1 = fig.add_subplot(gs[6, 4:6])
o2 = fig.add_subplot(gs[7, 4:6])

p1 = fig.add_subplot(gs[6, 6:8])
p2 = fig.add_subplot(gs[7, 6:8])

# fig.text(0.01, 0.99, 'a', fontsize=10, fontweight='bold', va='top')
# fig.text(0.5, 0.99, 'b', fontsize=10, fontweight='bold', va='top')
# fig.text(0.01, 0.75, 'c', fontsize=10, fontweight='bold', va='top')
# fig.text(0.5, 0.75, 'f', fontsize=10, fontweight='bold', va='top')
# fig.text(0.75, 0.75, 'g', fontsize=10, fontweight='bold', va='top')

# fig.text(0.01, 0.51, 'd', fontsize=10, fontweight='bold', va='top')
# fig.text(0.01, 0.4, 'e', fontsize=10, fontweight='bold', va='top')
# fig.text(0.01, 0.28, 'f', fontsize=10, fontweight='bold', va='top')
# fig.text(0.01, 0.17, 'g', fontsize=10, fontweight='bold', va='top')

# fig.text(0.51, 0.51, 'i', fontsize=10, fontweight='bold', va='top')
# fig.text(0.51, 0.4, 'j', fontsize=10, fontweight='bold', va='top')
# fig.text(0.51, 0.28, 'k', fontsize=10, fontweight='bold', va='top')
# fig.text(0.51, 0.17, 'l', fontsize=10, fontweight='bold', va='top')

# -----------------############################## A Panel #################################-----------------------

def convolveandplot(df, upper_plot, lower_plot, variable='reward_side', cluster_id = 153, delay = 10, labels=['Correct right stimulus','Correct left stimulus'],
                   colors=[COLORRIGHT,COLORLEFT], align = 'Stimulus_ON', j=1, alpha=1, label=False, spikes=True, kernel=200):
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
    
    y = np.arange(0,100,0.1)     
    panel.fill_betweenx(y, cue_on,cue_off, color='lightgrey', alpha=1, linewidth=0)  
    panel.fill_betweenx(y, cue_off+delay,cue_off+delay+.2, color='lightgrey', alpha=1, linewidth=0)
    panel.set_ylim(0,max( df_results['firing'])+np.mean(df_results['error']))  
    panel.set_ylabel('Firing rate (Hz)')
    # axis labels and legend
    lower_plot.legend(frameon=False)
    lower_plot.get_legend().remove()
    if label:
        panel.set_xlabel('Time from stimulus onset (s)')
    # panel.set_ylabel('Firing rate (spks/s)')
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
    upper_plot.set(xlabel=None)
    
    y = np.arange(0,j+1,0.1)
    panel.fill_betweenx(y, cue_on,cue_off, color='lightgrey', alpha=1, linewidth=0)
    panel.fill_betweenx(y, cue_off+delay,cue_off+delay+.2, color='lightgrey', alpha=1, linewidth=0)
    
    panel.locator_params(nbins=5) 
    upper_plot.get_xaxis().set_visible(False)
    
    return j

def plot_decoder(left, df_cum_sti,baseline=0.5,individual_sessions=False, colors=['black'], upper_limit=0.2, variables_combined=['WM_roll_1']):
    for color, variable,left in zip(colors,variables_combined,[left]):
        if individual_sessions == True:
            # Aligmnent for Stimulus cue - sessions separately
            real = df_cum_sti.groupby('session').median().reset_index()
            try:
                times = np.array(df_cum_sti.columns[:-4]).astype(float)
            except:
                times = np.array(df_cum_sti.columns[1:]).astype(float)
    
        
            x=times
            for i in range(len(real)):
                left.plot(times,real.iloc[i][1:-1], color=color,alpha=0.1)
                
        # Aligmnent for Stimulus cue
        real = np.array(df_cum_sti.loc[:, (df_cum_sti.columns != 'session_shuffle')
                                       &(df_cum_sti.columns != 'fold')].mean())
        try:
            times = np.array(df_cum_sti.columns[:-4]).astype(float)
            time_points = df_cum_sti.columns[:-4]

        except:
            times = np.array(df_cum_sti.columns[1:]).astype(float)
            time_points = df_cum_sti.columns[1:]

        mean_surr = []
        df_lower = pd.DataFrame()
        df_upper = pd.DataFrame()
    
        # df_for_boots = df_cum_sti.loc[:, df_cum_sti.columns != 'session_shuffle']
        df_for_boots = df_cum_sti.loc[:, (df_cum_sti.columns != 'session_shuffle')
                                       &(df_cum_sti.columns != 'fold')].groupby('session').mean().reset_index()
        for timepoint in time_points:
            mean_surr = []
    
            # recover the values for that specific timepoint
            array = df_for_boots[timepoint].to_numpy()

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
        
        # lower =  real - 2*df_for_boots.std()
        # upper =  real + 2*df_for_boots.std()
        lower =  df_cum_sti.quantile(0.025)
        upper =  df_cum_sti.quantile(0.975)
    
        left.plot(times,real, color=color)

        left.fill_betweenx(np.arange(-baseline-0.1,baseline+.5,0.1), 0,0.35, color='lightgrey', alpha=1, linewidth=0)
        left.fill_betweenx(np.arange(-baseline-0.1,baseline+.5,0.1), 3.35,3.55, color='lightgrey', alpha=1, linewidth=0)
        left.set_ylim(baseline-0.1,upper_limit+baseline)
        left.axhline(y=baseline,linestyle=':',color='black')
        left.set_xlabel('Time from Cue onset (s)')
        left.set_ylabel('Decoder\n accuracy')


def new_convolve(nx,df, kernel=200):
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

def plotsingledelay(df_cum_sti, panel, colors, variables_combined, delay):
    baseline = 0.5
    y_upper=baseline
    y_lower=baseline

    for color, variable in zip(colors,variables_combined):
    
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
        panel.fill_betweenx(np.arange(-baseline-0.1,baseline+.45,0.1), delay+.35,delay+.55, color='lightgrey', alpha=1, linewidth=0)
        panel.set_ylim(y_lower,y_upper+0.05)
        if panel=='crimson':
            panel.set_xlabel('Time to stimulus onset (s)')
            
# ----------------------------------------------------------------------------------------------------------------
            
# -----------------############################## A Panel - Crossdecoder for 3s  #######################-----------------------

# file_name = 'crossdecoder_WM_roll_1_3s_alignedstimulus_nosubstract'

# ----------------------------------------------------------------------------------------------------------------

# -----------------############################## C Panel - Single neuron example ############-----------------------

file_name = 'E20_2022-02-13_15-10-51_neuron_81'
df = pd.read_csv(path+file_name+'.csv', index_col=0)

delay = 10
cluster_id = df.cluster_id.unique()[0]

colors = [COLORRIGHT,COLORLEFT]
# colors = ['darkgreen','crimson']
labels = ['Right stimulus','Left stimulus']
# labels = ['Correct','Incorrect']
align = 'Stimulus_ON'

j=1
temp_df = df.loc[(df.WM_roll >0.6)&(df.hit ==1)]
j = convolveandplot(temp_df, a1, a2, variable='reward_side', cluster_id = cluster_id, delay = delay, j=j)
a1.set_title(file_name)

# ----------------------------------------------------------------------------------------------------------------

# -----------------############################## C Panel - Single neuron example ############-----------------------

file_name = 'E14_2021-04-02_12-53-42_neuron_361'
df = pd.read_csv(path+file_name+'.csv', index_col=0)

delay = 10
cluster_id = df.cluster_id.unique()[0]

j=1
temp_df = df.loc[(df.WM_roll >0.6)&(df.hit ==1)]
j = convolveandplot(temp_df, b1, b2, variable='reward_side', cluster_id = cluster_id, delay = delay, j=j)
b1.set_title(file_name)

# -----------------############################## C Panel - Single neuron example ############-----------------------

file_name = 'E17_2022-02-01_17-02-16_neuron_304'
df = pd.read_csv(path+file_name+'.csv', index_col=0)

delay = 10
cluster_id = df.cluster_id.unique()[0]

j=1
temp_df = df.loc[(df.WM_roll >0.6)&(df.hit ==1)]
j = convolveandplot(temp_df, c1, c2, variable='reward_side', cluster_id = cluster_id, delay = delay, j=j)
c1.set_title(file_name)

# -----------------############################## C Panel - Single neuron example ############-----------------------

file_name = 'E17_2022-02-02_17-13-06_neuron_33'
df = pd.read_csv(path+file_name+'.csv', index_col=0)

delay = 10
cluster_id = df.cluster_id.unique()[0]

j=1
temp_df = df.loc[(df.WM_roll >0.6)&(df.hit ==1)]
j = convolveandplot(temp_df, d1, d2, variable='reward_side', cluster_id = cluster_id, delay = delay, j=j)
d1.set_title(file_name)

# -----------------############################## C Panel - Single neuron example ############-----------------------

file_name = 'E17_2022-02-02_17-13-06_neuron_265'
df = pd.read_csv(path+file_name+'.csv', index_col=0)

delay = 1
cluster_id = df.cluster_id.unique()[0]

j=1
temp_df = df.loc[(df.WM_roll >0.6)&(df.hit ==1)]
j = convolveandplot(temp_df, e1, e2, variable='reward_side', cluster_id = cluster_id, delay = delay, j=j)
e1.set_title(file_name)


# -----------------############################## C Panel - Single neuron example ############-----------------------

file_name = 'E17_2022-02-02_17-13-06_neuron_233'
df = pd.read_csv(path+file_name+'.csv', index_col=0)

delay = 3
cluster_id = df.cluster_id.unique()[0]

j=1
temp_df = df.loc[(df.WM_roll >0.6)&(df.hit ==1)]
j = convolveandplot(temp_df, f1, f2, variable='reward_side', cluster_id = cluster_id, delay = delay, j=j)
f1.set_title(file_name)


# -----------------############################## C Panel - Single neuron example ############-----------------------

file_name = 'E17_2022-02-02_17-13-06_neuron_288'
df = pd.read_csv(path+file_name+'.csv', index_col=0)

delay = 3
cluster_id = df.cluster_id.unique()[0]

j=1
temp_df = df.loc[(df.WM_roll >0.6)&(df.hit ==1)]
j = convolveandplot(temp_df, g1, g2, variable='reward_side', cluster_id = cluster_id, delay = delay, j=j)
g1.set_title(file_name)


# -----------------############################## C Panel - Single neuron example ############-----------------------

file_name = 'E04_2021-03-30_11-20-16_neuron_169'
df = pd.read_csv(path+file_name+'.csv', index_col=0)

delay = 10
cluster_id = df.cluster_id.unique()[0]

j=1
temp_df = df.loc[(df.WM_roll >0.6)&(df.hit ==1)]
j = convolveandplot(temp_df, h1, h2, variable='reward_side', cluster_id = cluster_id, delay = delay, j=j)

h1.get_xaxis().set_visible(False)
h1.set_title(file_name)

# -----------------############################## C Panel - Single neuron example ############-----------------------

file_name = 'E14_2021-04-02_12-53-42_neuron_511'
df = pd.read_csv(path+file_name+'.csv', index_col=0)

delay = 10
cluster_id = df.cluster_id.unique()[0]

j=1
temp_df = df.loc[(df.WM_roll >0.6)&(df.hit ==1)]
j = convolveandplot(temp_df, i1, i2, variable='reward_side', cluster_id = cluster_id, delay = delay, j=j)
i1.set_title(file_name)

# -----------------############################## C Panel - Single neuron example ############-----------------------

file_name = 'E22_2022-01-22_17-09-15_neuron_246'
df = pd.read_csv(path+file_name+'.csv', index_col=0)

delay = 10
cluster_id = df.cluster_id.unique()[0]

j=1
temp_df = df.loc[(df.WM_roll >0.6)&(df.hit ==1)]
j = convolveandplot(temp_df, j1, j2, variable='reward_side', cluster_id = cluster_id, delay = delay, j=j)
j1.set_title(file_name)

# -----------------############################## C Panel - Single neuron example ############-----------------------

file_name = 'E22_2022-01-13_16-34-24_neuron_381'
df = pd.read_csv(path+file_name+'.csv', index_col=0)

delay = 10
cluster_id = df.cluster_id.unique()[0]

j=1
temp_df = df.loc[(df.WM_roll >0.6)&(df.hit ==1)]
j = convolveandplot(temp_df, k1, k2, variable='reward_side', cluster_id = cluster_id, delay = delay, j=j)
k1.set_title(file_name)

# -----------------############################## C Panel - Single neuron example ############-----------------------

file_name = 'E22_2022-01-14_16-50-37_neuron_16'
df = pd.read_csv(path+file_name+'.csv', index_col=0)

delay = 10
cluster_id = df.cluster_id.unique()[0]

j=1
temp_df = df.loc[(df.WM_roll >0.6)&(df.hit ==1)]
j = convolveandplot(temp_df, l1, l2, variable='reward_side', cluster_id = cluster_id, delay = delay, j=j, label=True)
l1.set_title(file_name)


# -----------------############################## C Panel - Single neuron example ############-----------------------

file_name = 'E20_2022-02-14_16-01-30_neuron_204'
df = pd.read_csv(path+file_name+'.csv', index_col=0)

delay = 10
cluster_id = df.cluster_id.unique()[0]

j=1
temp_df = df.loc[(df.WM_roll >0.6)&(df.hit ==1)]
j = convolveandplot(temp_df, m1, m2, variable='reward_side', cluster_id = cluster_id, delay = delay, j=j, label=True)
m1.set_title(file_name)

# -----------------############################## C Panel - Single neuron example ############-----------------------

file_name = 'E20_2022-03-01_16-11-01_neuron_88'
df = pd.read_csv(path+file_name+'.csv', index_col=0)

delay = 10
cluster_id = df.cluster_id.unique()[0]

j=1
temp_df = df.loc[(df.WM_roll >0.6)&(df.hit ==1)]
j = convolveandplot(temp_df, n1, n2, variable='reward_side', cluster_id = cluster_id, delay = delay, j=j, label=True)
o1.set_title(file_name)

# -----------------############################## C Panel - Single neuron example ############-----------------------

file_name = 'E20_2022-03-01_16-11-01_neuron_95'
df = pd.read_csv(path+file_name+'.csv', index_col=0)

delay = 10
cluster_id = df.cluster_id.unique()[0]

j=1
temp_df = df.loc[(df.WM_roll >0.6)&(df.hit ==1)]
j = convolveandplot(temp_df, o1, o2, variable='reward_side', cluster_id = cluster_id, delay = delay, j=j, label=True)

# -----------------############################## C Panel - Single neuron example ############-----------------------

file_name = 'E22_2022-01-14_16-50-37_neuron_56'
df = pd.read_csv(path+file_name+'.csv', index_col=0)

delay = 10
cluster_id = df.cluster_id.unique()[0]

j=1
temp_df = df.loc[(df.WM_roll >0.6)&(df.hit ==1)]
j = convolveandplot(temp_df, p1, p2, variable='reward_side', cluster_id = cluster_id, delay = delay, j=j, label=True)
p1.set_title(file_name)

# 'E22_2022-01-14_16-50-37_neuron_56' Delay
# 'E22_2022-01-14_16-50-37_neuron_47' Choice
# 'E22_2022-01-14_16-50-37_neuron_23' Choice

# Show the figure
plt.subplots_adjust(left=0.07,
                    bottom=0.07,
                    right=0.97,
                    top=0.97,
                    wspace=1.0,
                    hspace=0.25)

sns.despine()
plt.savefig(save_path+'/Supp 4.2. Example neurons.svg', bbox_inches='tight',dpi=300)
plt.savefig(save_path+'/Supp 4.2. Example neurons.png', bbox_inches='tight',dpi=300)

plt.show()