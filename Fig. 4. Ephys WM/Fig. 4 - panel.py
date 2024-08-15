
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
# from datahandler import Utils
from neo.core import SpikeTrain
from quantities import ms, s, Hz
from elephant.statistics import mean_firing_rate
from elephant.statistics import time_histogram, instantaneous_rate
from elephant.kernels import GaussianKernel
from elephant.statistics import mean_firing_rate
from cycler import cycler

utilities = 'G:/Mi unidad/WORKING_MEMORY/PAPER/WM_manuscript_FIGURES/'
os.chdir(utilities)
import functions as plots

save_path = 'G:/Mi unidad/WORKING_MEMORY/PAPER/WM_manuscript_FIGURES/Fig. 4. Ephys WM/'
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
fig = plt.figure(figsize=(15*cm, 30.*cm))
gs = gridspec.GridSpec(nrows=6, ncols=9, figure=fig)

# Create the subplots
a = fig.add_subplot(gs[0:3, 4:9])
b = fig.add_subplot(gs[3, 4:8])
c = fig.add_subplot(gs[4, 4:8])
d = fig.add_subplot(gs[5, 4:8])

e = fig.add_subplot(gs[2, 0:4])
f = fig.add_subplot(gs[3, 0:4])

g1 = fig.add_subplot(gs[4, 0:4])

h1 = fig.add_subplot(gs[1, 0:4])
h2 = fig.add_subplot(gs[0, 0:4])


fig.text(0.01, 0.99, 'a', fontsize=10, fontweight='bold', va='top')
fig.text(0.5, 0.99, 'b', fontsize=10, fontweight='bold', va='top')
fig.text(0.01, 0.75, 'c', fontsize=10, fontweight='bold', va='top')
fig.text(0.5, 0.75, 'f', fontsize=10, fontweight='bold', va='top')
fig.text(0.75, 0.75, 'g', fontsize=10, fontweight='bold', va='top')

fig.text(0.51, 0.51, 'd', fontsize=10, fontweight='bold', va='top')
fig.text(0.51, 0.4, 'e', fontsize=10, fontweight='bold', va='top')
fig.text(0.51, 0.28, 'f', fontsize=10, fontweight='bold', va='top')
fig.text(0.51, 0.17, 'g', fontsize=10, fontweight='bold', va='top')

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
    neuron['time_centered'] = np.round(neuron.time_centered/1000, 2) #### estos es importante
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

def plot_decoder(left, df_cum_sti, df_shuffle, baseline=0.5,individual_sessions=False, colors=['black'], upper_limit=0.2, variables_combined=['WM_roll_1']):
    for color, variable,left in zip(colors,variables_combined,[left]):
        if individual_sessions == True:
            # Aligmnent for Stimulus cue - sessions separately
            real = df_cum_sti.groupby('session').median().reset_index()
            try:
                times = np.array(df_cum_sti.columns[:-4]).astype(float)
            except:
                times = np.array(df_cum_sti.columns[1:]).astype(float)
        
            left.set_xlabel('Time (s) to Cue')
        
            x=times
            for i in range(len(real)):
                left.plot(times,real.iloc[i][1:-1], color=color,alpha=0.1)
                
        # Aligmnent for Stimulus cue
        real = np.array(df_cum_sti.loc[:, (df_cum_sti.columns != 'session_shuffle')
                                       &(df_cum_sti.columns != 'fold')&(df_cum_sti.columns != 'train')&(df_cum_sti.columns != 'session')
                                       &(df_cum_sti.columns != 'subject')].mean())
        
        df_shuffle = df_shuffle.groupby('times').median().reset_index()
        df_shuffle_mean = np.array(df_shuffle.loc[:, (df_shuffle.columns != 'times') &(df_shuffle.columns != 'fold')].mean(axis=1))
        
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
        left.plot(x, lower-df_shuffle_mean, color=color, linestyle = '',alpha=0.6, linewidth=0)
        left.plot(x, upper-df_shuffle_mean, color=color, linestyle = '',alpha=0.6, linewidth=0)
        left.fill_between(x, lower-df_shuffle_mean, upper-df_shuffle_mean, alpha=0.2, color=color, linewidth=0)
        
        lower =  df_cum_sti.quantile(0.025)
        upper =  df_cum_sti.quantile(0.975)
    
        left.plot(times,real-df_shuffle_mean, color=color)

        left.fill_betweenx(np.arange(-baseline-0.1,baseline+.5,0.1), 0,0.35, color='lightgrey', alpha=1, linewidth=0)
        left.fill_betweenx(np.arange(-baseline-0.1,baseline+.5,0.1), 3.35,3.55, color='lightgrey', alpha=1, linewidth=0)
        left.set_ylim(baseline-0.1,upper_limit+baseline)
        left.axhline(y=baseline,linestyle=':',color='black')
        left.set_xlabel('Time from Cue onset (s)')
        left.set_ylabel('Decoding\n accuracy')

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
            panel.set_xlabel('Time to Cue onset (s)')
            
# ----------------------------------------------------------------------------------------------------------------
            
# -----------------############################## A Panel - Crossdecoder for 3s  #######################-----------------------

file_name = 'crossdecoder_WM_roll1_3s_r0.25_choice_Stimulus_ON_substraction_V3'
df_animal_shuffle = pd.read_csv(path + file_name+'_shuffle.csv', index_col = 0)
df_animal_sti = pd.read_csv(path + file_name+'_sti.csv', index_col = 0)

color= sns.diverging_palette(220, 20, as_cmap=True)

panel = a

df_shuffle_mean = pd.DataFrame()
for epoch in df_animal_shuffle.train.unique():
    df_new=pd.DataFrame()
    # df_shuffle_upper[epoch] = df_new.loc[:, (df_new.columns != 'fold')&(df_new.columns != 'times')].mean(axis=1)
    df_shuffle_mean[epoch] = df_animal_shuffle.loc[(df_animal_shuffle.train == epoch)].groupby(['times']).mean().drop(columns='fold').mean(axis=1).values
df_shuffle_mean.index = df_animal_shuffle.groupby('times').mean().index

df_new = df_animal_sti.loc[:, df_animal_sti.columns != 'fold'].groupby(['subject','train']).mean()
df_new.reset_index(inplace=True)
df_new = df_new.groupby('train').mean()
df_new = df_new.reindex(index=df_animal_sti.train.unique())

df_shuffle_mean.index = df_new.columns
sns.heatmap(df_new-df_shuffle_mean.T, fmt='', linewidth=0.0, rasterized=True, square=True, vmin=-0.1, vmax=0.3, center=0.0, ax=panel, xticklabels=df_new.columns).invert_yaxis()

# sns.heatmap(df_new, fmt='', linewidth=0.0, rasterized=True, square=True, vmin=-0.1, vmax=0.3, center=0.0, ax=panel, xticklabels=df_new.columns).invert_yaxis()
panel.set_xticklabels(["-2",'',"","","","","","","0",'',"","","","","","","2","","","","","","","","4","","","","","","","","6","","","","","","","","8"])
panel.set_yticklabels(["-2",'',"","","0",'',"","","2","","","","4","","","","6","","","","8"])

# panel.set_yticklabels(["-2",'',"","","0",'',"","","2",'',"","","4",'',"","","6",'',"","","8"])
panel.set_xlabel("Test")
panel.set_ylabel("Train")

# Recover the diagonal for all the animals
first=True
df_temp=pd.DataFrame()
train_value_list = df_animal_sti.train.unique()

for train_value in train_value_list:
    real_value = np.around((float(train_value.split('_')[0]) + float(train_value.split('_')[1]))/2,3)
    # real_value = float(train_value.split('_')[0]) + float(train_value.split('_')[1]))/2
    if real_value == 7.875:
        continue
    elif real_value == 7.9:
        continue
    df_temp = df_animal_sti.loc[df_animal_sti.train==train_value].groupby('session')[[str(real_value)]].mean().reset_index()
    if first:
        df_diagonal = df_temp
        first=False
    else:
        df_diagonal = pd.merge(df_diagonal, df_temp, on=['session'])

# Customize the legend
cbar = panel.collections[0].colorbar
cbar.set_label("Color Scale", rotation=0) 
# ----------------------------------------------------------------------------------------------------------------
# -----------------############################## B Panel - Example segments of trained groups ############-----------------------

# This when we want to recover the traces of the crossdecoder
# for panel, df_cum_sti, upper_limit in zip([b,c,d,d1],[df_animal_sti.loc[df_animal_sti.train == '0.0_0.25'],
#                                       df_animal_sti.loc[df_animal_sti.train == '2.75_3.0'],
#                                       df_animal_sti.loc[df_animal_sti.train == '3.75_4.0'],
#                                       df_diagonal],[0.2,0.2,0.4,0.4]):
#     plot_decoder(panel, df_cum_sti,baseline=0.5,individual_sessions=False, upper_limit=upper_limit)
#     panel.margins(x=0)
    
for panel, df_cum_sti, df_shuffle, upper_limit in zip([b,c,d],[df_animal_sti.loc[df_animal_sti.train == '0.0_0.25'],
                                      df_animal_sti.loc[df_animal_sti.train == '3.0_3.25'],
                                      df_animal_sti.loc[df_animal_sti.train == '3.5_3.75']],
                                      [df_animal_shuffle.loc[df_animal_shuffle.train == '0.25_0.5'],
                                      df_animal_shuffle.loc[df_animal_shuffle.train == '3.0_3.25'],
                                      df_animal_shuffle.loc[df_animal_shuffle.train == '3.5_3.75']],  
                                      [0.25,0.25,0.4]):
#     plot_decoder(panel, df_cum_sti,baseline=0.0,individual_sessions=False, upper_limit=upper_limit)
#     panel.margins(x=0)
    
# for panel, df_cum_sti, upper_limit in zip([b,c,d],[df_animal_sti.loc[df_animal_sti.train == '-0.0_0.2'],
#                                       df_animal_sti.loc[df_animal_sti.train == '3.0_3.2'],
#                                       df_animal_sti.loc[df_animal_sti.train == '3.6_3.8'],
#                                       ],[0.2,0.2,0.4]):
    
    plot_decoder(panel, df_cum_sti, df_shuffle, baseline=0.0 ,individual_sessions=False, upper_limit=upper_limit)
    panel.margins(x=0)
    panel.locator_params(nbins=5) 
    sns.despine(offset=2,  ax=panel)
    panel.set_xlim(-2, 8)
    
b.text(x=0.0, y=0.7, s='Stimulus code',weight="bold", fontsize=7)
c.text(x=0.0, y=0.7,s='Delay code',weight="bold", fontsize=7)
d.text(x=0.0, y=0.7,s='Response code',weight="bold", fontsize=7)

# ----------------------------------------------------------------------------------------------------------------

# -----------------############################## C Panel - Single neuron example ############-----------------------

session = pd.read_csv('C:/Users/Tiffany/Documents/Ephys/summary_complete/E20_2022-02-14_16-01-30.csv', sep=',',index_col=0)
session['trial_start'] = session.trial.min()
session['trial_end'] = session.trial.max()

file_name = 'single_neuron_10s'
df = pd.read_csv(path+file_name+'.csv', index_col=0)
df = df.loc[df.WM_roll > 0.6]

delay = 10
cluster_id = 153

colors = [COLORRIGHT,COLORLEFT]
# colors = ['darkgreen','crimson']
labels = ['Right stimulus','Left stimulus']
# labels = ['Correct','Incorrect']
align = 'Stimulus_ON'

j=1
temp_df = df.loc[(df.WM_roll >0.6)&(df.hit ==1)]
j = convolveandplot(temp_df, e, f, variable='reward_side', cluster_id = cluster_id, delay = delay, j=j)

# neuron = new_convolve(df, df)
# neuron = neuron.loc[neuron.hit == 1]
# # neuron = neuron.loc[neuron.delay != 0.1] # Remove trials with no delay if the studied segment is that one. 

# # Align the data to the targeted align
# neuron['time_centered'] = neuron['times'] - neuron[align] 
# neuron['time_centered'] = np.round(neuron.time_centered/1000, 2) #### estos es importante!!
# neuron['firing_'] = neuron['firing']*1000

# df_results = pd.DataFrame(dtype=float)
# df_results['firing'] = neuron.loc[(neuron.time_centered <= stop)&(neuron.delay == delay)].groupby(['time_centered','reward_side'])['firing_'].mean()
# df_results['error'] = neuron.loc[(neuron.time_centered <= stop)&(neuron.delay == delay)].groupby(['time_centered','reward_side'])['firing_'].std()
# df_results.reset_index(inplace=True)

# for condition,color,name in zip([1,0],[COLORRIGHT,COLORLEFT],['Right stimulus','Left stimulus']):
#     y_mean= df_results[df_results.reward_side==condition].firing
#     error = 0.5*df_results[df_results.reward_side==condition].error
#     lower = y_mean - error
#     upper = y_mean + error
#     x=df_results[df_results.reward_side==condition].time_centered

#     panel = f
#     panel.plot(x, y_mean, label=name,color=color)
#     panel.plot(x, lower, color=color, alpha=0.1)
#     panel.plot(x, upper, color=color, alpha=0.1)
#     panel.fill_between(x, lower, upper, alpha=0.2, color=color)

# panel.set_xlim(start,stop)  
# panel.set_ylim(0,60)  
# panel.set_xlabel('Time (s) from response onset')
# y = np.arange(0,60,0.1)     
# panel.fill_betweenx(y, cue_on,cue_off, color='grey', alpha=.4)  
# panel.fill_betweenx(y, cue_off+delay,cue_off+delay+.2, color='beige', alpha=.6)

# # axis labels and legend
# panel.legend(frameon=False)    
# panel.set_xlabel('Time (s) from stimulus onset')
# panel.set_ylabel('Firing rate (s/s)')
        
# # Plot the rater plot
# SpikesRight = (df.loc[(df.reward_side == 1)&(df.delay == delay)])
# SpikesLeft = (df.loc[(df.reward_side == 0)&(df.delay == delay)])

# SpikesRight['a_'+align] = SpikesRight['fixed_times'] - SpikesRight['Stimulus_ON'] 
# SpikesLeft['a_'+align] = SpikesLeft['fixed_times'] - SpikesLeft['Stimulus_ON'] 

# panel = e
# trial=1
# j=1
# spikes = []
# trial_repeat = []
# for i in range(len(SpikesRight)):
#     # Plot for licks for left trials
#     if SpikesRight.trial.iloc[i] != trial:
#         panel.plot(spikes,trial_repeat, '|', markersize=0.5, color=COLORRIGHT, zorder=1)
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
#         panel.plot(spikes,trial_repeat, '|', markersize=0.5, color=COLORLEFT, zorder=1)
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
# panel.axes.get_xaxis().set_visible(False)

# -----------------############################## C Panel  -  Training weights  #################################-----------------------
# file_name = 'weights for the modelling_Lick'
# df_weights = pd.read_csv(path+file_name+'.csv')

# panel = g1
# sns.scatterplot(x='Stimulus_ON-0-0.3', y='Stimulus_ON-0.5-1', data=df_weights, ax=panel,alpha=0.5)
# sns.regplot(x='Stimulus_ON-0-0.3', y='Stimulus_ON-0.5-1', data=df_weights, ax=panel, marker='')
# panel.set_xlabel('Stimulus')
# panel.set_ylabel('Early Delay')
# panel.plot([-3,3], [-3,3], ls="--", c=".3")

# panel = g2
# sns.scatterplot(x='Delay_OFF--0.5-0', y='Stimulus_ON-0.5-1', data=df_weights, ax=panel,alpha=0.5)
# sns.regplot(x='Delay_OFF--0.5-0', y='Stimulus_ON-0.5-1', data=df_weights, ax=panG:\Mi unidad\WORKING_MEMORY\EXPERIMENTS\ELECTROPHYSIOLOGY\ANALYSIS\Fig6_panel_i_WM_RL_previous_choice.ipynbel, marker='')
# panel.set_xlabel('Late Delay')
# panel.set_ylabel('Early Delay')
# panel.plot([-3,3], [-3,3], ls="--", c=".3")

# panel = g3
# sns.scatterplot(x='Delay_OFF--0.5-0', y='Lick_ON--0.1-0.4', data=df_weights, ax=panel,alpha=0.5)
# sns.regplot(x='Delay_OFF--0.5-0', y='Lick_ON--0.1-0.4', data=df_weights, ax=panel, marker='')
# panel.set_xlabel('Late delay')
# panel.set_ylabel('Response')
# panel.set_ylabel('Response')
# panel.plot([-3,3], [-3,3], ls="--", c=".3")

file_name = 'parsed_weights for the modelling_late3'
df_temp = pd.read_csv(path+file_name+'.csv', index_col=0)

panel = g1
# color_list = ['lightblue','darkgrey', 'blue','lightgrey','grey']
color_list = ['darkgrey','darkgrey', 'darkgrey', 'darkgrey']
orderlist=['Stim x Late Delay','Late Delay x Response','Early x Late Delay','Late Delay x Late Delay*',]

sns.stripplot(x='condition',y='vector', data=df_temp, order=orderlist, jitter=0.2, size=3, palette = color_list, edgecolor='white', ax=panel, linewidth=0.1)

color_list = ['lightgrey','lightgrey', 'lightgrey','lightgrey']
# sns.violinplot(x='condition',y='vector', data=df_temp, order=orderlist, legend=False, palette = color_list,ax=panel,linewidth=0, width=1)
sns.boxplot(x='condition',y='vector', data=df_temp, order=orderlist, palette = ['black','black', 'black','black'],ax=panel, width = 0.4)

panel.hlines(y=0., xmin=-0.5, xmax=len(color_list)-0.5, linestyle=':')
panel.set_xlabel('')
panel.set_ylim(-1.1,1.1)
# panel.set_xticklabels(['ED-Stim','ED-LD','LD-Res'])
panel.set_xticklabels(['Stimulus\n-Late Delay','Response\n-Late Delay', 'Early Delay\nLate Delay', 'Late Delay\nLate Delay*',])

panel.set_ylabel('Code overlap')

print('LateevsResponse ', stats.ttest_1samp(df_temp.loc[df_temp.condition == 'Late Delay x Response'].vector.values,0))
print('StimvsLate ', stats.ttest_1samp(df_temp.loc[df_temp.condition == 'Stim x Late Delay'].vector.values,0))
print('EarlyvsLate* ', stats.ttest_rel(df_temp.loc[df_temp.condition == 'Early x Late Delay'].vector.values, df_temp.loc[df_temp.condition == 'Late Delay x Late Delay*'].vector.values))
print('Earlyvslate',stats.ttest_1samp(df_temp.loc[df_temp.condition == 'Early x Late Delay'].vector.values,0))


# -----------------############################## D Panel - Example trial for population #################################-----------------------

color = plt.cm.viridis(np.linspace(0, 1,4))
mpl.rcParams['axes.prop_cycle'] = cycler(color=color)

file_name = 'single_trial_example_237'
big_data = pd.read_csv(path+file_name+'.csv', index_col=0)
file_name = 'single_trial_example_df_237'
dft = pd.read_csv(path+file_name+'.csv', index_col=0)
dft['a_Stimulus_ON'] = dft['fixed_times'] - dft['Stimulus_ON']

delay = 3
align = 'Stimulus_ON'
cue_on=0
cue_off=0.38
start=-2
stop =  max(dft['a_'+align])

new_convolve

# Align to specific epoch, in this case Stimulus 
big_data['time_centered'] = big_data['times'] - big_data['Stimulus_ON'] 
big_data['time_centered'] = np.round(big_data.time_centered/1000, 2) #### estos es importante!!
big_data['firing_'] = big_data['firing']*1000

df_results = pd.DataFrame(dtype=float)
df_results['firing'] = big_data.loc[(big_data.time_centered <= stop)].groupby(['time_centered','neuron'])['firing_'].mean()
df_results['error'] = big_data.loc[(big_data.time_centered <= stop)].groupby(['time_centered','neuron'])['firing_'].std()
df_results.reset_index(inplace=True)

panel = h1
for N in df_results.neuron.unique():
    y_mean= df_results.loc[df_results.neuron == N].firing
    error = df_results.loc[df_results.neuron == N].error
    lower = y_mean - error
    upper = y_mean + error
    x=df_results.loc[df_results.neuron == N].time_centered

    panel.plot(x, y_mean, label=N, alpha=0.5)
#                 sns.lineplot(x='time_centered',y='firing',data=df_results)

panel.set_xlim(start,stop)   
panel.set_xlabel('Time from stimulus onset (s)')

y = np.arange(0,80,0.1)     
panel.fill_betweenx(y, cue_on,cue_off, color='lightgrey', alpha=1, linewidth=0)  
panel.fill_betweenx(y, cue_off+delay,cue_off+delay+.2, color='lightgrey', alpha=1, linewidth=0)
panel.set_ylabel('Firing rate (spks/s)')

panel = h2
# Organized by cluster_id and corrected for FR ____________________________
cluster_id=[]
FR_mean=[]
start_FR=0.2
stop_FR=delay

for N in dft.cluster_id.unique():
    spikes = dft.loc[(dft.cluster_id==N)&(dft['a_'+align] >start_FR)&(dft['a_'+align] <stop_FR)]['a_'+align].values
    FR_mean.append(len(spikes)/abs(stop_FR-start_FR))
    cluster_id.append(N)

df_spikes = pd.DataFrame(list(zip(cluster_id,FR_mean)), columns =['cluster_id','FR'])
df_spikes = df_spikes.sort_values('FR')
df_spikes['new_order'] = np.arange(len(df_spikes))
dft = pd.merge(df_spikes, dft, on=['cluster_id'])

j=0
for N in dft.new_order.unique():
    if dft.loc[dft['new_order'] == N]['cluster_id'].iloc[0] == 285 or dft.loc[dft['new_order'] == N]['cluster_id'].iloc[0] == 302:
        color = 'orangered'
    else:
        color='steelblue'

    spikes = dft.loc[dft.new_order==N]['a_'+align].values
    j+=1
    panel.plot(spikes,np.repeat(j, len(spikes)), '|', markersize=0.5, color='black', zorder=1)

panel.set_ylabel('Single units')
panel.set_ylim(0,j)
panel.set_xlim(start,stop)
panel.axes.get_xaxis().set_visible(False)

y = np.arange(0,j+1,0.1)
panel.fill_betweenx(y, cue_on,cue_off, color='grey', alpha=1, linewidth=0)
panel.fill_betweenx(y, cue_off+delay,cue_off+delay+.2, color='lightgrey', alpha=1, linewidth=0) 

# ------############################## E Panel -- WM error trials for all sessions #################################-----------------------

# y_lower = 0
# y_upper=0

# colors=['crimson','darkgreen']
# variables = ['WM_roll','WM_roll']
# hits = [0,1]
# variables_combined=[variables[0]+'_'+str(hits[0]),variables[1]+'_'+str(hits[1])]

# file_name = 'single_delay_WM_roll0.6_0.5_folded5_overlap0.25'
# # file_name = 'single_delay_WM_roll0.6_0.5_2_folded3'
# df_cum_sti = pd.read_csv(path+file_name+'.csv', index_col=0)


# scores = df_cum_sti.groupby('session').score.mean().reset_index()
# list_exclude = scores.loc[scores.score<0.55].session.unique()
# df_cum_sti = df_cum_sti[~df_cum_sti['session'].isin(list_exclude)] 

# delay=10
# panel=i1
# plotsingledelay(df_cum_sti, panel, colors, variables_combined, delay)

# ------#########################################################################################-----------------------


# ------############################## J Panel -- WM error trials individual example for stimulus #################################-----------------------

# colors=['crimson','darkgreen']
# variables = ['WM_roll','WM_roll']
# hits = [0,1]
# variables_combined=[variables[0]+'_'+str(hits[0]),variables[1]+'_'+str(hits[1])]

# # file_name = 'single_delay_WM_roll0.6_0.5_folded5_sti_all'
# # file_name = 'single_delay_WM_roll0.6_window0.5_folded3_nooverlap_stimulus'
# file_name = 'single_delay_alldelays_WMandHB_stimulus_substracted'
# df_cum_sti = pd.read_csv(path+file_name+'.csv', index_col=0)

# scores = df_cum_sti.groupby('session').score.mean().reset_index()
# list_exclude = scores.loc[scores.score<0.55].session.unique()
# df_cum_sti = df_cum_sti[~df_cum_sti['session'].isin(list_exclude)] 

# panel=j1
# plotsingledelay(df_cum_sti, panel, colors, variables_combined, delay)

# ------#########################################################################################-----------------------
# ------############################## K Panel -- WM error trials individual example for response #################################-----------------------


# save_path = 'C:/Users/Tiffany/Google Drive/WORKING_MEMORY/PAPER/Figures/'
# os.chdir(save_path)
# file_name = 'single_delay_WM_roll0.6_0.5_folded5_overlap0.25_response3'
# df_cum_sti = pd.read_csv(file_name+'.csv', index_col=0)

# scores = df_cum_sti.groupby('session').score.mean().reset_index()
# list_exclude = scores.loc[scores.score<0.55].session.unique()
# df_cum_sti = df_cum_sti[~df_cum_sti['session'].isin(list_exclude)] 

# panel=j2
# plotsingledelay(df_cum_sti, panel, colors, variables_combined, delay)

# ------#########################################################################################-----------------------

# ------############################## J Panel -- WM error trials individual example #################################-----------------------

# delays = [10]
# save_path = 'C:/Users/Tiffany/Google Drive/WORKING_MEMORY/PAPER/Figures/'
# os.chdir(save_path)
# file_name = 'single_delay_WM_roll0.6_0.5_folded5_overlap0.25'
# df_cum_sti = pd.read_csv(file_name+'.csv', index_col=0)
# df_cum_shuffle = pd.read_csv(file_name+'_shuffle.csv', index_col=0)

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

# ------####################################### Example neuron for correct and incorrect trials ##################################-----------------------


# delay = 10
# cluster_id = 153

# colors = [COLORRIGHT,COLORLEFT]
# # colors = ['darkgreen','crimson']
# labels = ['Right stimulus','Left stimulus']
# # labels = ['Correct','Incorrect']

# align = 'Stimulus_ON'

# print('First correct')
# j=1
# temp_df = df.loc[(df.WM_roll >0.6)&(df.hit ==1)]
# j = convolveandplot(temp_df, k, k, variable='reward_side', cluster_id = cluster_id, delay = delay, j=j, alpha=0.3,colors = ['grey','grey'], spikes = False)

# temp_df = df.loc[(df.WM_roll >0.6)&(df.hit ==0)]
# j = convolveandplot(temp_df, k, k, variable='reward_side', cluster_id = cluster_id, delay = delay, j=j, 
#                    labels=['Incorrect right stimulus','Incorrect left stimulus'], spikes = False, kernel=100)

# ------#########################################################################################-----------------------
            



# ------################################       2-point decoder      ##########################-----------------------

# save_path = 'C:/Users/Tiffany/Google Drive/WORKING_MEMORY/PAPER/Figures/'
# os.chdir(save_path)

# file_name = 'single_delay_WM_roll0.6_0.5_folded5_2points'
# df_cum_sti = pd.read_csv(file_name+'.csv', index_col=0)

# df_cum_sti = df_cum_sti.groupby(['session', 'trial_type']).mean()
# df_cum_sti.reset_index(inplace=True)

# # scores = df_cum_sti.groupby('session').score.mean().reset_index()
# # list_exclude = scores.loc[scores.score<0.55].session.unique()
# # df_cum_sti = df_cum_sti[~df_cum_sti['session'].isin(list_exclude)] 

# plot = pd.DataFrame({'Correct early': df_cum_sti.loc[df_cum_sti.trial_type == 'WM_roll_1']['2.5'], 'Incorrect early':  df_cum_sti.loc[df_cum_sti.trial_type == 'WM_roll_0']['2.5'],
#                    'Correct late': df_cum_sti.loc[df_cum_sti.trial_type == 'WM_roll_1']['6.5'], 'Incorrect late':  df_cum_sti.loc[df_cum_sti.trial_type == 'WM_roll_0']['6.5']})

# panel=g2
# sns.violinplot(data=plot, palette=['darkgreen', 'crimson'], width=1,saturation=0.6,linewidth=0, ax=panel)
# sns.violinplot(data=plot, palette=['darkgreen', 'crimson'], width=1,linewidth=1, ax=panel)

# xA = np.random.normal(0, 0.1, len(plot))
# sns.scatterplot(x=xA,y='Correct early',data=plot,alpha=0.9,color='darkgreen',size=2,legend=False, ax=panel)

# xA = np.random.normal(1, 0.1, len(plot))
# sns.scatterplot(x=xA,y='Incorrect early',data=plot,alpha=0.9,color='crimson',size=2,legend=False, ax=panel)

# xA = np.random.normal(2, 0.1, len(plot))
# sns.scatterplot(x=xA,y='Correct late',data=plot,alpha=0.9,color='darkgreen',size=2,legend=False, ax=panel)

# xA = np.random.normal(3, 0.1, len(plot))
# sns.scatterplot(x=xA,y='Incorrect late',data=plot,alpha=0.9,color='crimson',size=2,legend=False, ax=panel)

# panel.hlines(xmin=0, xmax=3.5, y=0.5, linestyle=':')
# panel.set_ylabel('Decoder Accuracy')

# save_path = 'C:/Users/Tiffany/Google Drive/WORKING_MEMORY/PAPER/Figures/'
# os.chdir(save_path)
# file_name = 'single_delay_2point_new'

# df_cum_sti = pd.read_csv(file_name+'.csv', index_col=0)

# df_cum_sti = df_cum_sti.groupby(['session', 'trial_type']).mean()
# df_cum_sti.reset_index(inplace=True)

# # scores = df_cum_sti.groupby('session').score.mean().reset_index()
# # list_exclude = scores.loc[scores.score<0.55].session.unique()
# # df_cum_sti = df_cum_sti[~df_cum_sti['session'].isin(list_exclude)] 

# plot = pd.DataFrame({'Correct WM early': df_cum_sti.loc[df_cum_sti.trial_type == 'WM_roll_1']['2.25'], 'Incorrect WM early':  df_cum_sti.loc[df_cum_sti.trial_type == 'WM_roll_0']['2.25'],
#                    'Correct WM late': df_cum_sti.loc[df_cum_sti.trial_type == 'WM_roll_1']['6.25'], 'Incorrect WM late':  df_cum_sti.loc[df_cum_sti.trial_type == 'WM_roll_0']['6.25'],
#                    'Correct HB early': df_cum_sti.loc[df_cum_sti.trial_type == 'RL_roll_1']['2.25'], 'Incorrect HB early':  df_cum_sti.loc[df_cum_sti.trial_type == 'RL_roll_0']['2.25'],
#                    'Correct HB late': df_cum_sti.loc[df_cum_sti.trial_type == 'RL_roll_1']['6.25'], 'Incorrect HB late':  df_cum_sti.loc[df_cum_sti.trial_type == 'RL_roll_0']['6.25']})

# panel=g2
# sns.violinplot(data=plot, palette=['darkgreen', 'crimson', 'indigo', 'purple' ], width=1,saturation=0.6,linewidth=0, ax=panel)
# sns.violinplot(data=plot, palette=['darkgreen', 'crimson', 'indigo', 'purple' ], width=1,linewidth=1, ax=panel)

# xA = np.random.normal(0, 0.1, len(plot))
# sns.scatterplot(x=xA,y='Correct WM early',data=plot,alpha=0.9,color='darkgreen',size=2,legend=False, ax=panel)

# xA = np.random.normal(1, 0.1, len(plot))
# sns.scatterplot(x=xA,y='Incorrect WM early',data=plot,alpha=0.9,color='crimson',size=2,legend=False, ax=panel)

# xA = np.random.normal(2, 0.1, len(plot))
# sns.scatterplot(x=xA,y='Correct WM late',data=plot,alpha=0.9,color='darkgreen',size=2,legend=False, ax=panel)

# xA = np.random.normal(3, 0.1, len(plot))
# sns.scatterplot(x=xA,y='Incorrect WM late',data=plot,alpha=0.9,color='crimson',size=2,legend=False, ax=panel)


# panel.hlines(xmin=0, xmax=3.5, y=0.5, linestyle=':')
# panel.set_ylabel('Decoder Accuracy')

# ------#########################################################################################-----------------------

# Show the figure
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.95,
                    top=0.95,
                    wspace=1.0,
                    hspace=0.55)

plt.locator_params(nbins=5) 
sns.despine()
# plt.savefig(save_path+'/Fig 4_panel_temp_V6.svg', bbox_inches='tight',dpi=300)
# plt.savefig(save_path+'/Fig 4_panel.png', bbox_inches='tight',dpi=300)

plt.show()