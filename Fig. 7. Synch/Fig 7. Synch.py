# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 14:34:53 2023

@author: Tiffany
"""

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

from neo.core import SpikeTrain
from quantities import ms, s, Hz
from elephant.statistics import mean_firing_rate
from elephant.statistics import time_histogram, instantaneous_rate
from elephant.kernels import GaussianKernel
from elephant.statistics import mean_firing_rate

import os 
import numpy as np
import pandas as pd
import scipy.io
import seaborn as sns

from cycler import cycler

utilities = 'G:/Mi unidad/WORKING_MEMORY/PAPER/WM_manuscript_FIGURES/'
os.chdir(utilities)
import functions as plots

save_path = 'G:/Mi unidad/WORKING_MEMORY/PAPER/WM_manuscript_FIGURES/Fig. 7. Synch/'
path = 'G:/Mi unidad/WORKING_MEMORY/PAPER/ANALYSIS_figures/'
cm = 1/2.54
sns.set_context('paper', rc={'axes.labelsize': 7,
                            'lines.linewidth': 1, 
                            'lines.markersize': 3, 
                            'legend.fontsize': 6,  
                            'xtick.major.size': 1,
                            'xtick.labelsize': 6, 
                            'ytick.major.size': 1, 
                            'ytick.labelsize': 6,
                            'xtick.major.pad': 0,
                            'ytick.major.pad': 0,
                            'xlabel.labelpad': -10})

# Create a figure with 6 subplots using a GridSpec
fig = plt.figure(figsize=(21*cm, 15*cm))
gs = gridspec.GridSpec(nrows=6, ncols=9, figure=fig)

# e = fig.add_subplot(gs[0, 4:5])
c1 = fig.add_subplot(gs[0:2, 3:5])

# Create the subplots
a2 = fig.add_subplot(gs[0, 0:3])
a3 = fig.add_subplot(gs[2, 0:3])
a1 = fig.add_subplot(gs[1, 0:3])

f1 = fig.add_subplot(gs[3, 1])
f2 = fig.add_subplot(gs[4, 1])
g1 = fig.add_subplot(gs[3, 0])
g2 = fig.add_subplot(gs[4, 0])
h1 = fig.add_subplot(gs[3, 2])
h2 = fig.add_subplot(gs[4, 2])
i1 = fig.add_subplot(gs[3:5, 3:5]) # 
d = fig.add_subplot(gs[5, 0:3])

j1 = fig.add_subplot(gs[0:2, 5:7])
j2 = fig.add_subplot(gs[0:2, 7:9])
k1 = fig.add_subplot(gs[3:5, 5:7])
k2 = fig.add_subplot(gs[3:5, 7:9])

fig.text(0.01, 1, 'a', fontsize=10, fontweight='bold', va='top')
fig.text(0.01, 0.53, 'b', fontsize=10, fontweight='bold', va='top')
fig.text(0.36, 1, 'c', fontsize=10, fontweight='bold', va='top')
fig.text(0.36, 0.53, 'd', fontsize=10, fontweight='bold', va='top')
fig.text(0.56, 1, 'e', fontsize=10, fontweight='bold', va='top')
fig.text(0.56, 0.53, 'f', fontsize=10, fontweight='bold', va='top')

##################################### Functions #####################

def trials(row):
    val = 0
    val = row['T']/row['total trials']
    return val

def synch_trial(df, T, lower_plot, upper_plot = None, trial=0, start=-2, stop=0, color='indigo', surrogates=100, bins=20):
    dft = df.loc[df.trial ==T]
    align='Stimulus_ON'
    # stop = dft.delay.unique()[0]+5 # end of analyzed window
    delay = dft.delay.unique()[0]
    #Filter for the last 2 seconds of the ITI
    dft = dft.loc[(dft['a_'+align]>start)&(dft['a_'+align]<stop)]
    
    # Recover amount of neurons that were being registered at that trial interval
    n_neurons = len(df.cluster_id.unique())
    times_spikes = dft['a_'+align].values
    times_spikes = times_spikes*1000*ms #transform to ms
    
    ############################################################ Set the strat and end time of the train
    stop_time =  stop*1000*ms ## End of the trial in ms
    start_time = start*1000*ms ## Start of the trial in ms     
    
    ############################################################ Spiketrain
    spiketrain = SpikeTrain(times_spikes, units=ms, t_stop=stop_time, t_start=start_time) 
    
    ############################################################ 
    histogram_rate = time_histogram([spiketrain], bins*ms, output='rate')
    times_ = histogram_rate.times.rescale(s)
    firing_real = histogram_rate.rescale(histogram_rate.dimensionality).magnitude.flatten()
    
    real_std=np.std(firing_real) # Store the real std value
    
    #         t1_start = process_time() 
    list_std = []
    for i in range(surrogates):
        # Create a random shuffle for the same amount of spikes in that interval
        random_float_list = np.random.uniform(start, stop, len(times_spikes))
        surrogate_spikes = np.array(random_float_list)*1000*ms #transform to ms
        spiketrain = SpikeTrain(surrogate_spikes, units=ms, t_stop=stop_time, t_start=start_time) 
    
        histogram_rate = time_histogram([spiketrain], bins*ms, output='rate')
        times_ = histogram_rate.times.rescale(s)
        firing = histogram_rate.rescale(histogram_rate.dimensionality).magnitude.flatten()
    
        list_std.append(np.std(firing))
    
    # Organized by cluster_id and corrected for FR ____________________________
    cluster_id=[]
    FR_mean=[]
    
    for N in df.cluster_id.unique():
        spikes = dft.loc[dft.cluster_id==N]['a_'+align].values
        FR_mean.append(len(spikes)/abs(stop-start))
        cluster_id.append(N)
    
    df_spikes = pd.DataFrame(list(zip(cluster_id,FR_mean)), columns =['cluster_id','FR'])
    df_spikes = df_spikes.sort_values('FR')
    df_spikes['new_order'] = np.arange(len(df_spikes))
    
    dft = pd.merge(df_spikes, dft, on=['cluster_id'])
    
    print('Synch:', real_std/np.mean(list_std), '; WM:', str(dft.WM_roll.unique()[0]))
    
    if upper_plot != None:
        panel = upper_plot
        panel.set_title(trial)
        j=0
        for N in dft.new_order.unique():
            spikes = dft.loc[dft.new_order==N]['a_'+align].values
            j+=1
            panel.plot(spikes,np.repeat(j, len(spikes)), '|', markersize=1, color='black', zorder=1)
    
    panel = lower_plot
    panel.plot(times_,firing_real/n_neurons*1000, color=color, linewidth=0.5)
    # y = np.arange(0,j+1,0.1)
    # panel.fill_betweenx(y, cue_on,cue_off, color='grey', alpha=.4)
    # panel.fill_betweenx(y, cue_off+delay,cue_off+delay+.2, color='beige', alpha=.8)
    panel.set_ylim(0,20)
    panel.set_ylabel('Firing rate\n(spks/s)')
    
    
def distribution(df_final, variable='WM_roll'):
    r_value_lower=[]
    r_value_upper =[]
    for animal in df_final.animal.unique():
        test_df = df_final.loc[df_final.animal==animal]
        test_df = test_df.dropna()
        corr_synch = test_df['synch'].values

        r_value_list=[]
        for animal in df_final.animal.unique():
            try:
                corr_WM = df_final.loc[df_final.animal==animal][variable].values[-len(corr_synch):]
                slope, intercept, r_value, p_value, std_err = stats.linregress(corr_synch,corr_WM)
                # print('Slope WM:'+str(round(slope,3))+' Intercept:'+ str(round(intercept,3))+ ' R_value:'+ str(round(r_value,3)) + ' P_value:'+ str(round(p_value,3)) + ' Std_err:'+ str(std_err))
                r_value_list.append(r_value)
            except:

                continue
        # r_value_upper.append(np.quantile(r_value_list, 0.9))
        # r_value_lower.append(np.quantile(r_value_list, 0.1))
        r_value_upper.append(np.mean(r_value_list))
    return r_value_upper

    #######################################################################################
    
# -----------------############## C Panel - Synch correlation with p(WM) ################-----------------------

# file_name = 'synch_data'
# df_final = pd.read_csv(file_name+'.csv')

# panel = b
# sns.boxplot(x='subject',  y='R_WM', data=df_final, ax=panel)
# panel.scatter(np.random.uniform(8.75,9.25,len(df_final)),df_final['R_WM'], s=1, alpha=0.5, color='black')
# panel.errorbar(9, np.mean(df_final['R_WM']),yerr=stats.sem(df_final['R_WM']),marker='o',markersize= 4, color='black',zorder=1)  
# panel.hlines(xmin=0, xmax=9, y=0, linestyle=':')
# panel.set_xlabel('')

# ----------------------------------------------------------------------------------------------------------------


# -----------------############## A Panel - Example trial  #############-----------------------
# path = 'C:/Users/Tiffany/Google Drive/WORKING_MEMORY/PAPER/Figures/'
os.chdir(save_path)
color = plt.cm.viridis(np.linspace(0, 1,4))
mpl.rcParams['axes.prop_cycle'] = cycler(color=color)

file_name = 'single_trial_example_432'
big_data = pd.read_csv(save_path+file_name+'.csv', index_col=0)

file_name = 'single_trial_example_df_432'
dft = pd.read_csv(save_path+file_name+'.csv', index_col=0)
dft['a_Stimulus_ON'] = dft['fixed_times'] - dft['Stimulus_ON']

delay = 3
align = 'Stimulus_ON'
cue_on=0
cue_off=0.4
start=-2
stop =  max(dft['a_'+align])

# Align to specific epoch, in this case Stimulus 
big_data['time_centered'] = big_data['times'] - big_data['Stimulus_ON'] 
big_data['time_centered'] = np.round(big_data.time_centered/1000, 2) #### estos es importante!!
big_data['firing_'] = big_data['firing']*1000

df_results = pd.DataFrame(dtype=float)
df_results['firing'] = big_data.loc[(big_data.time_centered <= stop)].groupby(['time_centered','neuron'])['firing_'].mean()
df_results['error'] = big_data.loc[(big_data.time_centered <= stop)].groupby(['time_centered','neuron'])['firing_'].std()
df_results.reset_index(inplace=True)

panel = a1
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

y = np.arange(0,120,0.1)     
panel.fill_betweenx(y, cue_on,cue_off, color='lightgrey', alpha=.8)  
panel.fill_betweenx(y, cue_off+delay,cue_off+delay+.2, color='grey', alpha=.8)
panel.set_ylabel('Firing rate\n(spks/s)')

panel = a2
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
panel.set_title('Session')

y = np.arange(0,j+1,0.1)
panel.fill_betweenx(y, cue_on,cue_off, color='lightgrey', alpha=1)
panel.fill_betweenx(y, cue_off+delay,cue_off+delay+.2, color='grey', alpha=1)  

panel=a3

# Plot the instantaneous firing rate
T=432
synch_trial(dft, T, a3, trial=T, start=-2, stop=10, bins=30, color='black')
panel.set_ylim(-5,25)

# ----------------------------------------------------------------------------------------------------------------


# -----------------############## A2 Panel - Example trial  #############-----------------------
# file_name = 'synch_data_trials_2beforeSti'
# df_final = pd.read_csv(path+file_name+'.csv', index_col=0)

# # df_final = df_final.loc[df_final.T_norm < 0.6 ]
# df_final['state'] = np.where(df_final['WM_roll']>0.5, 1, 0)

# df_final['T_norm'] = np.around(df_final['T_norm'],2)

# df_results = pd.DataFrame()
# df_results['synch_window'] = df_final.groupby(['T_norm','animal'])['synch_window'].mean()
# df_results.reset_index(inplace=True)

# panel = e
# sns.lineplot(x='T_norm',y='synch_window',data=df_results,ax=panel, color='orange', ci=95)

# panel.hlines(xmin=0,xmax=1,y=1, linestyle=':')
# panel.set_ylim(0.9,2)


# -----------------############### C Panel  -  Synchrony depending on state  ######################-----------------------
file_name = 'synch_data_trials_2beforeSti'
df_final = pd.read_csv(save_path+file_name+'.csv', index_col=0)

df_final['state'] = np.where(df_final['WM_roll']>0.5, 0, 1)

panel = c1

# sns.histplot(x='synch',data=df_final, hue='state',bins=np.arange(0.5,2.5,0.1), palette=['indigo','green',], stat='probability',ax=panel,
#              common_norm=False, shrink=.8, kde=True)

# data_group1 = df_final.loc[df_final.state==1]['synch'].values
# data_group2 = df_final.loc[df_final.state==0]['synch'].values
# print(stats.ks_2samp(df_final.loc[df_final.state==1]['synch'].values,df_final.loc[df_final.state==0]['synch_window'].values))
# stats.ttest_ind(data_group1, data_group2, equal_var = False)

# panel.set_xlim(0.5,2.5)
# panel.axvline(df_final.loc[df_final.state==0].synch.mean(), color='indigo', linestyle='dashed', linewidth=1)
# panel.axvline(df_final.loc[df_final.state==1].synch.mean(), color='darkgreen', linestyle='dashed', linewidth=1)

# panel.set_xlabel('Synch')

df_results = pd.DataFrame()
df_results['synch'] = df_final.groupby(['animal','state']).synch.mean()
df_results.reset_index(inplace=True)
df_results['state'] = pd.Categorical(df_results['state'], categories=[0,1], ordered=True)
df_results['state'] = np.where(df_results.state == 0, 'WM', 'RepL')

palette = sns.color_palette(['black'], len(df_results.animal.unique()))
sns.lineplot(x="state", y="synch",data=df_results, hue='animal', alpha=0.8, palette=palette,ax=panel, linewidth=0.2, markeredgewidth = 0.2,marker='',legend=False,markersize=3)
sns.boxplot(x='state',y="synch", data=df_results, width=0.5, showfliers=False, palette=['darkgreen', 'indigo'], ax=panel, linewidth=1)

panel.set_xticks([0,1],['WM','RepL'])
panel.legend(loc='lower right', ncol=2)
add_stat_annotation(panel, data=df_results, x='state', y='synch',
                    box_pairs=[( 'WM','RepL')],
                    test='t-test_paired', text_format='star', loc='inside', line_offset_to_box=0.05, text_offset=-0.5, line_offset=0, verbose=1, fontsize=6, linewidth=0.5)

# ----------------------------------------------------------------------------------------------------------------

# -----------------############### E Panel  -  Synch versus trial ######################-----------------------

# panel = e
# df_final['T_norm'] = np.around(df_final['T_norm'],2)
# df_final = df_final.loc[df_final.T_norm > 0.02]

# df_results = pd.DataFrame()
# df_results['synch_window'] = df_final.groupby(['T_norm','animal'])['synch_window'].mean()
# df_results.reset_index(inplace=True)

# sns.lineplot(x='T_norm',y='synch_window',data=df_results, ax=panel, color='orange')

# panel.hlines(xmin=0,xmax=1,y=1, linestyle=':')
# panel.set_ylim(0.9,2.2)

# panel.set_xlabel('Normalized trial index')
# panel.set_ylabel('Synch')

# ----------------------------------------------------------------------------------------------------------------

# -----------------############### D Panel #################-----------------------

animal = "E22_2022-01-13_16-34-24.csv"
# file_name = 'synch_data_trials_2beforeSti'
# df_final = pd.read_csv(file_name+'.csv', index_col=0)
threshold = 0.5

df_session = df_final.loc[df_final.animal==animal]

panel = d
panel.fill_between(df_session['trial'],0.9 , 2.5, where=df_session['WM_roll'] <= threshold,
                 facecolor='indigo', alpha=0.3)
panel.fill_between(df_session['trial'], 0.9, 2.5,  where=df_session['WM_roll'] >= threshold,
                 facecolor='darkgreen', alpha=0.3)
sns.lineplot(x="trial", y="synch_window",data=df_session, color='black',ci=68,ax=panel)      
panel.set_ylabel('Synch')
panel.set_ylim(0.9,max(df_session.synch_window)+0.3)
panel.set_xlabel('Trials')
panel.set_title('Mouse E22 13-01')
# Select the trial that we want to look at this time

T=340
filename = 'single_trial_synch_'+str(T)
df = pd.read_csv(filename+'.csv', sep=',',index_col=0)
synch_trial(df, T, h1, h2, trial=T)
h1.axis('off')
h2.axis('off')

T=153
filename = 'single_trial_synch_'+str(T)
df = pd.read_csv(filename+'.csv', sep=',',index_col=0)
synch_trial(df, T, g1, g2, trial=T)
f1.axis('off')
f2.axis('off')

# T=225
T=212
filename = 'single_trial_synch_'+str(T)
df = pd.read_csv(filename+'.csv', sep=',',index_col=0)
synch_trial(df, T, f1, f2, trial=T)
g1.axis('off')
g2.axis('off')


# ------##############################################################################-----------------------

# ------############################## E Panel #################################-----------------------

file_name = 'synch_corrdata_final'
df_corr = pd.read_csv(save_path+file_name+'.csv', index_col=0)

panel = i1

df_corr = df_corr
# sns.violinplot(data=df_corr,  palette=['grey', 'grey','grey'], ax = panel, order = ['r_WM_shuff','r_acc_shuff','r_repeat_shuff'], saturation=0.6,linewidth=0)
# sns.violinplot(data=df_corr,  palette=['black', 'black','black'], ax = panel, order=['r_WM_shuff','r_acc_shuff','r_repeat_shuff'], edgecolor='black', showmeans=True)

sns.boxplot(data=df_corr,  palette=['grey', 'grey','grey'], ax = panel, order = ['r_WM_shuff','r_acc_shuff','r_repeat_shuff'], saturation=0.6,linewidth=1, width=0.5)
# sns.boxplot(data=df_corr,  palette=['black', 'black','black'], ax = panel, order = ['r_WM_shuff','r_acc_shuff','r_repeat_shuff'], saturation=0.6,linewidth=0)


xA = np.random.normal(0, 0.2, len(df_corr))
# sns.scatterplot(x=xA,y='r_WM',data=df_corr,alpha=0.9,color='darkgrey',size=2,legend=False)
sns.scatterplot(x=xA,y='r_WM_shuff',data=df_corr,alpha=0.9, ax = panel, color='black',legend=False)
print('r_WM_shuff = ', df_corr.r_WM_shuff.mean(), 'Std = ', df_corr.r_WM_shuff.std())
# print(stats.ttest_1samp(df_corr['r_WM_shuff'],0))

xA = np.random.normal(1, 0.2, len(df_corr))
# sns.scatterplot(x=xA,y='r_acc',data=df_corr,alpha=0.9,color='darkgrey',size=2,legend=False)
sns.scatterplot(x=xA,y='r_acc_shuff',data=df_corr,alpha=0.9, ax = panel, color='black',legend=False)
print('r_acc_shuff = ', df_corr.r_acc_shuff.mean(),'Std = ' , df_corr.r_acc_shuff.std())
# print(stats.ttest_1samp(df_corr['r_acc_shuff'],0))

xA = np.random.normal(2, 0.2, len(df_corr))
# sns.scatterplot(x=xA,y='r_repeat',data=df_corr,alpha=0.9,color='darkgrey',size=2,legend=False)
sns.scatterplot(x=xA,y='r_repeat_shuff',data=df_corr,alpha=0.9, ax = panel, color='black',legend=False)

print('r_repeat_shuff = ', df_corr.r_repeat_shuff.mean() ,'Std = ', df_corr.r_repeat_shuff.std())
# print(stats.ttest_1samp(df_corr['r_repeat_shuff'],0))

panel.hlines(y=0, xmin=-0.5, xmax=2.5, linestyle=":")

for const, regressor in zip(range(3),['r_acc_shuff','r_repeat_shuff','r_WM_shuff']):
    if stats.ttest_1samp(df_corr[regressor],0)[1] <=0.001:
        panel.text(const-0.12,  0.3, '***') 
        

    elif stats.ttest_1samp(df_corr[regressor],0)[1] <=0.01:
        panel.text(const-0.08,  0.3, '**')

    elif stats.ttest_1samp(df_corr[regressor],0)[1] <=0.05:
        panel.text(const-0.04, 0.3, '*')

    else:
        panel.text(const-0.1,  0.3, 'ns')
    print(stats.ttest_1samp(df_corr[regressor],0))
# panel.xticks([0,1,2], ['Accuracy','Repeat','pWM'])
panel.set_xlabel('Corr. coef. (Synch. X)')
panel.set_xticklabels(['p(WM)','Accuracy','RB'])

# ------##############################################################################-----------------------

# ------##########################Autocrrelations states ##########################-----------------------
file_name = 'indiv_sess_auto_corrs_REPLapses'
df = pd.read_csv(file_name+'.csv', header=None, index_col=0)
panel = j1

df['mean'] = df.mean(axis=1)
panel.plot(df.index,df['mean'], color='indigo')

file_name = 'indiv_sess_auto_corrs_WM_phases'
df = pd.read_csv(file_name+'.csv', header=None, index_col=0)

df['mean'] = df.mean(axis=1)

panel.plot(df.index,df['mean'], color='darkgreen')

panel.hlines(xmin=-1, xmax=1, y=0, linestyle=':')
panel.set_title('MUA Autocorr.')
panel.set_xlabel('Time lag (s)')

# ------##############################################################################-----------------------
# ------##########################Autocrrelations difference ##########################-----------------------
panel = j2

file_name = 'indiv_sess_auto_corrs_REPLapses'
df_rep = pd.read_csv(file_name+'.csv', header=None, index_col=0)
df_rep['mean'] = df_rep.mean(axis=1)

file_name = 'indiv_sess_auto_corrs_WM_phases'
df_wm = pd.read_csv(file_name+'.csv', header=None, index_col=0)
df_wm['mean'] = df_wm.mean(axis=1)
x= df_wm.index
y= df_rep['mean']-df_wm['mean']

panel.plot(x,y, color='darkgreen')

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

y = smooth(y,5)
panel.plot(x,y, color='black')

panel.hlines(xmin=-1, xmax=1, y=0, linestyle=':')
panel.locator_params(nbins=3) 
panel.set_xlabel('Time lag (s)')
panel.set_title('Difference RepL - WM')

# ------##############################################################################-----------------------
# ------########################## PSD both  ##########################-----------------------
panel = k1
file_name = 'indiv_sess_PSD_REPLapses'
df = pd.read_csv(file_name+'.csv', header=None, index_col=0)
x=df.index
y= df.mean(axis=1)
panel.plot(x, y, color='indigo')

file_name = 'indiv_sess_PSD_WM_phases'
df = pd.read_csv(file_name+'.csv', header=None, index_col=0)
x=df.index
y= df.mean(axis=1)
panel.plot(x, y, color='darkgreen')

panel.set_yscale('log')
panel.set_xscale('log')
panel.set_xlim(2,99)
panel.tick_params(axis="both", which='both',direction="in")
panel.set_ylim(0.01,0.4)
panel.set_title('MUA Power Spectrum')
panel.set_xlabel('Frequency (Hz)')

# ------##############################################################################-----------------------
# ------########################## PSD difference zoomed in ##########################-----------------------
panel = k2
file_name = 'indiv_sess_PSD_REPLapses'
df = pd.read_csv(file_name+'.csv', header=None, index_col=0)
x=df.index
y_R= df.mean(axis=1)

file_name = 'indiv_sess_PSD_WM_phases'
df = pd.read_csv(file_name+'.csv', header=None, index_col=0)
x=df.index
y_W= df.mean(axis=1)

y = y_R - y_W 

panel.plot(x, y, color='black')

panel.set_yscale('linear')
panel.set_xscale('log')
panel.set_xlim(1.99,100)
panel.hlines(xmin=2, xmax=100, y=0, linestyle=':')
panel.tick_params(axis="both", which='both',direction="in")
panel.set_title('Difference RepL - WM')
panel.set_xlabel('Frequency (Hz)')

# ------##############################################################################-----------------------
# Show the figure
sns.despine()
plt.subplots_adjust(left=0.07,
                    bottom=0.07,
                    right=0.97,
                    top=0.97,
                    wspace=0.75,
                    hspace=0.75)

# plt.savefig(save_path+'/Fig 7_Brain state_V2.svg', bbox_inches='tight',dpi=1000)
# plt.savefig(save_path+'/Fig 7_Brain state.pdf', bbox_inches='tight',dpi=1000)

plt.show()