# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 20:36:06 2023

@author: Tiffany
"""


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

# from plots import functions_Fig_4 as plots
# import functions_Fig_4 as plots

save_path = 'C:/Users/Tiffany/Google Drive/WORKING_MEMORY/PAPER/WM_manuscript_FIGURES/'
os.chdir(save_path)
import functions as plots

save_path = 'C:/Users/Tiffany/Google Drive/WORKING_MEMORY/PAPER/WM_manuscript_FIGURES/Fig. 5. Errors in WM/'
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
fig = plt.figure(figsize=(16*cm, 24*cm))
gs = gridspec.GridSpec(nrows=6, ncols=8, figure=fig)

a1 = fig.add_subplot(gs[0, 4:8])
a2 = fig.add_subplot(gs[1, 4:8])
a3 = fig.add_subplot(gs[2, 4:8])

b1 = fig.add_subplot(gs[0, 0:4])
b2 = fig.add_subplot(gs[1, 0:4])
b3 = fig.add_subplot(gs[2, 0:4])

c1 = fig.add_subplot(gs[3, 4:8])
c2 = fig.add_subplot(gs[4, 4:8])
c3 = fig.add_subplot(gs[5, 4:8])

d1 = fig.add_subplot(gs[3, 0:4])
d2 = fig.add_subplot(gs[4, 0:4])
d3 = fig.add_subplot(gs[5, 0:4])

def single_trial_with_decoder(df, df_decoder, big_data, filename, T, panels = [], threshold = 0.4, 
                              align = 'Stimulus_ON', show_y = True ):

    delay = df.loc[df.trial==T].delay.unique()[0]
    cue_on=0
    cue_off=0.38
    start=-2.5
    stop=4+delay
    # stop= max(df.loc[df.trial==T]['a_'+align])
    endrange = max(df.loc[df.trial==T]['a_'+align])
    
    ### ------ Filter neurons with substantial weight in the decoder
    window='Delay_OFF--0.5-0'
    path = 'C:/Users/Tiffany/Google Drive/WORKING_MEMORY/PAPER/ANALYSIS_Figures/'
    file_name = 'weights for the modelling_complete'
    df_weights = pd.read_csv(path+file_name+'.csv', index_col=0)
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
    filter_neuron = filter_neuron.loc[filter_neuron.firing > 0.5]
    neurons = filter_neuron.neuron.unique()
    df_results = df_results.loc[df_results.neuron.isin(neurons)]
    
    significant_left_neurons = df_weights.loc[(df_weights["Delay_OFF--0.5-0"] < -threshold)].neuron.unique()
    significant_right_neurons = df_weights.loc[(df_weights["Delay_OFF--0.5-0"] > threshold)].neuron.unique()
    
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
        print(cluster_id)
        # if df_weights.loc[df_weights.neuron == cluster_id]["Delay_OFF--0.5-0"].iloc[0] >0:
        if df_weights.loc[df_weights.neuron == cluster_id]["Delay_OFF--0.5-0"].iloc[0] >threshold:
            color_selectivity=COLORRIGHT
            j=right
            right+=1
        elif df_weights.loc[df_weights.neuron == cluster_id]["Delay_OFF--0.5-0"].iloc[0] <-threshold:
        # elif df_weights.loc[df_weights.neuron == cluster_id]["Delay_OFF--0.5-0"].iloc[0] <0:
            color_selectivity=COLORLEFT
            j=left
            left-=1
        spikes = dft.loc[dft.new_order==N]['a_'+align].values
        panel.plot(spikes,np.repeat(j, len(spikes)), '|', markersize=3.5, color=color_selectivity, zorder=3)
    
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
        

    
        
# -----------------############################## Example trial correct  #######################-----------------------

path = 'C:/Users/Tiffany/Google Drive/WORKING_MEMORY/PAPER/ANALYSIS_figures/'

T=83
filename = 'E17_2022-02-02_17-13-06.csv'

file_name = 'decoder_'+str(T)+'_'+filename
df_decoder =pd.read_csv(path+file_name+'.csv')

file_name = 'df_'+str(T)+'_'+filename
df = pd.read_csv(path+file_name)

file_name = 'convolve_'+str(T)+'_'+filename
big_data = pd.read_csv(path+file_name)

dft = plots.single_trial_with_decoder(df, df_decoder, big_data, filename, T, panels = [a1,a2,a3])

# ------#########################################################################################-----------------------

T=185

file_name = 'decoder_'+str(T)+'_'+filename
df_decoder =pd.read_csv(path+file_name+'.csv')

file_name = 'df_'+str(T)+'_'+filename
df = pd.read_csv(path+file_name)

file_name = 'convolve_'+str(T)+'_'+filename
big_data = pd.read_csv(path+file_name)

plots.single_trial_with_decoder(df, df_decoder, big_data, filename, T, panels = [b1,b2,b3], show_y = True)
# a1.set_ylim(0,11.5)
b2.set_ylim(-0,15)
a2.set_ylim(-0,15)
b3.set_ylim(-10,10)
a3.set_ylim(-10,10)

# ------#########################################################################################-----------------------


# T=223
# filename = 'E17_2022-01-31_16-30-44.csv'

# file_name = 'decoder_'+str(T)+'_'+filename
# df_decoder = pd.read_csv(path+file_name+'.csv', index_col=0)

# file_name = 'df_'+str(T)+'_'+filename
# df = pd.read_csv(path+file_name, index_col=0)

# file_name = 'convolve_'+str(T)+'_'+filename
# big_data = pd.read_csv(path+file_name, index_col=0)

# plots.single_trial_with_decoder(df, df_decoder, big_data, filename, T, panels = [c1,c2,c3])
# c1.set_title('Correct WM trial (Right stimulus)', fontsize=8)
# # c1.set_subtitle('Mouse E17 2022-01-31', fontsize=8)

# # ------#########################################################################################-----------------------


# T=21
# filename = 'E17_2022-01-31_16-30-44.csv'

# file_name = 'decoder_'+str(T)+'_'+filename
# df_decoder =pd.read_csv(path+file_name, index_col=0)

# file_name = 'df_'+str(T)+'_'+filename
# df = pd.read_csv(path+file_name, index_col=0)

# file_name = 'convolve_'+str(T)+'_'+filename
# big_data = pd.read_csv(path+file_name, index_col=0)

# plots.single_trial_with_decoder(df, df_decoder, big_data, filename, T, panels = [d1,d2,d3])
# d3.set_ylim(-10,10)
# c3.set_ylim(-10,10)
# d1.set_title('Correct RepL trial (Right stimulus)', fontsize=8)

plt.subplots_adjust(left=0.07,
                    bottom=0.07,
                    right=0.97,
                    top=0.97,
                    wspace=2,
                    hspace=0.5)
sns.despine()
plt.show()