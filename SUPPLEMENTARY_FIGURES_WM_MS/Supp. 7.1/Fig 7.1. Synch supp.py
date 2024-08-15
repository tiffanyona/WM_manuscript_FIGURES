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

save_path = 'G:/Mi unidad/WORKING_MEMORY/PAPER/WM_manuscript_FIGURES/SUPPLEMENTARY_FIGURES_WM_MS/Supp. 7.1/'
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
fig = plt.figure(figsize=(17*cm, 4*cm))
gs = gridspec.GridSpec(nrows=1, ncols=3, figure=fig)

a = fig.add_subplot(gs[0, 0:2])
b = fig.add_subplot(gs[0, 2:3])


fig.text(0.01, 1, 'a', fontsize=10, fontweight='bold', va='top')
fig.text(0.65, 1, 'b', fontsize=10, fontweight='bold', va='top')

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
    panel.set_ylim(0,15)
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
    

# ----------------------------------------------------------------------------------------------------------------


# -----------------############## A2 Panel - Example trial  #############-----------------------
file_name = 'synch_data_trials_2beforeSti'
df_final = pd.read_csv(save_path+file_name+'.csv', index_col=0)

# df_final = df_final.loc[df_final.T_norm < 0.6 ]
df_final['state'] = np.where(df_final['WM_roll']>0.6, 1, 0)

df_final['T_norm'] = np.around(df_final['T_norm'],2)

df_results = pd.DataFrame()
df_results['synch_window'] = df_final.groupby(['T_norm','animal'])['synch_window'].mean()
df_results.reset_index(inplace=True)

panel = a
sns.lineplot(x='T_norm',y='synch_window',data=df_results,ax=panel, color='black', ci=95)

panel.hlines(xmin=0,xmax=1,y=1, linestyle=':')
panel.set_ylim(0.9,2.5)
panel.set_ylabel('Synch')
panel.set_xlabel('Normalized trial index')


# -----------------############### C Panel  -  Synchrony depending on state  ######################-----------------------
file_name = 'avg_session_MEAN_rate_per_level'
df_final = pd.read_csv(save_path+file_name+'.csv', header=None)
df_final.columns = ['RL', 'WM']

panel = b

melted_data = pd.melt(df_final)
sns.boxplot(data=melted_data, x='variable', y='value', ax=b, order=['WM', 'RL'], palette=['darkgreen', 'indigo'])

panel.plot([1, 0], df_final.T.values, color='black', alpha=0.3, marker='')
panel.set_ylabel('Mean rate (spks/s)')
panel.set_title('Mean pre-stim rate')
panel.set_xticks([0,1],['WM','RepL'])
panel.legend(loc='lower right', ncol=2)

add_stat_annotation(panel, data=melted_data, x='variable', y='value',
                    box_pairs=[( 'WM','RL')],
                    test='t-test_paired', text_format='star', loc='inside', line_offset_to_box=0.05, text_offset=-0.5, line_offset=0, verbose=1, fontsize=6, linewidth=0.5)

# ----------------------------------------------------------------------------------------------------------------



# ------##############################################################################-----------------------
# Show the figure
sns.despine(offset = 10)
plt.subplots_adjust(left=0.07,
                    bottom=0.07,
                    right=0.97,
                    top=0.97,
                    wspace=0.5,
                    hspace=0.5)

# plt.savefig(save_path+'/Fig 7_Brain state_V3.svg', bbox_inches='tight',dpi=1000)
# plt.savefig(save_path+'/Fig 7_Brain state_V2.png', bbox_inches='tight',dpi=1000)

plt.show()