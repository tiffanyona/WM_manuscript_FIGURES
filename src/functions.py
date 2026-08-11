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
from pathlib import Path
_ANALYSIS_DATA = Path(__file__).resolve().parents[2] / 'general_data'
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from scipy import special
import json
from sklearn.linear_model import LogisticRegression
from scipy.optimize import curve_fit
from neo.core import SpikeTrain
from quantities import ms, s, Hz
from elephant.statistics import mean_firing_rate, time_histogram, instantaneous_rate
from elephant.kernels import GaussianKernel

from statannotations.Annotator import Annotator as _StAnn
def add_stat_annotation(ax, data=None, x=None, y=None, hue=None,
                        order=None, hue_order=None, box_pairs=None,
                        test='Mann-Whitney', text_format='star', loc='inside',
                        verbose=2, **kwargs):
    if 'line_offset_to_box' in kwargs:
        kwargs['line_offset_to_group'] = kwargs.pop('line_offset_to_box')
    if 'linewidth' in kwargs:
        kwargs['line_width'] = kwargs.pop('linewidth')
    ann = _StAnn(ax, box_pairs, data=data, x=x, y=y, hue=hue,
                 order=order, hue_order=hue_order)
    ann.configure(test=test, text_format=text_format, loc=loc,
                  verbose=verbose, **kwargs)
    return ann.apply_and_annotate()


def plot_results_shuffle(df_cum_sti, df_cum_res, colors, variables_combined, fig = False, ax1=False, ax2=False, upper_limit=0.4, baseline=0):
    if fig == False:
        fig, ([ax1,ax2])  = plt.subplots(1,2, figsize=(14, 4), sharey=True)

    for color, variable,left,right in zip(colors,variables_combined,[ax1],[ax2]):

        # Aligmnent for Stimulus cue
        real = np.array(df_cum_sti.groupby('session').mean(numeric_only=True).drop(columns=['session_shuffle']).mean())
        # real = np.array(np.mean(df_cum_sti.groupby('session').median()))
        times = df_cum_sti
        times = np.array(times.drop(columns=['session','session_shuffle','subject'],axis = 1).columns.astype(float))

        mean_surr = []
        df_lower = pd.DataFrame()
        df_upper = pd.DataFrame()

        df_for_boots = df_cum_sti.groupby('session').mean(numeric_only=True)
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
        real = np.array(df_cum_res.groupby('session').mean(numeric_only=True).drop(columns=['session_shuffle']).mean())
        times = df_cum_res
        times = np.array(times.drop(columns=['session','session_shuffle','subject'],axis = 1).columns.astype(float))

        mean_surr = []
        df_lower = pd.DataFrame()
        df_upper = pd.DataFrame()

        # df_for_boots = df_cum_res.loc[:, df_cum_res.columns != 'session']
        df_for_boots = df_cum_res.groupby('session').mean(numeric_only=True)
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
    path = str(_ANALYSIS_DATA) + '/'
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
    # panel.set_ylabel('Firing rate (spks/s)')
    panel.xaxis.set_visible(False)

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
    panel.xaxis.set_visible(False)

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
                   colors=[COLORRIGHT,COLORLEFT], align = 'Stimulus_ON', j=1, alpha=1, spikes=True, kernel=50, add_state=False):
    cue_on=0
    cue_off=0.4
    start=-2
    stop= 5 + delay

    neuron = new_convolve(df.loc[df.cluster_id==cluster_id], df, kernel, add_state=add_state)

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
    SpikesRight = df.loc[(df[variable] == 1)&(df.cluster_id == cluster_id)&(df.delay == delay)].copy()
    SpikesLeft = df.loc[(df[variable] == 0)&(df.cluster_id == cluster_id)&(df.delay == delay)].copy()

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
            real = df.groupby('session').median(numeric_only=True).reset_index()
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

        real = np.array(df_loop.groupby('session').mean(numeric_only=True)[numeric_columns].mean())
        times = df_loop[numeric_columns].columns.astype(float)


        df_results = pd.DataFrame()
        df_results['times'] = times
        df_results['real'] = real
        df_results = df_results.sort_values(by='times')

        mean_surr = []
        df_lower = pd.DataFrame()
        df_upper = pd.DataFrame()

        df_for_boots = df_loop.groupby('session').mean(numeric_only=True)[numeric_columns]

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

def new_convolve(nx, df, kernel=50, bin_size=20, add_state=False):
    '''
    nx = already selected cluster
    df = dataframe from the session with all trials
    kernel   : Gaussian kernel s.d. in ms (default 50)
    bin_size : histogram bin size in ms (default 20)
    add_state: if True, add a 'state' column from dft.state (default False)
    '''

    frames = []

    for T in df.trial.unique():
        if T > nx.iloc[0].trial_start+1 and T < nx.iloc[0].trial_end+1:
            nxt = nx.loc[nx['trial']==T]['fixed_times']
            dft = df.loc[df['trial']==T]

            times_spikes = nxt * 1000  # ms

            stop_time = (dft.END.unique()[0]) * 1000 * ms
            try:
                start_time = (dft.START_adjusted.unique()[0] - 0.1) * 1000 * ms
            except Exception:
                start_time = dft.START.unique()[0] * 1000 * ms

            spiketrain = SpikeTrain(times_spikes, units=ms, t_stop=stop_time, t_start=start_time)

            histogram_rate = time_histogram([spiketrain], bin_size*ms, output='rate')
            gaus_rate = instantaneous_rate(spiketrain, sampling_period=bin_size*ms,
                                           kernel=GaussianKernel(kernel*ms))
            times_ = gaus_rate.times.rescale(ms)
            firing = gaus_rate.rescale(histogram_rate.dimensionality).magnitude.flatten()

            df_trial = pd.DataFrame({'times': times_, 'firing': firing})
            df_trial['trial'] = T
            df_trial['Delay_OFF'] = dft.Delay_OFF.unique()[0] * 1000
            df_trial['Stimulus_ON'] = dft.Stimulus_ON.unique()[0] * 1000
            df_trial['delay'] = dft.delay.unique()[0]
            df_trial['vector_answer'] = dft.vector_answer.unique()[0]
            df_trial['reward_side'] = dft.reward_side.unique()[0]
            df_trial['hit'] = dft.hit.unique()[0]
            if add_state:
                df_trial['state'] = dft.state.unique()[0]

            frames.append(df_trial)
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

        real = np.array(df_loop.groupby('session').mean(numeric_only=True)[numeric_columns].mean())
        times = df_loop[numeric_columns].columns.astype(float)

        df_lower = pd.DataFrame()
        df_upper = pd.DataFrame()

        for timepoint in times:
            mean_surr = []

            # recover the values for that specific timepoint
            try:
                array = df_cum_sti.loc[(df_cum_sti.trial_type ==variable)&(df_cum_sti['delay'] == delay)].drop(columns='delay').groupby('session').mean(numeric_only=True)[str(timepoint)].to_numpy()
            except:
                array = df_cum_sti.loc[(df_cum_sti.trial_type ==variable)&(df_cum_sti['delay'] == delay)].drop(columns='delay').groupby('session').mean(numeric_only=True)[timepoint].to_numpy()

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
        real = np.array(df_cum_sti.groupby('session').mean(numeric_only=True).mean())
        # real = np.array(np.mean(df_cum_sti.groupby('session').median()))
        times = df_cum_sti
        times = np.array(times.drop(columns=['session','subject'],axis = 1).columns.astype(float))

        mean_surr = []
        df_lower = pd.DataFrame()
        df_upper = pd.DataFrame()

        df_for_boots = df_cum_sti.groupby('session').mean(numeric_only=True)
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
        real = np.array(df_cum_res.groupby('session').mean(numeric_only=True).mean())
        times = df_cum_res
        times = np.array(times.drop(columns=['session','subject'],axis = 1).columns.astype(float))

        mean_surr = []
        df_lower = pd.DataFrame()
        df_upper = pd.DataFrame()

        # df_for_boots = df_cum_res.loc[:, df_cum_res.columns != 'session']
        df_for_boots = df_cum_res.groupby('session').mean(numeric_only=True)
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

        df_grouped = df_cum_sti.groupby('session').mean(numeric_only=True)-df_cum_sti_shuffle.drop(columns=['session_shuffle']).groupby('session').mean(numeric_only=True)
        real = np.array(df_grouped.mean())
        times = np.array(df_grouped.mean().index).astype(float)

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
        df_grouped = df_cum_res.groupby('session').mean(numeric_only=True)-df_cum_res_shuffle.drop(columns=['session_shuffle']).groupby('session').mean(numeric_only=True)
        real = np.array(df_grouped.mean())
        times = np.array(df_grouped.mean().index).astype(float)

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

        df_loop = (df.loc[(df['trial_type'] == variable)].groupby('session').mean(numeric_only=True)
                    - df_shuffle.loc[(df_shuffle['trial_type'] == variable)].groupby('session').mean(numeric_only=True))
        df_loop.fillna(0)
        # df_loop = df.loc[(df['trial_type'] == variable)].groupby('session').mean(numeric_only=True)

        # Select only columns where the column name is a number or can be transformed to a number
        numeric_columns = df_loop.columns[df_loop.columns.to_series().apply(pd.to_numeric, errors='coerce').notna()]

        real = np.array(df_loop.groupby('session').mean(numeric_only=True)[numeric_columns].mean())
        times = np.array(df_loop[numeric_columns].mean().index).astype(float)

        df_results = pd.DataFrame()
        df_results['times'] = times
        df_results['real'] = real
        df_results = df_results.sort_values(by='times')

        if x_range == None:
            x_range = [min(times), max(times)]

        mean_surr = []
        df_lower = pd.DataFrame()
        df_upper = pd.DataFrame()

        df_for_boots = df_loop.groupby('session').mean(numeric_only=True)
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
                                 y_range = [], x_range = None, epoch = 'Stimulus_ON', baseline=0.5,
                                 epoch_markers=None):

    for color, variable, ax in zip(colors, variables_combined, np.repeat(plot, len(variables_combined))):
        try:
            df_loop = df.loc[(df['trial_type'] == variable)]
        except:
            df_loop = df

        df_loop = df_loop.dropna(axis=1, how='all')

        # Select only columns where the column name is a number or can be transformed to a number
        numeric_columns = df_loop.columns[df_loop.columns.to_series().apply(pd.to_numeric, errors='coerce').notna()]

        real = np.array(df_loop.groupby('session').mean(numeric_only=True)[numeric_columns].mean())
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

        df_for_boots = df_loop.groupby('session').mean(numeric_only=True)
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
        else:
            ax.set_xlabel('Time to go cue (s)')

        if epoch_markers is not None:
            for x0, x1, clr, alpha in epoch_markers:
                ax.fill_betweenx(np.arange(-1, 1.15, 0.1), x0, x1, color=clr, alpha=alpha, edgecolor='none')
        elif epoch == 'Stimulus_ON':
            ax.fill_betweenx(np.arange(-1, 1.15, 0.1), 0, 0.4, color='grey', alpha=.4)
        else:
            ax.fill_betweenx(np.arange(-1, 1.15, 0.1), 0, 0.2, color='grey', alpha=.4)

        sns.despine()


# ── Utility / math ────────────────────────────────────────────────────────────

def exp_decay(x, a, tau):
    return a * np.exp(-x / tau)


def repeat_reward_side(row):
    if row['trials'] != 0:
        if row['reward_side'] == row['previous_reward_side']:
            if row['reward_side'] == 1:
                return 2
            else:
                return 1
        else:
            return 0
    else:
        return np.nan


def trials_synch(row):
    """Normalized trial index for synch scripts (uses 'T' / 'total trials')."""
    return row['T'] / row['total trials']


def trials_normalized(row):
    """Normalized trial index for behavior supp scripts (uses 'trials' / 'total_trials')."""
    return np.around(row['trials'] / row['total_trials'], 2)


def trials_label(row):
    if row['T'] < 0.5:
        return 'Early'
    elif row['T'] >= 0.5:
        return 'Late'
    else:
        return 'Mid'


# ── Rolling-window helpers ─────────────────────────────────────────────────────

def compute_window(data, runningwindow, option):
    """Expanding window at session start, trailing window thereafter."""
    performance = []
    end = False
    for i in range(len(data)):
        if data['trials'].iloc[i] <= runningwindow:
            if end == False:
                start = i
                end = True
            performance.append(round(np.mean(data[option].iloc[start:i + 1]), 2))
        else:
            end = False
            performance.append(round(np.mean(data[option].iloc[i - runningwindow:i]), 2))
    return performance


def compute_window_centered(data, runningwindow, option):
    """Centered rolling average: expanding at start, full window in middle, shrinking at end."""
    performance = []
    start_on = False
    for i in range(len(data)):
        if data['trial'].iloc[i] <= int(runningwindow / 2):
            if start_on == False:
                start = i
                start_on = True
            performance.append(round(np.mean(data[option].iloc[start:i + int(runningwindow / 2)]), 2))
        elif i < (len(data) - runningwindow):
            if data['trial'].iloc[i] > data['trial'].iloc[i + runningwindow]:
                if end == True:
                    end_value = i + runningwindow - 1
                    end = False
                performance.append(round(np.mean(data[option].iloc[i:end_value]), 2))
            else:
                start_on = False
                end = True
                performance.append(round(np.mean(data[option].iloc[i - int(runningwindow / 2):i + int(runningwindow / 2)]), 2))
        else:
            performance.append(round(np.mean(data[option].iloc[i:len(data)]), 2))
    return performance


# ── Model figure helpers ───────────────────────────────────────────────────────

def figureplot(new_df_real, new_df, panel):
    Left = 'teal'
    Right = '#FF8D3F'

    df_results = pd.DataFrame()
    df_results['accuracy'] = new_df_real.groupby(['delays', 'session', 'stim'])['hit'].mean()
    df_results.reset_index(inplace=True)
    sns.lineplot(x='delays', y='accuracy', data=df_results, errorbar=('ci', 67),
                 markeredgewidth=0.2, ax=panel, marker='o', color='black',
                 linestyle='', err_style='bars')
    sns.lineplot(x='delays', y='accuracy', hue='stim', data=df_results,
                 markeredgewidth=0.2, ax=panel, marker='o', palette=[Left, Right],
                 linestyle='', err_style='bars', legend=False)

    df_results = pd.DataFrame()
    df_results['accuracy'] = new_df.groupby(['delays', 'session'])['hit'].mean()
    df_results.reset_index(inplace=True)
    sns.lineplot(x='delays', y='accuracy', data=df_results, color='black', ax=panel, markersize=3)

    df_results = pd.DataFrame()
    df_results['accuracy'] = new_df.groupby(['delays', 'stim', 'session'])['hit'].mean()
    df_results.reset_index(inplace=True)
    sns.lineplot(x='delays', y='accuracy', hue='stim', markeredgewidth=0.2, data=df_results,
                 markersize=3, ax=panel, palette=[Left, Right], legend=False)

    panel.set_ylim(0.4, 1)
    panel.hlines(xmin=0, xmax=10, y=0.5, linestyles=':')
    panel.set_xlabel('Delay (s)')
    panel.set_ylabel('Accuracy')
    panel.locator_params(nbins=3)


# ── Synchrony helpers ──────────────────────────────────────────────────────────

def synch_trial(df, T, lower_plot, upper_plot=None, trial=0, start=-2, stop=0,
                color='indigo', surrogates=100, bins=20):
    dft = df.loc[df.trial == T]
    align = 'Stimulus_ON'
    delay = dft.delay.unique()[0]
    dft = dft.loc[(dft['a_' + align] > start) & (dft['a_' + align] < stop)]

    n_neurons = len(df.cluster_id.unique())
    times_spikes = dft['a_' + align].values * 1000 * ms

    stop_time = stop * 1000 * ms
    start_time = start * 1000 * ms

    spiketrain = SpikeTrain(times_spikes, units=ms, t_stop=stop_time, t_start=start_time)
    histogram_rate = time_histogram([spiketrain], bins * ms, output='rate')
    times_ = histogram_rate.times.rescale(s)
    firing_real = histogram_rate.rescale(histogram_rate.dimensionality).magnitude.flatten()

    real_std = np.std(firing_real)

    list_std = []
    for i in range(surrogates):
        random_float_list = np.random.uniform(start, stop, len(times_spikes))
        surrogate_spikes = np.array(random_float_list) * 1000 * ms
        st = SpikeTrain(surrogate_spikes, units=ms, t_stop=stop_time, t_start=start_time)
        hr = time_histogram([st], bins * ms, output='rate')
        times_ = hr.times.rescale(s)
        firing = hr.rescale(hr.dimensionality).magnitude.flatten()
        list_std.append(np.std(firing))

    cluster_id = []
    FR_mean = []
    for N in df.cluster_id.unique():
        spikes = dft.loc[dft.cluster_id == N]['a_' + align].values
        FR_mean.append(len(spikes) / abs(stop - start))
        cluster_id.append(N)

    df_spikes = pd.DataFrame(list(zip(cluster_id, FR_mean)), columns=['cluster_id', 'FR'])
    df_spikes = df_spikes.sort_values('FR')
    df_spikes['new_order'] = np.arange(len(df_spikes))
    dft = pd.merge(df_spikes, dft, on=['cluster_id'])

    print('Synch:', real_std / np.mean(list_std), '; WM:', str(dft.WM_roll.unique()[0]))

    if upper_plot is not None:
        panel = upper_plot
        panel.set_title(trial)
        j = 0
        for N in dft.new_order.unique():
            spikes = dft.loc[dft.new_order == N]['a_' + align].values
            j += 1
            panel.plot(spikes, np.repeat(j, len(spikes)), '|', markersize=1, color='black', zorder=1)

    panel = lower_plot
    panel.plot(times_, firing_real / n_neurons * 1000, color=color, linewidth=0.5)
    panel.set_ylim(0, 20)
    panel.set_ylabel('Firing rate\n(spks/s)')


def distribution(df_final, variable='WM_roll'):
    r_value_upper = []
    for animal in df_final.animal.unique():
        test_df = df_final.loc[df_final.animal == animal].dropna()
        corr_synch = test_df['synch'].values
        r_value_list = []
        for animal in df_final.animal.unique():
            try:
                corr_WM = df_final.loc[df_final.animal == animal][variable].values[-len(corr_synch):]
                slope, intercept, r_value, p_value, std_err = stats.linregress(corr_synch, corr_WM)
                r_value_list.append(r_value)
            except Exception:
                continue
        r_value_upper.append(np.mean(r_value_list))
    return r_value_upper


# ── Decoder plot variants ──────────────────────────────────────────────────────

def plot_decoder_shuffle(left, df_cum_sti, df_shuffle, baseline=0.5,
                         individual_sessions=False, colors=['black'],
                         upper_limit=0.2, variables_combined=['WM_roll_1']):
    """plot_decoder variant that subtracts a shuffle baseline before plotting."""
    for color, variable, left in zip(colors, variables_combined, [left]):
        if individual_sessions:
            real = df_cum_sti.groupby('session').median(numeric_only=True).reset_index()
            try:
                times = np.array(df_cum_sti.columns[:-4]).astype(float)
            except Exception:
                times = np.array(df_cum_sti.columns[1:]).astype(float)
            for i in range(len(real)):
                left.plot(times, real.iloc[i][1:-1], color=color, alpha=0.1)

        real = np.array(df_cum_sti.loc[:, (df_cum_sti.columns != 'session_shuffle')
                                       & (df_cum_sti.columns != 'fold')
                                       & (df_cum_sti.columns != 'train')
                                       & (df_cum_sti.columns != 'session')
                                       & (df_cum_sti.columns != 'subject')].mean())

        df_shuffle = df_shuffle.groupby('times').median(numeric_only=True).reset_index()
        df_shuffle_mean = np.array(df_shuffle.loc[:, (df_shuffle.columns != 'times')
                                                  & (df_shuffle.columns != 'fold')].mean(axis=1))

        try:
            times = np.array(df_cum_sti.columns[:-4]).astype(float)
            time_points = df_cum_sti.columns[:-4]
        except Exception:
            times = np.array(df_cum_sti.columns[1:]).astype(float)
            time_points = df_cum_sti.columns[1:]

        df_lower = pd.DataFrame()
        df_upper = pd.DataFrame()
        df_for_boots = (df_cum_sti.loc[:, (df_cum_sti.columns != 'session_shuffle')
                                       & (df_cum_sti.columns != 'fold')]
                        .groupby('session').mean(numeric_only=True).reset_index())

        for timepoint in time_points:
            mean_surr = []
            array = df_for_boots[timepoint].to_numpy()
            for _ in range(1000):
                x = np.random.choice(array, size=len(array), replace=True)
                mean_surr.append(np.mean(x))
            df_lower.at[0, timepoint] = np.percentile(mean_surr, 0.5)
            df_upper.at[0, timepoint] = np.percentile(mean_surr, 99.5)

        lower = df_lower.iloc[0].values
        upper = df_upper.iloc[0].values
        left.plot(times, lower - df_shuffle_mean, color=color, linestyle='', alpha=0.6, linewidth=0)
        left.plot(times, upper - df_shuffle_mean, color=color, linestyle='', alpha=0.6, linewidth=0)
        left.fill_between(times, lower - df_shuffle_mean, upper - df_shuffle_mean,
                          alpha=0.2, color=color, linewidth=0)
        left.plot(times, real - df_shuffle_mean, color=color)
        left.fill_betweenx(np.arange(-baseline - 0.1, baseline + .5, 0.1), 0, 0.35,
                           color='lightgrey', alpha=1, linewidth=0)
        left.fill_betweenx(np.arange(-baseline - 0.1, baseline + .5, 0.1), 3.35, 3.55,
                           color='lightgrey', alpha=1, linewidth=0)
        left.set_ylim(baseline - 0.1, upper_limit + baseline)
        left.axhline(y=baseline, linestyle=':', color='black')
        left.set_xlabel('Time from Cue onset (s)')
        left.set_ylabel('Decoding\n accuracy')


def plot_decoder_single(left, df_cum_sti, baseline=0.5, individual_sessions=False,
                        align='Stimulus_ON', colors=['black'], upper_limit=0.2,
                        alpha=1, variables_combined=['WM_roll_1']):
    """Mean-trace-only decoder plot with no bootstrap CI bands."""
    for color, variable, left in zip(colors, variables_combined, left):
        if individual_sessions:
            real = (df_cum_sti.loc[(df_cum_sti['trial_type'] == variable)]
                    .groupby('session').mean(numeric_only=True).drop(columns=['fold', 'score']).reset_index())
            times = df_cum_sti.loc[(df_cum_sti['trial_type'] == variable)]
            try:
                times = np.array(times.drop(columns=['trial_type', 'session', 'fold', 'score'],
                                            axis=1).columns.astype(float))
            except Exception:
                times = np.array(times.drop(columns=['trial_type', 'session', 'fold', 'score', 'subject'],
                                            axis=1).columns.astype(float))
            for i in range(len(real)):
                left.plot(times, real.iloc[i][1:-1], color=color, alpha=0.1)

        try:
            times = df_cum_sti.loc[(df_cum_sti['trial_type'] == variable)]
            times = np.array(times.drop(columns=['trial_type', 'session', 'fold', 'score'],
                                        axis=1).columns.astype(float))
            real = np.array(df_cum_sti.loc[(df_cum_sti['trial_type'] == variable)]
                                    .groupby('session').mean(numeric_only=True).drop(columns=['fold', 'score']).mean())
        except Exception:
            try:
                times = df_cum_sti.loc[(df_cum_sti['trial_type'] == variable)]
                times = np.array(times.drop(columns=['trial_type', 'session', 'score_type'],
                                            axis=1).columns.astype(float))
                real = np.array(df_cum_sti.loc[(df_cum_sti['trial_type'] == variable)]
                                        .groupby('session').mean(numeric_only=True).mean())
            except Exception:
                times = df_cum_sti.loc[(df_cum_sti['trial_type'] == variable)]
                times = np.array(times.drop(columns=['subject', 'trial_type', 'session', 'fold', 'score'],
                                            axis=1).columns.astype(float))
                real = np.array(df_cum_sti.loc[(df_cum_sti['trial_type'] == variable)]
                                        .groupby('session').mean(numeric_only=True).drop(columns=['fold', 'score']).mean())

        left.plot(times, real, color=color, alpha=alpha)
        left.set_ylim(baseline - 0.1, upper_limit + baseline)
        left.axhline(y=baseline, linestyle=':', color='black')
        left.set_ylabel('Decoding\n accuracy')


# ── Notebook helpers ───────────────────────────────────────────────────────────

def plot_lines(data, ax, variable='total_rewards', group='patch_label',
               one_line='mouse', order=None):
    """Draw individual connecting lines between two conditions on a grouped boxplot."""
    if order is None:
        order = sorted(data[group].unique())
    x_map = {label: i for i, label in enumerate(order)}
    for value in data[one_line].unique():
        df_subset = data[data[one_line] == value]
        y = df_subset[variable].values
        x_labels = df_subset[group].values
        x = [x_map[label] for label in x_labels]
        lst_new = [x[0] - 0.925, x[1] - 0.075]
        ax.plot(lst_new, y, marker='', linestyle='-', color='black', alpha=0.4, linewidth=1)
