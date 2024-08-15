# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 17:11:14 2023

@author: Tiffany
"""
COLORLEFT = 'teal'
COLORRIGHT = '#FF8D3F'

from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn import metrics

import json
from datahandler import Utils

from scipy import stats
from matplotlib import gridspec
import random 
from time import process_time
import statsmodels.api as sm

import os 
import pandas as pd
import scipy.io
import seaborn as sns
sns.set_context('talk')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
# warnings.simplefilter(action='ignore', category=PerformanceWarning)
warnings. filterwarnings('ignore', category=UserWarning)
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
pd.options.mode.chained_assignment = None  # default='warn'

save_path = 'C:/Users/Tiffany/Google Drive/WORKING_MEMORY/PAPER/Panel figures/Fig. 4. Ephys WM//'

def plot(df_cum_sti, df_cum_shuffle):
    y_lower = 0
    y_upper = 0
    baseline = 0.5
    delays = [10]
    for delay in delays:
        fig, ax1 = plt.subplots(1,1, figsize=(10, 4), sharey=True)
                    
        for color, variable in zip(colors,variables_combined):
            print(variable)
            individual_sessions = False
            if individual_sessions == True:
                # Aligmnent for Stimulus cue - sessions separately
                real = df_cum_sti.loc[(df_cum_sti['trial_type'] == variable)&(df_cum_sti['delay'] == delay)].groupby('session').median().reset_index().drop(columns=['session','fold','delay','score'])
                try:
                    times = df_cum_sti.loc[(df_cum_sti['trial_type'] == variable)&(df_cum_sti['delay'] == delay)]
                    times = np.array(times.drop(columns=['fold','score','trial_type', 'delay','session'],axis = 1).columns.astype(float))
                except:
                    times = np.array(df_cum_sti.columns[1:]).astype(float)
            
                left.set_xlabel('Time (s) to Cue')
            
                x=times
                for i in range(len(real)):
                    ax1.plot(times,real.iloc[i], color=color,alpha=0.1)
                    
            # Aligmnent for Stimulus cue
            real = np.array(np.mean(df_cum_sti.loc[(df_cum_sti['trial_type'] == variable)&(df_cum_sti['delay'] == delay)].groupby('session').median().drop(columns=['fold','delay','score'])))
            
        #     times = np.array(df_cum_sti.loc[:, df_cum_sti.columns != 'trial_type' and df_cum_sti.columns != 'trial_type'].columns).astype(float)
            times = df_cum_sti.loc[(df_cum_sti['trial_type'] == variable)&(df_cum_sti['delay'] == delay)]
            times = np.array(times.drop(columns=['fold','score','trial_type', 'delay','session'],axis = 1).columns.astype(float))
    
            if color=='crimson':
                left.set_xlabel('Time (s) to Cue')
            sns.despine()
    
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
    
                df_lower.at[0,timepoint] = np.percentile(mean_surr, 5)
                df_upper.at[0,timepoint] = np.percentile(mean_surr, 95)
    
            lower =  df_lower.iloc[0].values
            upper =  df_upper.iloc[0].values
            x=times
    
            ax1.plot(times,real, color=color)
            ax1.plot(x, lower, color=color, linestyle = '',alpha=0.6)
            ax1.plot(x, upper, color=color, linestyle = '',alpha=0.6)
            ax1.fill_between(x, lower, upper, alpha=0.2, color=color)
            ax1.set_ylim(0.4,0.8)
            ax1.axhline(y=baseline,linestyle=':',color='black')
            ax1.fill_betweenx(np.arange(baseline-0.1,baseline+0.5,0.1), 0,0.4, color='grey', alpha=.4)
            ax1.fill_betweenx(np.arange(baseline-0.1,baseline+0.5,0.1), delay+0.4,delay+0.6, color='beige', alpha=.8)
            if color=='crimson':
                ax1.set_xlabel('Time (s) to Go')
    
            sns.despine()
        plt.tight_layout()
        # plt.savefig(save_path+'/delay_'+str(delay)+'_'+align+'_'+trials+'start'+str(start)+'_stop'+str(stop)+'_summary.svg', dpi=300, bbox_inches='tight') 
        plt.show()
        
def interval_extraction_trial(df, cluster_list=[], variable = 'vector_answer', align = 'Delay_OFF', start = 0, stop = 1, delay_only=False):
    y = []
    d = {}
    
    if delay_only == False:
        # print('Skipping delays')
        if align == 'Delay_OFF' and start < 0:
            df = df.loc[(df.delay != 0.1) & (df.delay != 0.2)]
        if align == 'Delay_OFF' and start < -1:
            df = df.loc[(df.delay != 0.1) & (df.delay != 0.2) & (df.delay != 1)]

        if align == 'Stimulus_ON' and stop > 0.5:
            df = df.loc[(df.delay != 0.1) & (df.delay != 0.2)]

        if align == 'Stimulus_ON' and stop > 1.5:
            df = df.loc[(df.delay != 0.1) & (df.delay != 0.2) & (df.delay != 1)]
    
    # print('Recovered from: ', str(len(df.trial.unique())), ' trials')
    # Create new aligment to the end of the session
    df['a_'+align] = df.fixed_times-df[align]

    # cluster_list = df_all.cluster_id.unique()
    df = df.sort_values('trial')
    
    y = df.groupby('trial').mean()[variable]

    # Filter for the spikes that occur in the interval we are analyzing
    df = df.loc[(df['a_'+align]>start)&(df['a_'+align]<stop)]

    df_final = pd.DataFrame()
    df_final = df.groupby(['trial','cluster_id']).count()
    df_final.reset_index(inplace=True)
    df_final = df_final.pivot_table(index=['trial'], columns='cluster_id', values='fixed_times', fill_value=0).rename_axis(None, axis=1)
    df_final = df_final.reindex(cluster_list, axis=1,fill_value=0)

    result = pd.merge(df_final, y, how="right", on=["trial"]).fillna(0)
    result = result.rename(columns={variable: "y"})
    result['y'] = np.where(result['y'] == 0, -1, result['y']) 
    
    return result, result['y']

def train(df, decode='vector_answer', align='Delay_OFF', start=-0.5, stop=0, cluster_list = [], ratio=0.65, test_index=[],  train_index=[], fakey=[], delay_only=False):

    df_final, y = interval_extraction_trial(df,variable = decode, align = align, start = start, stop = stop, cluster_list = cluster_list, delay_only=delay_only)
    
    # This is mainly for the session shuffles
    if len(fakey) > 1:
        print('Using shuffled session')
        y = fakey[len(fakey)-len(y):]
        df_final['y'] = y   
        
    train_cols = df_final.columns
    
    #Train the model   
    df_final.reset_index(inplace=True)
    df_final = df_final.drop(columns ='trial')
    
    if len(test_index) >= 1:
        print('Using splits')
        train = df_final.loc[train_index,:]
        test = df_final.loc[test_index,:]
        # print('Fold',str(fold_no),'Class Ratio:',sum(test['y'])/len(test['y']))
        x_test = test.iloc[:, test.columns != 'y']
        y_test = test['y']
        x_train = train.iloc[:, train.columns != 'y']
        y_train = train['y']
        
    else:
        x_train = df_final.iloc[:, df_final.columns != 'y']
        y_train = df_final['y']
        x_test = x_train
        y_test = y_train
        
    #Normalize the X data
    sc = RobustScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.fit_transform(x_test)
    
    model = LogisticRegression(solver='liblinear', penalty = 'l2', fit_intercept=True).fit(x_train, y_train)
    # model = LogisticRegression(solver='liblinear', penalty = 'l1', C=0.95, fit_intercept=True).fit(x_train, y_train)
    train_cols = df_final.columns
    
    p_pred = model.predict_proba(x_test)    
    y_pred = model.predict(x_test)    
    f1score= f1_score(y_test, y_pred, average='weighted')

    y_test = np.where(y_test == -1, 0, y_test) 
    y_new = y_test.reshape(len(y_test), 1).astype(int)
    # y_new = y_test.values.reshape(len(y_test), 1).astype(int)
    score_ =  np.take_along_axis(p_pred,y_new,axis=1)   

    # print('Trained model on ', len(train_cols), ' neurons.')
    print('score:', np.mean(score_), 'f1_score ', f1score)
    
    return model, train_cols, np.mean(score_)

def test(df,epoch='Stimulus_ON',initrange=-0.4,endrange=1.5,r=0.2, model = None, train_cols=None, variable='ra_accuracy',
                      hit=1, nsurrogates = 100, decode='vector_answer', ratio=0, cluster_list = [], test_index=[], fakey=[], delay_only=False):
    '''
    Function that tests a previously trained function (func. train_decoder) on population activity of specific segments
    
    Attributes
        - df: DataFrame. it contains a whole ephys session without curation. 
        - WM and RL are the variables to consider a trial in the RL or in the WM-module. Both need to be floats. 
        - epoch: str. Moment at which the data will be aligned to. 
        - initrange: float. 
        - endrange: float.
        - r: float 
        - model. function. 
        - train_cols
        - name. String
        - variables. List. 
        - hits. List. 
        - colors. List
        - nsurrogates. Int. 
        - indexes. List 
        - decode. String
    
    Return
        - df_real
        - df_iter
        It will also make a plot. 
    '''
    
    df_real = pd.DataFrame()
    df_iter = pd.DataFrame()
        
    times = [] # Timestamps
    real_score = [] # real scoring of the decoded
    mean_sur=[] # mean of the surrogate data

    for start, stop in zip(np.arange(initrange,endrange-r,r/2),np.arange(initrange+r,endrange,r/2)):
        times.append((start+stop)/2)
        df_final, y = interval_extraction_trial(df,variable = decode, align = epoch, start = start, stop = stop, cluster_list=cluster_list, delay_only=delay_only)

        # Sometimes the testing and the trainind dataset have different neurons since they are looking at different trials and perhaps there were no spikes
        # coming from all neurons. We compare which columns are missing and add them containing 0 for the model to work. 
        test_cols = df_final.columns
        common_cols = train_cols.intersection(test_cols)
        train_not_test = train_cols.difference(test_cols)
        for col in train_not_test:
            df_final[col] = 0

        #The other way round. When training in segmented data, sometimes the training set is smaller than the testing (for instance, when training in Hb trials and testing in WM)
        test_not_train = test_cols.difference(train_cols)
        for col in test_not_train:
            df_final.drop(columns=[col],inplace=True)

        #Train the model"
        if len(test_index) >= 1:
            print('Train splitting trials')
            # Split data in training and testing
            # x_train, x_test, y_train, y_test =\
            #     train_test_split(df_final, y, test_size=test_sample,random_state=random_state)
            
            df_final.reset_index(inplace=True)
            df_final = df_final.drop(columns ='trial')
            test = df_final.loc[test_index,:]
            # print('Fold',str(fold_no),'Class Ratio:',sum(test['y'])/len(test['y']))
            x_test = test.iloc[:, test.columns != 'y']
            y_test = test['y']             

        else:
            x_train = df_final.iloc[:, df_final.columns != 'y']
            y_train = df_final['y']
            x_test = x_train
            y_test = y_train
        
        #Normalize the X data
        sc = RobustScaler()
        x_test = sc.fit_transform(x_test)

        p_pred = model.predict_proba(x_test)
        y_pred = model.predict(x_test)
        score_ = model.score(x_test, y_test)
        
        y_test = np.where(y_test == -1, 0, y_test) 
        y_new = y_test.reshape(len(y_test), 1).astype(int)
        corrected_score =  np.take_along_axis(p_pred,y_new,axis=1)   

        real_score.append(np.mean(corrected_score))
        # real_score.append(score_)

        # print('score:', score_, 'corrected score: ', np.mean(corrected_score), end='\n\n')

        i=0
        while i <= nsurrogates:
            i+=1
            y_perr = shuffle(y_test)
            # score_ = model.score(x_test, y_perr)

            y_new = y_perr.reshape(len(y_perr), 1).astype(int)
            result =  np.take_along_axis(p_pred,y_new,axis=1)     
            score_  = np.mean(result)

            df_iter = df_iter.append({'iteration': i, 'score': score_, 'times': (start+stop)/2, 'epoch' : epoch, 'variable' : variable+'_'+str(hit)}, ignore_index = True)
        
    times.append('trial_type')
    real_score.append(variable+'_'+str(hit))
    a_series = pd.Series(real_score, index = times)
    df_real = df_real.append(a_series, ignore_index=True)
    
    return df_real, df_iter


# Dataframe used for cumulative analysis
df_cum_sti = pd.DataFrame()
df_cum_shuffle = pd.DataFrame()

os.chdir('C:/Users/Tiffany/Documents/Ephys/summary_complete')
for filename in os.listdir(os.getcwd()):
# for filename in list_of_sessions:
    # 
    if filename[-3:] != 'pdf':
        df = pd.read_csv(filename, sep=',',index_col=0)
    else:
        continue
        
    print(filename, '/ Total session trials: ', len(df.trial.unique()), '/ Number of neurons: ', len(df.cluster_id.unique()))
    
    # df['WM_roll'] = compute_window_centered(df, 3,'WM')

####################################### ----------------- Add 2 more seconds of the previous trials before the current stimulus
    # df = df.rename(columns={'past_choices_x' : 'past_choices', 'streak_x' : 'streak', 'past_rewards_x' : 'past_rewards'})
    # df = df.drop(columns=['past_choices_y','streak_y', 'past_rewards_y'])

    # Create a DataFrame only with info for the session
    trials = df.groupby(['trial']).mean()
    try:
        trials = trials[['START','END','Delay_ON','Delay_OFF', 'Stimulus_ON', 'Response_ON', 'Lick_ON', 'Motor_OUT','new_trial',
               'vector_answer', 'reward_side', 'hit', 'delay','total_trials', 'T', 'previous_vector_answer', 'previous_reward_side','repeat_choice',
                'WM_roll', 'RL_roll', 'WM', 'RL', 'streak']]
    except:
        trials = trials[['START','END','Delay_ON','Delay_OFF', 'Stimulus_ON', 'Response_ON', 'Lick_ON', 'Motor_OUT','new_trial',
               'vector_answer', 'reward_side', 'hit', 'delay','total_trials', 'T', 'previous_vector_answer', 'previous_reward_side','repeat_choice',
                'WM_roll', 'RL_roll', 'WM', 'RL']]
    trials = trials.reset_index()

    # Make an aligment to END column
    df['a_END'] = df['fixed_times'] - df['END']

    # Create a new DataFrame with all spikes
    try:
        # Some sessions include the group column that indicates the type of cluste,r other don't
        spikes = df[['trial','fixed_times','a_END','cluster_id', 'group']]
    except:
        spikes = df[['trial','fixed_times','a_END','cluster_id']]

    # Locate spikes that happen 2s prior to end of trial and copy them changing the new_trial index
    duplicate_spikes = spikes.loc[spikes.a_END >-2]
    duplicate_spikes['trial'] +=1 

    # Add the duplicates
    spikes = pd.concat([spikes, duplicate_spikes])

    # Merge trial data with spikes on trial idnex
    df = pd.DataFrame()
    df = pd.merge(trials, spikes, on=["trial"])

    # Create the columns for start and end and change trial to new trial index ( without taking the misses into account)
    # df['trial_start'] = min(df.new_trial)
    # df['trial_end'] = max(df.new_trial)
    # df = df.drop(columns=['trial'])
    # df = df.rename(columns={'new_trial' : 'trial'})

    # This in case we don't do this and want to preserve the orginal trial indexes. 
    df['trial_start'] = min(df.trial)
    df['trial_end'] = max(df.trial)

    # Crate the aligment that ew will need for the analysis. 
    df['a_Stimulus_ON'] =  df['fixed_times'] - df['Stimulus_ON']
    df['a_Lick_ON'] =  df['fixed_times'] - df['Lick_ON']
    df['a_Delay_OFF'] =  df['fixed_times'] - df['Delay_OFF']
    df['a_Motor_OUT'] =  df['fixed_times'] - df['Motor_OUT']
    df['a_Response_ON'] =  df['fixed_times'] - df['Response_ON']
    df['START_adjusted'] =  df['START'] - 2.1
    
############################################################# -------------------------------------------------------------------------

    substract = False

    # Variables used for decoder training
    decode = 'vector_answer'
    align='Delay_OFF'
    ratio = 0.6
    start = -0.5
    stop = 0
    type_trial ='WM_roll'
    hit = 1
    nsplits = 5
    
    #Variables for testing
    colors=['darkgreen','crimson', 'indigo']
    variables = ['WM_roll','WM_roll','RL_roll']
    hits = [1,0,'all']
    ratios = [0.6,0.6,0.35]
    variables_combined=[variables[0]+'_'+str(hits[0]),variables[1]+'_'+str(hits[1]),variables[2]+'_'+str(hits[2])]

    # colors=['crimson','darkgreen']
    # variables = ['WM_roll','WM_roll']
    # hits = [0,1]
    # ratios = [0.5,0.6]
    # variables_combined=[variables[0]+'_'+str(hits[0]),variables[1]+'_'+str(hits[1])]

    # colors=['crimson','darkgreen','indigo','purple']
    # variables = ['WM_roll','WM_roll','RL_roll','RL_roll']
    # hits = [0,1,1,0]
    # ratios = [0,0.6,0.35,0.35]
    # variables_combined=[variables[0]+'_'+str(hits[0]),variables[1]+'_'+str(hits[1]),variables[2]+'_'+str(hits[2]),variables[3]+'_'+str(hits[3])]

    cluster_list = df.cluster_id.unique()

    skf = StratifiedKFold(n_splits=nsplits, shuffle=True)
    
    # Create a dataframe for training data
    if type_trial == 'all':
        df_train = df.loc[(df.hit==hit)]
    elif hit == 'all':
        df_train = df.loc[(df[type_trial]>=ratio)]
    else:
        df_train = df.loc[(df[type_trial]>=ratio)&(df.hit==hit)]
        
    df_train = df_train.loc[(df.delay!=0.2)&(df.delay!=0.1)]
    
    df_final, y = interval_extraction_trial(df_train, variable = decode, align = align, start = start, stop = stop, cluster_list=cluster_list)
    df_final.reset_index(inplace=True)
    df_final = df_final.drop(columns ='trial')
    
    fold_no = 1
    if len(y) < nsplits:
        print('Skip session because not enough trials')
        continue
        
    for train_index, test_index in skf.split(df_final, y):
        
        print('Fold_no:', fold_no)
        model, train_cols, score = train(df_train, decode=decode, align=align, start=start,stop=stop, cluster_list = cluster_list, 
                                  test_index=test_index,  train_index=train_index)
        
        if score <0.6:
            continue
        # Remove a fifth of the dataset so it can be compared to crossvalidated data. If we want to randomly reduce it, add reduce to trian function
        # drop_list = np.array_split(df_train.trial.unique(), 5)[fold_no]
        # df_train = df_train[~df_train['trial'].isin(drop_list)]
        # index_train_trials = df_train.trial.unique()
        # print('Total of left: ', len(df_train.loc[df_train['vector_answer'] == 0].groupby('trial').mean()), '; Total of right: ', len(df_train.loc[df_train['vector_answer'] == 1].groupby('trial').mean()))

        for delay in df.delay.unique():
            try:
                df_delay = df.loc[np.around(df.delay,1)==delay]
                delay=np.around(df_delay.delay.iloc[0],1)
                print('Delay:', delay)
            except:
                continue

            if delay == 0.1:
                endrange=3.5
                r=0.5
                continue
            elif delay == 1:
                endrange=6.5
                r=0.5
                continue
            elif delay == 3:
                endrange=9.5    
                r=0.5
                continue
            elif delay == 10:
                endrange=16.5
                r=0.5

            if delay == 0.1:
                fig, ax1 = plt.subplots(1,1, figsize=(10, 4), sharey=True)
            elif delay == 1:
                fig, ax1 = plt.subplots(1,1, figsize=(10, 4), sharey=True)
            elif int(delay) == 3:
                fig, ax1 = plt.subplots(1,1, figsize=(12, 4), sharey=True)
            elif delay == 10:
                fig, ax1 = plt.subplots(1,1, figsize=(14, 4), sharey=True)

            df_res = pd.DataFrame()
            df_sti = pd.DataFrame()
            df_iter = pd.DataFrame()

            for color, variable,hit,ratio,left in zip(colors,variables,hits,ratios,[ax1,ax1,ax1,ax1]):

                # Create a dataframe for testing data
                if variable == 'all':
                    df_test = df_delay.loc[(df_delay.hit==hit)]
                elif hit == 'all':
                    df_test = df_delay.loc[(df_delay[variable]>=ratio)]
                else:
                    df_test = df_delay.loc[(df_delay[variable]>=ratio)&(df_delay.hit==hit)]

                # print(variable, 'Threshold:', ratio, 'Hit:', hit, 'Nº of trials:', len(df_test.trial.unique()))

    # -----------  Remove the trials that overlap with the training set.
                list_train_trials = df_train.trial.unique()[train_index]
                df_test = df_test[~df_test['trial'].isin(list_train_trials)] 
                
                if len(df_test.trial.unique())<5:
                    # print('Not enough trials with this condition')
                    continue

                df_real,df_temp = test(df_test, decode= decode,epoch='Stimulus_ON',initrange=-2,endrange=endrange, r=r, model = model, delay_only=delay, 
                                                  variable=variable, hit=hit, nsurrogates = 50,train_cols = train_cols, cluster_list = cluster_list)

                df_sti = pd.concat([df_real,df_sti])
                df_iter = pd.concat([df_iter,df_temp])

                variable = str(variable)+'_'+str(hit)

                # Aligmnent for Stimulus cue
                real = df_sti.loc[(df_sti['trial_type'] ==variable)].mean(axis=0).to_numpy()
                times = np.around(np.array(df_sti.columns)[:-1].astype(float),2)

                df_new= df_iter.loc[(df_iter.epoch=='Stimulus_ON')].groupby('times')['score']
                y_mean= df_new.mean().values
                lower =  df_new.quantile(q=0.975, interpolation='linear')-y_mean
                upper =  df_new.quantile(q=0.025, interpolation='linear')-y_mean
                x=times

                left.set_xlabel('Time (s) to Cue')

                if substract == True:
                    left.plot(times,real-y_mean, color=color)
                    left.plot(x, lower+real-y_mean, color=color, linestyle = '',alpha=0.6)
                    left.plot(x, upper+real-y_mean, color=color, linestyle = '',alpha=0.6)
                    left.fill_between(x, lower+real-y_mean, upper+real-y_mean, alpha=0.2, color=color)
                    left.set_ylim(-0.5,0.6)
                    left.axhline(y=0.0,linestyle=':',color='black')
                    left.fill_betweenx(np.arange(-1,1.15,0.1), 0,0.4, color='grey', alpha=.4)
                    left.fill_betweenx(np.arange(-1.1,1.1,0.1), delay+0.3,delay+0.5, color='beige', alpha=.8)
                    try:
                        a_series = pd.DataFrame(pd.Series(real-y_mean, index = times)).T
                        a_series['trial_type'] = variable
                        a_series['session'] = filename
                        a_series['delay'] = delay
                        a_series['score'] = score    
                        a_series['fold'] = fold_no 

                        df_cum_sti = df_cum_sti.append(a_series, ignore_index=True)

                        df_cum_iter=pd.DataFrame()
                        df_cum_iter['times'] = df_iter.groupby('times').score.mean().reset_index()['times'].values
                        df_cum_iter['delay'] = delay
                        df_cum_iter['session'] = filename
                        df_cum_iter['fold'] = fold_no 
                        df_cum_iter['trial_type'] = variable 

                        for iteration in df_iter.iteration.unique():
                            df_cum_iter[iteration]= df_iter.loc[(df_iter.iteration==iteration)].groupby('times').mean()['score'].values - df_iter.groupby('times').mean()['score'].values

                        df_cum_shuffle = pd.concat([df_cum_iter, df_cum_shuffle])
                    except:
                        print('Did not add to summary: ', variable)
                        continue
                        
                elif substract == False:
                    left.plot(times,real, color=color)
                    left.plot(x, lower+real, color=color, linestyle = '',alpha=0.6)
                    left.plot(x, upper+real, color=color, linestyle = '',alpha=0.6)
                    left.fill_between(x, lower+real, upper+real, alpha=0.2, color=color)
                    left.set_ylim(0,1)
                    left.axhline(y=0.5,linestyle=':',color='black')
                    left.fill_betweenx(np.arange(0,1.15,0.1), 0,0.4, color='grey', alpha=.4)
                    left.fill_betweenx(np.arange(0,1.1,0.1), delay+0.35,delay+0.55, color='beige', alpha=.8)

                    try:
                        a_series = pd.DataFrame(pd.Series(real, index = times)).T
                        a_series['trial_type'] = variable
                        a_series['session'] = filename
                        a_series['delay'] = delay
                        a_series['score'] = score    
                        a_series['fold'] = fold_no 

                        df_cum_sti = df_cum_sti.append(a_series, ignore_index=True)

                        df_cum_iter=pd.DataFrame()
                        df_cum_iter['times'] = df_iter.groupby('times').score.mean().reset_index()['times'].values
                        df_cum_iter['delay'] = delay
                        df_cum_iter['session'] = filename
                        df_cum_iter['fold'] = fold_no 
                        df_cum_iter['trial_type'] = variable 

                        for iteration in df_iter.iteration.unique():
                            df_cum_iter[iteration]= df_iter.loc[(df_iter.iteration==iteration)].groupby('times').mean()['score'].values

                        df_cum_shuffle = pd.concat([df_cum_iter, df_cum_shuffle])
                    except:
                        print('Did not add to summary: ', variable)
                        continue

                sns.despine()
                plt.close()
                
        plt.tight_layout()
        plt.show()
        fold_no+=1
        
