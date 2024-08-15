# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 11:51:57 2022

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
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from scipy import stats
from scipy import special
import json 
from sklearn.linear_model import LogisticRegression
from scipy.optimize import curve_fit
from matplotlib.lines import Line2D
import os
import pandas as pd
import numpy as np
import seaborn as sns
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from statannot import add_stat_annotation

import warnings

utilities = 'G:/Mi unidad/WORKING_MEMORY/PAPER/WM_manuscript_FIGURES/'
os.chdir(utilities)
import functions as plots

path = 'G:/Mi unidad/WORKING_MEMORY/PAPER/WM_manuscript_FIGURES/Data/'
save_path = 'G:/Mi unidad/WORKING_MEMORY/PAPER/ANALYSIS_figures/'

os.chdir(path)

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
cm = 1/2.54

# Create a figure with 6 subplots using a GridSpec
fig = plt.figure(figsize=(17*cm, 15*cm))
gs = gridspec.GridSpec(nrows=4, ncols=8, figure=fig)

# Create the subplots
a = fig.add_subplot(gs[0, 2:4])
a2 = fig.add_subplot(gs[0, 4:6])
b = fig.add_subplot(gs[0, 6:8])
b2 = fig.add_subplot(gs[1, 0:2])
c = fig.add_subplot(gs[1, 2:4])
c2 = fig.add_subplot(gs[1, 4:6]) 
# d = fig.add_subplot(gs[1, 3:5])  # span 2 rows and 2 columns
d = fig.add_subplot(gs[1, 6:8])  # span 2 rows and 1 column
e = fig.add_subplot(gs[3, 0:2])
f = fig.add_subplot(gs[3, 2:4])
g = fig.add_subplot(gs[3, 4:6])
h = fig.add_subplot(gs[3, 6:8])

i = fig.add_subplot(gs[2, 0:2])
j = fig.add_subplot(gs[2, 2:4])

fig.text(0.01, 1, 'a', fontsize=10, fontweight='bold', va='top')
fig.text(0.25, 1, 'b', fontsize=10, fontweight='bold', va='top')
fig.text(0.5, 1, 'c', fontsize=10, fontweight='bold', va='top')
fig.text(0.75, 1, 'd', fontsize=10, fontweight='bold', va='top')
fig.text(0.01, 0.75, 'e', fontsize=10, fontweight='bold', va='top')
fig.text(0.25, 0.75, 'f', fontsize=10, fontweight='bold', va='top')
fig.text(0.5, 0.75, 'g', fontsize=10, fontweight='bold', va='top')
fig.text(0.75, 0.75, 'h', fontsize=10, fontweight='bold', va='top')

fig.text(0.01, 0.5, 'i', fontsize=10, fontweight='bold', va='top')
fig.text(0.25, 0.5, 'j', fontsize=10, fontweight='bold', va='top')
fig.text(0.5, 0.5, 'k', fontsize=10, fontweight='bold', va='top')
# fig.text(0.75, 0.5, 'j', fontsize=10, fontweight='bold', va='top')

fig.text(0.01, 0.25, 'l', fontsize=10, fontweight='bold', va='top')
fig.text(0.25, 0.25, 'm', fontsize=10, fontweight='bold', va='top')
fig.text(0.5, 0.25, 'n', fontsize=10, fontweight='bold', va='top')
fig.text(0.75, 0.25, 'o', fontsize=10, fontweight='bold', va='top')

# -----------------############################## FUNCTIONS #################################-----------------------

def repeat_reward_side(row):
    '''

    '''
    # Compare the current response with the previous one. If that matches, return a 1 meaning it repeated. 
    if row['trials'] != 0:
        # Compare the current response with the previous one. .  
        if row['reward_side'] == row['previous_reward_side']:
            # if it matched and the answer was 1, it means that it repeated a right response
            if row['reward_side'] == 1:
                return 2
            # if it matched and the answer was 0, it means that it repeated a left response
            else:
                return 1
        # Alternations
        else:
            return 0
    else:
        return np.nan

# -----------------###############################################################-----------------------


#-----------------############################## A Panel #################################-----------------------
file_name = 'global_behavior_10_paper'
df = pd.read_csv(path+file_name+'.csv', index_col=0)

df_results = pd.DataFrame()
df_results['hit'] = df.groupby(['subject','delay_times'])['hit'].mean()
df_results.reset_index(inplace=True)

panel=a
palette = sns.color_palette(['black'], len(df.subject.unique()))
sns.lineplot(x='delay_times',y='hit',hue='subject',data=df_results, marker='', palette = 'Greys', ax=panel, legend=False,err_style=None)
panel.set_ylim(0.45,1)
panel.hlines(xmin=0, xmax=10, y=0.5, linestyles=':')
panel.set_ylabel('Accuracy')
panel.set_xlabel('Delay (s)')
panel.text(x=0,y=0.55, s='N: ' + str(len(df_results.subject.unique())), fontsize=5)
panel.locator_params(nbins=4)

panel=a2
xaxis=[0,1,3,10]
fit = np.polyfit(xaxis,np.array(df_results.groupby('delay_times')['hit'].mean()),1)
fit_fn = np.poly1d(fit) 
panel.plot(xaxis,np.array(df_results.groupby('delay_times')['hit'].mean()), 'k.', xaxis, fit_fn(xaxis),'k')
panel.errorbar(xaxis,np.array(df_results.groupby('delay_times')['hit'].mean()),yerr=2*np.array(df_results.groupby('delay_times')['hit'].sem()), markersize=1, fmt='o',ecolor='grey',color='black', capsize=2)
panel.set_ylim(0.45,1)
panel.hlines(xmin=0, xmax=10, y=0.5, linestyles=':')
panel.set_xlabel('Delay (s)')
panel.set_ylabel('Accuracy')

slope, intercept, r_value, p_value, std_err = stats.linregress(xaxis,df_results.groupby('delay_times')['hit'].mean())
panel.text(x=0,y=0.55, s= 'Slope= '+str(round(slope,3))+'\nP= '+ str(round(p_value,3)), fontsize=5)
panel.locator_params(nbins=4)

group1 = df.loc[df.delay_times == 0.01].groupby('subject')['hit'].mean()
group2 = df.loc[df.delay_times == 1].groupby('subject')['hit'].mean()
group3 = df.loc[df.delay_times == 3].groupby('subject')['hit'].mean()
group4 = df.loc[df.delay_times == 10].groupby('subject')['hit'].mean()

f_statistic, p_value = stats.f_oneway(group1, group2, group3, group4)

# slope_list = []
# for animal in df_results.subject.unique():
#     values = df_results.loc[df_results.subject == animal]['hit'].values
#     slope, intercept, r_value, p_value, std_err = stats.linregress(xaxis,values)
#     slope_list.append(slope)

# ----------------------------------------------------------------------------------------


# -----------------############################## B Panel #################################-----------------------
df_temp=pd.DataFrame()
df_cumulative = pd.DataFrame()

for animal in df.subject.unique():
    fit = np.polyfit([0,1,3,10],np.array(df.loc[(df['subject'] == animal)].groupby('delay_times')['hit'].mean()),1) 
    fit_fn = np.poly1d(fit) 
    df_temp = pd.DataFrame({'subject': [animal], 'slope': [fit[0]]})
    df_cumulative = pd.concat([df_cumulative,df_temp],sort=True)

df_results = pd.DataFrame()
df_results['slope'] = df_cumulative.groupby(['subject'])['slope'].mean()
df_results['accuracy'] = df.loc[df.delay_times == 10].groupby(['subject'])['hit'].mean()
df_results.reset_index(inplace=True)

palette = sns.color_palette(['black'], len(df.subject.unique()))
sns.regplot(x="slope", y="accuracy",data=df_results, ax=b, color='black',marker='')
sns.scatterplot(x="slope", y="accuracy", hue='subject',data=df_results, palette='Greys',ci=68,ax=b,legend=False)      
b.set_xlabel('Rate of Accuracy decay (($s^-1$)')
b.set_ylabel('Accuracy')
b.set_ylim(0.5,1)
b.set_xlim(-0.02,0)
b.locator_params(nbins=3)
slope, intercept, r_value, p_value, std_err = stats.linregress((np.array(df_results['slope']),np.array(df_results['accuracy'])))
b.text(x=-0.018,y=0.9, s= 'R='+ str(round(r_value**2,3))+'\nP= '+ str(round(p_value,3)), fontsize=5)

# ----------------------------------------------------------------------------------------------------------------


# -----------------############################## B2 Panel #################################-----------------------
df_results = pd.DataFrame()
# df_results['repeat_choice'] = df.loc[df['presented_delays']=='0.01,1.0,3.0,10.0'].groupby(['subject'])['repeat_choice'].mean()
df_results['acc0'] = df.loc[(df['delay_times']==0.01)].groupby(['subject'])['hit'].mean()
df_results['lapse'] = 1 - df_results['acc0']
df_results['accuracy'] = df.groupby(['subject'])['hit'].mean()
df_results.reset_index(inplace=True)

palette = sns.color_palette(['black'], len(df.subject.unique()))
sns.regplot(x="lapse", y="accuracy", data=df_results, color='black',ci=68,ax=b2,marker='.')      
sns.scatterplot(x="lapse", y="accuracy", hue='subject',data=df_results, palette=palette,ci=68,ax=b2,legend=False)      

b2.set_ylabel('Accuracy')
b2.set_xlabel('Lapse rate')
b2.set_ylim(0.6,1)
# b2.set_xlim(0.6,1)
plt.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0, ncol=3).get_frame().set_linewidth(0.0)

slope, intercept, r_value, p_value, std_err = stats.linregress((np.array(df_results['lapse']),np.array(df_results['accuracy'])))
plt.locator_params(nbins=3)
b2.text(x=0.05,y=0.7, s= 'R= '+ str(round(r_value**2,3))+'\nP= '+ str(round(p_value,3)), fontsize=5)
# ----------------------------------------------------------------------------------------------------------------


# -----------------############################## C Panel #################################-----------------------
groupings=['subject','delay_times']
df_results = pd.DataFrame()
df_results['repeat_choice'] = 0.5* df.loc[(df['repeat_choice_side']==1)].groupby(groupings)['valids'].count()/df.loc[(df.vector_answer == 0)].groupby(groupings)['valids'].count() + 0.5*df.loc[(df['repeat_choice_side']==2)].groupby(groupings)['valids'].count()/df.loc[(df.vector_answer == 1)].groupby(groupings)['valids'].count()
# df_results['repeat_choice'] =df.loc[(df['presented_delays']==presented_delay)].groupby(groupings)['repeat_choice'].mean()
df_results.reset_index(inplace=True)

palette = sns.color_palette(['black'], len(df.subject.unique()))
sns.lineplot(x='delay_times',y='repeat_choice',hue='subject',data=df_results, legend=False, marker='', palette = 'Greys', ax=c, err_style=None)
c.set_ylim(0.4,0.8)
c.set_ylabel('Repeating bias')
c2.set_ylabel('Repeating bias')
c2.set_xlabel('Delay (s)')
c.set_xlabel('Delay (s)')
c.hlines(y=0.5,xmin=0,xmax=10,linestyle=':')

fit = np.polyfit(xaxis,np.array(df_results.groupby('delay_times')['repeat_choice'].mean()),1)
fit_fn = np.poly1d(fit) 

c2.plot(xaxis,np.array(df_results.groupby('delay_times')['repeat_choice'].mean()), 'k.', xaxis, fit_fn(xaxis), 'k')
c2.errorbar(xaxis,np.array(df_results.groupby('delay_times')['repeat_choice'].mean()),yerr=np.array(df_results.groupby('delay_times')['repeat_choice'].sem()*2), fmt='.', markersize= 1, ecolor='grey',color='black', capsize=2)
c2.hlines(y=0.5,xmin=0,xmax=10,linestyle=':')
#Print the slopes
slope, intercept, r_value, p_value, std_err = stats.linregress((xaxis,np.array(df_results.groupby('delay_times')['repeat_choice'].mean())))
c2.text(x=0,y=0.65, s= 'Slope= '+str(round(slope,3))+'\nP= '+ str(round(p_value,3)), fontsize=5)

group1 = df_results.loc[df_results.delay_times == 0.01].groupby('subject')['repeat_choice'].mean()
group2 = df_results.loc[df_results.delay_times == 1].groupby('subject')['repeat_choice'].mean()
group3 = df_results.loc[df_results.delay_times == 3].groupby('subject')['repeat_choice'].mean()
group4 = df_results.loc[df_results.delay_times == 10].groupby('subject')['repeat_choice'].mean()

f_statistic, p_value = stats.f_oneway(group1, group2, group3, group4)

c.locator_params(nbins=3)
c2.locator_params(nbins=3)
c2.set_ylim(0.4,0.8)

# ----------------------------------------------------------------------------------------------------------------


# -----------------############################## D Panel #################################-----------------------
df_results = pd.DataFrame()
# df_results['repeat_choice'] = df.loc[df['presented_delays']=='0.01,1.0,3.0,10.0'].groupby(['subject'])['repeat_choice'].mean()
df_results['prob_repeat'] = 0.5* df.loc[(df['delay_times']==0.01)&(df['repeat_choice_side']==1)].groupby(['subject'])['valids'].count()/df.loc[(df['delay_times']==0.01)&(df.vector_answer == 0)].groupby(['subject'])['valids'].count() + 0.5*df.loc[(df['delay_times']==0.01)&(df['repeat_choice_side']==2)].groupby(['subject'])['valids'].count()/df.loc[(df['delay_times']==0.01)&(df.vector_answer == 1)].groupby(['subject'])['valids'].count()
df_results['accuracy'] = df.loc[(df['delay_times']==0.01)].groupby(['subject'])['hit'].mean()
df_results['lapse'] = 1 - df_results['accuracy'] 

# df_results['prob_repeat'] = 0.5* df.loc[(df['delay_times']==0.01)&(df['repeat_choice_side']==1)].groupby(['subject'])['valids'].count()/df.loc[(df['delay_times']==0.01)&(df.vector_answer == 0)].groupby(['subject'])['valids'].count() + 0.5*df.loc[(df['delay_times']==0.01)&(df['repeat_choice_side']==2)].groupby(['subject'])['valids'].count()/df.loc[(df['delay_times']==0.01)&(df.vector_answer == 1)].groupby(['subject'])['valids'].count()
# df_results['accuracy'] = df.loc[(df['delay_times']==0.01)].groupby(['subject'])['hit'].mean()
df_results.reset_index(inplace=True)

palette = sns.color_palette(['black'], len(df.subject.unique()))
sns.regplot(x="lapse", y="prob_repeat", data=df_results, color='black',ci=68,ax=d,marker='.')      
sns.scatterplot(x="lapse", y="prob_repeat", hue='subject',data=df_results, palette=palette,ci=68,ax=d,legend=False)      

d.set_xlabel('Lapse rate')
d.set_ylabel('Repeating bias')
d.hlines(y=0.5,xmin=0.0,xmax=0.32,linestyle=':')
plt.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0, ncol=3).get_frame().set_linewidth(0.0)

slope, intercept, r_value, p_value, std_err = stats.linregress((np.array(df_results['prob_repeat']),np.array(df_results['lapse'])))
d.text(x=0.03,y=0.6, s= 'R= '+str(round(r_value**2,3))+'\nP= '+ str(round(p_value,3)), fontsize=5)

df['repeat_reward_side'] = df.apply(repeat_reward_side, axis=1)
df_results = pd.DataFrame()
df_results['repeat_choice'] = (0.5* (df.loc[(df['repeat_reward_side']==1)].groupby(groupings)['valids'].count()
                                     /df.loc[(df.reward_side == 0)].groupby(groupings)['valids'].count()) 
                             + 0.5*(df.loc[(df['repeat_reward_side']==2)].groupby(groupings)['valids'].count()
                                    /df.loc[(df.reward_side == 1)].groupby(groupings)['valids'].count()))
df_results.reset_index(inplace=True)

sns.lineplot(x='delay_times',y='repeat_choice',data=df_results, legend=False, marker='', color='black', ax=c2)
d.set_xlim(0.0,0.35)
plt.locator_params(nbins=3)

# ----------------------------------------------------------------------------------------------------------------

# -----------------############################## E Panel GLM results #################################-----------------------

file_name = 'GLMM_final_data'
coef_matrix = pd.read_csv(path+file_name+'.csv',  index_col=0)
coef_matrix['const'] = 0
for regressor, panel in zip(['SL','SR','SL:D','SR:D','exp_C','D:exp_C'], [e,e,f,f,g,h]):

    if regressor == 'SR' or regressor == 'SR:D' or regressor == 'SR:D:T' or regressor == 'SR:T':
        main= COLORRIGHT
        light = 'grey'
    elif regressor == 'exp_C' or regressor == 'D:exp_C':
        main = 'indigo'
        light = 'grey' 
        
    if regressor=='SR':
        plot = pd.DataFrame({'SL': coef_matrix['SL'], 'SR': coef_matrix['SR']})
        sns.violinplot(data=plot, palette=[COLORLEFT, COLORRIGHT],ax=panel, width=0.5,saturation=0.6,linewidth=0)
        sns.violinplot(data=plot, palette=[COLORLEFT, COLORRIGHT],ax=panel, width=0.5,linewidth=1)
        xA = np.random.normal(1, 0.15, len(coef_matrix))
        const=1
        panel.set_xlabel('Left          Right')
        panel.set_title('Stimulus $S_t$', fontsize=7)
    elif regressor=='SR:D': 
        plot = pd.DataFrame({'SL:D': coef_matrix['SL:D'], 'SR:D': coef_matrix['SR:D']})
        sns.violinplot(data=plot, palette=[COLORLEFT, COLORRIGHT],ax=panel, width=0.5,saturation=0.6,linewidth=0)
        sns.violinplot(data=plot, palette=[COLORLEFT, COLORRIGHT],ax=panel, width=0.5,linewidth=1)
        xA = np.random.normal(1, 0.15, len(coef_matrix))
        panel.set_title('Stimulus x Delay\n $S_t·D_t$', fontsize=7)
        panel.set_xlabel('Left          Right')
        const=1
    elif regressor=='SR:T':
        plot = pd.DataFrame({'SL:T': coef_matrix['SL:T'], 'SR:T': coef_matrix['SR:T']})
        sns.violinplot(data=plot, palette=[COLORLEFT, COLORRIGHT],ax=panel, width=0.5,saturation=0.6,linewidth=0)
        sns.violinplot(data=plot, palette=[COLORLEFT, COLORRIGHT],ax=panel, width=0.5,linewidth=1)
        xA = np.random.normal(1, 0.15, len(coef_matrix))
        const=1
        panel.set_xlabel('Left            Right')
        panel.set_title('Stimulus x Trial $S_t·T_t$', fontsize=7)

    elif regressor=='SR:D:T':
        plot = pd.DataFrame({'SL:D:T': coef_matrix['SL:D:T'], 'SR:D:T': coef_matrix['SR:D:T']})
        sns.violinplot(data=plot, palette=[COLORLEFT, COLORRIGHT],ax=panel, width=0.5,saturation=0.6,linewidth=0)
        sns.violinplot(data=plot, palette=[COLORLEFT, COLORRIGHT],ax=panel, width=0.5,linewidth=1)
        xA = np.random.normal(1, 0.15, len(coef_matrix))
        panel.set_xlabel('Left            Right')
        const=1
    elif regressor == 'SL' or regressor == 'SL:D'or regressor == 'SL:D:T'or regressor == 'SL:T':
        main= COLORLEFT
        light = 'grey'
        xA = np.random.normal(0, 0.15, len(coef_matrix))
        const=0
        pass
    else:
        const=0
        xA = np.random.normal(0, 0.2, len(coef_matrix))
        sns.violinplot(x=coef_matrix['const'].astype(float),y=coef_matrix[regressor].astype(float), data=coef_matrix, width=0.5, color=main,ax=panel, saturation=0.6,linewidth=0)
        sns.violinplot(x=coef_matrix['const'].astype(float),y=coef_matrix[regressor].astype(float), data=coef_matrix, width=0.5, color=main,ax=panel, legend=False,linewidth=1)
        if regressor == 'exp_C':
            panel.set_title('Previous Choices $C_{t-k}$', fontsize=7)
        if regressor == 'D:exp_C':
            panel.set_title('Previous Choices\n x Delay $C_{t-k}·D_t$', fontsize=7)
        panel.set_xlabel('')

    panel.hlines(y=0,xmin=-2,xmax=2,linestyle=':')
    try:
        sns.scatterplot(x=xA,y=regressor,data=coef_matrix,hue=coef_matrix[regressor+'_sig'],palette=[light,main],ax=panel,alpha=0.9,legend=False)
    except:
        if coef_matrix[regressor+'_sig'].all()==1:
            sns.scatterplot(x=xA,y=regressor,data=coef_matrix,color=main,ax=panel,alpha=0.9,legend=False)
        else:
            sns.scatterplot(x=xA,y=regressor,data=coef_matrix,color='grey',ax=panel,alpha=0.9,legend=False)
            
    panel.set_xticks([])
    panel.set(ylabel=None)
    if regressor=='SR' or regressor=='SR:D'  or regressor=='SR:T' or  regressor=='SR:D:T':
        panel.set_xlim(-0.5,1.5)
    else:
        panel.set_xlim(-1.5,1.5)
        
    if regressor=='D:exp_C':
        panel.set_ylim(-0.05,0.05)
    panel.set_ylabel('Weights')
    panel.locator_params(axis='y', nbins=3)
    y_min, y_max = panel.get_ylim() 
    
    # print(regressor)    
    # print(stats.ttest_1samp(coef_matrix[regressor],0)[1])    
    
    if stats.ttest_1samp(coef_matrix[regressor],0)[1] <=0.001:
        panel.text(const-0.13, y_max, '***', fontsize=6) 
        
    elif stats.ttest_1samp(coef_matrix[regressor],0)[1] <=0.01:
        panel.text(const-0.1, y_max, '**')
    
    elif stats.ttest_1samp(coef_matrix[regressor],0)[1] <=0.05:
        panel.text(const-0.05, y_max, '*')
   
    else:
        panel.text(const-0.1, y_max, 'ns', fontsize=6)
    
    plt.gca().tick_params(direction='out') #direction
   
h.set_ylabel('')
# -------------------------------------           Autocorrelogram  for correct  ------------------------------------------
file_name = 'hit_autocorrelation'
# file_name = 'autocorrelation_hit'

corr =pd.read_csv(save_path+file_name+'.csv', index_col=0)

panel=i
df_lower= pd.DataFrame()
df_upper= pd.DataFrame()

# for timepoint in range(len(corr)):
#     mean_surr = []
#     # recover the values for that specific timepoint
#     array = corr.iloc[timepoint].to_numpy()
#     array = array[~np.isnan(array)]
#     # iterate several times with resampling: chose X time among the same list of values
#     for iteration in range(1000):
#         x = np.random.choice(array, size=len(array), replace=True)
#         # recover the mean of that new distribution
#         mean_surr.append(np.mean(x))
#     df_lower.at[0,timepoint] = np.percentile(mean_surr, 2.5)
#     df_upper.at[0,timepoint] = np.percentile(mean_surr, 97.5)
# lower =  df_lower.iloc[0].values
# upper =  df_upper.iloc[0].values
    
mean=corr.mean(axis=1)
lower = mean+corr.std(axis=1)
upper = mean-corr.std(axis=1)

panel.plot(np.arange(1,len(corr)+1), mean ,marker='o', color='black')
panel.fill_between(np.arange(1,len(corr)+1),  lower, upper, alpha=0.2, color='black')
panel.hlines(y=0,xmin=0,xmax=25,linestyles=':')
panel.set_xlabel('Trial lag')
panel.set_title('Choice outcome', fontsize=8)

# ----------------------------------------------------------------------------------------------------------------

# -------------------------------------           Autocorrelogram  for repeatition    ------------------------------------------
# file_name = 'autocorrelation_rep'
file_name = 'repeat_autocorrelation'

corr =pd.read_csv(save_path+file_name+'.csv', index_col=0)

panel=j
df_lower= pd.DataFrame()
df_upper= pd.DataFrame()

# for timepoint in range(len(corr)):
#     mean_surr = []
#     # recover the values for that specific timepoint
#     array = corr.iloc[timepoint].to_numpy()
#     array = array[~np.isnan(array)]
#     # iterate several times with resampling: chose X time among the same list of values
#     for iteration in range(1000):
#         x = np.random.choice(array, size=len(array), replace=True)
#         # recover the mean of that new distribution
#         mean_surr.append(np.mean(x))
#     df_lower.at[0,timepoint] = np.percentile(mean_surr, 2.5)
#     df_upper.at[0,timepoint] = np.percentile(mean_surr, 97.5)
# lower =  df_lower.iloc[0].values
# upper =  df_upper.iloc[0].values
    
mean=corr.mean(axis=1)
lower = mean+corr.std(axis=1)
upper = mean-corr.std(axis=1)

panel.plot(np.arange(1,len(corr)+1), mean ,marker='o', color='black')
panel.fill_between(np.arange(1,len(corr)+1),  lower, upper, alpha=0.2, color='black')
panel.hlines(y=0,xmin=0,xmax=25,linestyles=':')
panel.set_xlabel('Trial indexes')
panel.set_title('Repetitions', fontsize=8)

# ----------------------------------------------------------------------------------------------------------------

sns.despine()
plt.subplots_adjust(left=0.07,
                    bottom=0.07,
                    right=0.97,
                    top=0.97,
                    wspace=2.,
                    hspace=0.65)
# plt.tight_layout()
# plt.savefig(path+'/Fig 1_Behavior_panel_V2.svg', bbox_inches='tight',dpi=300)
plt.show()