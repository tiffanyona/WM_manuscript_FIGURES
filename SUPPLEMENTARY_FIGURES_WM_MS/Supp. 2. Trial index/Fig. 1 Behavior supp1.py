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

path = 'C:/Users/Tiffany/Google Drive/WORKING_MEMORY/PAPER/Panel figures/Fig. 1 Behavior/Fig. 1 Supp. 3 Trial index'
os.chdir(path)
save_path = 'C:/Users/Tiffany/Google Drive/WORKING_MEMORY/PAPER/Panel figures/Fig. 1 Behavior/'

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
a = fig.add_subplot(gs[0, 0:4])
e = fig.add_subplot(gs[0, 4:8])

c = fig.add_subplot(gs[1, 0:4])
d = fig.add_subplot(gs[1, 4:8])

h = fig.add_subplot(gs[2, 4:6])
i = fig.add_subplot(gs[2, 6:8])
j = fig.add_subplot(gs[3, 4:6])
k = fig.add_subplot(gs[3, 6:8])


fig.text(0.01, 1, 'a', fontsize=10, fontweight='bold', va='top')
fig.text(0.5, 1, 'b', fontsize=10, fontweight='bold', va='top')
fig.text(0.01, 0.75, 'c', fontsize=10, fontweight='bold', va='top')
fig.text(0.5, 0.75, 'd', fontsize=10, fontweight='bold', va='top')

fig.text(0.01, 0.5, 'e', fontsize=10, fontweight='bold', va='top')

fig.text(0.5, 0.5, 'f', fontsize=10, fontweight='bold', va='top')
fig.text(0.75, 0.5, 'g', fontsize=10, fontweight='bold', va='top')
fig.text(0.5, 0.25, 'h', fontsize=10, fontweight='bold', va='top')
fig.text(0.75, 0.25, 'i', fontsize=10, fontweight='bold', va='top')

# -----------------############################## FUNCTIONS #################################-----------------------

def compute_window(data, runningwindow, option):
    """
    Computes a rolling average with a length of runningwindow samples.
    """
    performance = []
    end=False
    for i in range(len(data)):
        if data['trials'].iloc[i] <= runningwindow:
            # Store the first index of that session
            if end == False:
                start=i
                end=True
            performance.append(round(np.mean(data[option].iloc[start:i + 1]), 2))
            # performance.append(np.nan)
        else:
            end=False
            performance.append(round(np.mean(data[option].iloc[i - runningwindow:i]), 2))
    return performance

def trials(row):
    val = 0
    val = np.around(row['trials']/(row['total_trials']),2)
    return val

def trials_label(row):
    val = 0
    if row['T'] <0.5:
        val = 'Early'
        return val
    elif row['T'] >=0.5:
        val = 'Late'
        return val
    else:
        val = 'Mid'
        return val
    return val

# -----------------###############################################################-----------------------


#-----------------############################## A Panel #################################-----------------------
file_name = '/global_behavior_10s'
df = pd.read_csv(save_path+file_name+'.csv', index_col=0)

df['T'] = df.apply(trials, axis=1)
df['trial_label'] = df.apply(trials_label, axis=1) 
df['running_accuracy'] = compute_window(df, 20,'hit')
df['running_repeat'] = compute_window(df, 20,'repeat_choice')

df = df[df.valids==1]
df_results = pd.DataFrame()
df_results['accuracy'] = df.groupby(['subject','T'])['hit'].mean()
df_results.reset_index(inplace=True)

panel=a
sns.lineplot(x='T',y='accuracy', ax=panel, data=df_results[(df_results['T'] >=0.1)&(df_results['T'] !=1)],ci=95, color='grey', legend=False)

panel.set_ylim(0.4,1)
# ax.set_xlim(-0.5,2.5)
panel.set_xlabel('Normalized trial index')
panel.set_ylabel('Accuracy')
panel.hlines(y=0.5,xmin=0.1,xmax=1,linestyle=':')
#  ----------------------------------------------------------

#-----------------############################## C Panel #################################-----------------------

df_results = pd.DataFrame()
df_results['accuracy'] = df.groupby(['subject','T','delay_times'])['hit'].mean()
df_results.reset_index(inplace=True)

panel=c
sns.lineplot(x='T',y='accuracy', hue='delay_times',ax=panel, palette ='Purples', legend=False, data=df_results[(df_results['T'] >=0.1)&(df_results['T'] !=1)],ci=None)

panel.set_ylim(0.4,1)
panel.set_xlabel('Normalized trial index')
panel.set_ylabel('Accuracy')
panel.hlines(y=0.5,xmin=0.1,xmax=1,linestyle=':')
#  ----------------------------------------------------------

# -----------------############################## E Panel #################################-----------------------
grouping = ['subject','T','hit']
df_results = pd.DataFrame()
df_results['repeat_choice'] =(df.loc[(df['repeat_choice_side']==1)].groupby(grouping)['valids'].count()/df.loc[(df.vector_answer == 0)].groupby(grouping)['valids'].count() + df.loc[(df['repeat_choice_side']==2)].groupby(grouping)['valids'].count()/df.loc[(df.vector_answer == 1)].groupby(grouping)['valids'].count())/2
df_results.reset_index(inplace=True)

panel = e
sns.lineplot(x='T',y='repeat_choice', data=df_results[(df_results['T'] >=0.1)&(df_results['T'] !=1)],ci=95, ax=panel, color='grey')
panel.set_ylim(0.4,0.8)
# ax.set_xlim(-0.5,2.5)
panel.set_xlabel('Normalized trial index')
panel.set_ylabel('Repeating bias')
panel.hlines(y=0.5,xmin=0.1,xmax=1,linestyle=':')

# ----------------------------------------------------------------------------------------------------------------

# -----------------############################## B Panel GLM results #################################-----------------------
# panel = b

# df_results = pd.DataFrame()
# df_results['accuracy'] = df.groupby(['subject','trial_label'])['hit'].mean()
# df_results.reset_index(inplace=True)

# df_results['trial_label'] = pd.Categorical(df_results['trial_label'], categories=['Early','Late'], ordered=True)

# sns.violinplot(x='trial_label',y='accuracy', ax=panel, data=df_results, inner='box',color='grey',saturation=0.6, order=['Early','Late'],linewidth=0)
# sns.violinplot(x='trial_label',y='accuracy', ax=panel, data=df_results, inner='box',color='grey', order=['Early','Late'])

# panel.set_ylim(0.4,1)
# panel.set_xlim(-0.5,1.5)
# panel.hlines(y=0.5,xmin=-0.5,xmax=1.5,linestyle=':')
# panel.set_xticks([0,1],['Early','Late'])
# panel.set_xlabel('Trial segment')
# panel.set_ylabel('Accuracy')

# add_stat_annotation(panel, data=df_results, x='trial_label', y='accuracy',
#                     box_pairs=[( "Early",'Late')],
#                     test='t-test_paired', text_format='star', loc='outside', verbose=2)

# plt.legend(handles='', ncol=2).get_frame().set_linewidth(0.0)

# -----------------############################## B Panel GLM results #################################-----------------------
panel = d

df_results = pd.DataFrame()
df_results['accuracy'] = df.groupby(['subject','trial_label','delay_times'])['hit'].mean()
df_results.reset_index(inplace=True)

df_results['trial_label'] = pd.Categorical(df_results['trial_label'], categories=['Early','Late'], ordered=True)

sns.violinplot(x='trial_label',y='accuracy', hue='delay_times', ax=panel,  legend=False,data=df_results, inner='box',color='black',saturation=0.6, order=['Early','Late'],linewidth=0)
sns.violinplot(x='trial_label',y='accuracy', hue='delay_times', ax=panel, legend=False, data=df_results, inner='box', color='black', order=['Early','Late'])

panel.set_ylim(0.4,1)
panel.set_xlim(-0.5,1.5)
panel.hlines(y=0.5,xmin=-0.5,xmax=1.5,linestyle=':')
panel.set_xticks([0,1],['Early','Late'])
panel.set_xlabel('Trial segment')
panel.set_ylabel('Accuracy')


# add_stat_annotation(panel, data=df_results, x='trial_label', y='accuracy',
#                     box_pairs=[( "Early",'Late')],
#                     test='t-test_paired', text_format='star', loc='outside', verbose=2)

plt.legend(handles='', ncol=2).get_frame().set_linewidth(0.0)


# -----------------############################# #################################-----------------------
# panel = f

# df_results = pd.DataFrame()
# df_results['repeat_choice'] = 0.5* df.loc[(df['repeat_choice_side']==1)].groupby(['subject','trial_label'])['valids'].count()/df.loc[(df.vector_answer == 0)].groupby(['subject','trial_label'])['valids'].count() + 0.5*df.loc[(df['repeat_choice_side']==2)].groupby(['subject','trial_label'])['valids'].count()/df.loc[(df.vector_answer == 1)].groupby(['subject','trial_label'])['valids'].count()
# df_results.reset_index(inplace=True)

# df_results['trial_label'] = pd.Categorical(df_results['trial_label'], categories=['Early','Late'], ordered=True)

# sns.violinplot(x='trial_label',y='repeat_choice', ax=panel, data=df_results, inner='box',color='black',saturation=0.6, order=['Early','Late'],linewidth=0)
# sns.violinplot(x='trial_label',y='repeat_choice', ax=panel, data=df_results, inner='box',color='black', order=['Early','Late'])

# # sns.stripplot(x='trial_label',y='accuracy', data=df_results, color='darkcyan')

# panel.set_ylim(0.4,0.8)
# panel.set_xlim(-0.5,1.5)
# panel.hlines(y=0.5,xmin=-0.5,xmax=1.5,linestyle=':')
# panel.set_xticks([0,1],['Early','Late'])
# panel.set_xlabel('Trial segment')
# panel.set_ylabel('Repeating bias')

# add_stat_annotation(panel, data=df_results, x='trial_label', y='repeat_choice',
#                     box_pairs=[( "Early",'Late')],
#                     test='t-test_paired', text_format='star', loc='outside', verbose=2)

# plt.legend(handles='', ncol=2).get_frame().set_linewidth(0.0)
# ----------------------------------------------------------------------------------------------------------------

# -----------------###############################################################-----------------------

file_name = 'GLMM_final_data'
coef_matrix = pd.read_csv(save_path+file_name+'.csv',  index_col=0)
coef_matrix['const'] = 0
for regressor, panel in zip(['SL:T','SR:T','SL:D:T','SR:D:T','exp_C:T','D:exp_C:T'], [h,h,i,i,j,k]):

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
        xA = np.random.normal(1, 0.1, len(coef_matrix))
        const=1
        panel.set_xlabel('Left          Right')
        panel.set_title('Stimulus $S_t$', fontsize=7)
    elif regressor=='SR:D': 
        plot = pd.DataFrame({'SL:D': coef_matrix['SL:D'], 'SR:D': coef_matrix['SR:D']})
        sns.violinplot(data=plot, palette=[COLORLEFT, COLORRIGHT],ax=panel, width=0.5,saturation=0.6,linewidth=0)
        sns.violinplot(data=plot, palette=[COLORLEFT, COLORRIGHT],ax=panel, width=0.5,linewidth=1)
        xA = np.random.normal(1, 0.1, len(coef_matrix))
        panel.set_title('Stimulus x Delay\n $S_t·D_t$', fontsize=7)
        panel.set_xlabel('Left          Right')
        const=1
    elif regressor=='SR:T':
        plot = pd.DataFrame({'SL:T': coef_matrix['SL:T'], 'SR:T': coef_matrix['SR:T']})
        sns.violinplot(data=plot, palette=[COLORLEFT, COLORRIGHT],ax=panel, width=0.5,saturation=0.6,linewidth=0)
        sns.violinplot(data=plot, palette=[COLORLEFT, COLORRIGHT],ax=panel, width=0.5,linewidth=1)
        xA = np.random.normal(1, 0.1, len(coef_matrix))
        const=1
        panel.set_xlabel('Left            Right')
        panel.set_title('Stimulus x Delay\n x Trial $S_t·D_t·T_t$', fontsize=7)

    elif regressor=='SR:D:T':
        plot = pd.DataFrame({'SL:D:T': coef_matrix['SL:D:T'], 'SR:D:T': coef_matrix['SR:D:T']})
        sns.violinplot(data=plot, palette=[COLORLEFT, COLORRIGHT],ax=panel, width=0.5,saturation=0.6,linewidth=0)
        sns.violinplot(data=plot, palette=[COLORLEFT, COLORRIGHT],ax=panel, width=0.5,linewidth=1)
        panel.set_title('Stimulus x Trial $S_t·T_t$', fontsize=7)
        xA = np.random.normal(1, 0.1, len(coef_matrix))
        panel.set_xlabel('Left            Right')
        const=1
    elif regressor == 'SL' or regressor == 'SL:D'or regressor == 'SL:D:T'or regressor == 'SL:T':
        main= COLORLEFT
        light = 'grey'
        xA = np.random.normal(0, 0.1, len(coef_matrix))
        const=0
        pass
    else:
        const=0
        xA = np.random.normal(0, 0.15, len(coef_matrix))
        sns.violinplot(x=coef_matrix['const'].astype(float),y=coef_matrix[regressor].astype(float), data=coef_matrix, width=0.75, color=main,ax=panel, saturation=0.6,linewidth=0)
        sns.violinplot(x=coef_matrix['const'].astype(float),y=coef_matrix[regressor].astype(float), data=coef_matrix, width=0.75, color=main,ax=panel, legend=False,linewidth=1)
        main='indigo'
        if regressor == 'exp_C':
            panel.set_title('Prev. Choices x Trial\n $C_{t-k}·T_t$', fontsize=7)
        if regressor == 'D:exp_C':
            panel.set_title('Prev. Choices x Delay x Trial\n$C_{t-k}·D_t·T_t$', fontsize=7)
        if regressor == 'exp_C:T':
            panel.set_title('Previous Choices $C_{t-k}$', fontsize=7)
        if regressor == 'D:exp_C:T':
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
        panel.set_ylim(-0.1,0.1)
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

# ----------------------------------------------------------------------------------------------------------------

sns.despine()
plt.subplots_adjust(left=0.07,
                    bottom=0.07,
                    right=0.97,
                    top=0.97,
                    wspace=2.1,
                    hspace=0.7)
# plt.tight_layout()
# plt.savefig(path+'/Fig 1_supp1.svg', bbox_inches='tight',dpi=300)
plt.show()