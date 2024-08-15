# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 12:55:12 2022

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

from scipy import stats
from scipy import special
import json 
from sklearn.linear_model import LogisticRegression
from scipy.optimize import curve_fit
#Import all needed libraries
from matplotlib.lines import Line2D
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from matplotlib.backends.backend_pdf import PdfPages
from statannot import add_stat_annotation
import warnings

utilities = 'G:/Mi unidad/WORKING_MEMORY/PAPER/WM_manuscript_FIGURES/'
os.chdir(utilities)
import functions as plots

save_path = 'G:/Mi unidad/WORKING_MEMORY/PAPER/WM_manuscript_FIGURES/Fig. 3. Pharma/'
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
fig = plt.figure(figsize=(17*cm, 15*cm))
gs = gridspec.GridSpec(nrows=4, ncols=8, figure=fig)

# Create the subplots
a1 = fig.add_subplot(gs[0, 2:4])
a2 = fig.add_subplot(gs[0, 4:6])
a3 = fig.add_subplot(gs[1, 4:6])
a4 = fig.add_subplot(gs[1, 6:8])

b1 = fig.add_subplot(gs[1, 0:2])
b2 = fig.add_subplot(gs[1, 2:4])
b3 = fig.add_subplot(gs[3, 0:3])
b4 = fig.add_subplot(gs[3, 3:6])
c = fig.add_subplot(gs[3, 6:8])

e = fig.add_subplot(gs[2, 2:4])
f = fig.add_subplot(gs[2, 4:6])
g = fig.add_subplot(gs[2, 6:8])
h = fig.add_subplot(gs[2, 0:2])

fig.text(0.01, 1, 'a', fontsize=10, fontweight='bold', va='top')
fig.text(0.26, 1, 'b', fontsize=10, fontweight='bold', va='top')
fig.text(0.5, 1, 'c', fontsize=10, fontweight='bold', va='top')

fig.text(0.01, 0.75, 'd', fontsize=10, fontweight='bold', va='top')
fig.text(0.26, 0.75, 'e', fontsize=10, fontweight='bold', va='top')
fig.text(0.5, 0.75, 'f', fontsize=10, fontweight='bold', va='top')
fig.text(0.74, 0.75, 'g', fontsize=10, fontweight='bold', va='top')

fig.text(0.01, 0.26, 'h', fontsize=10, fontweight='bold', va='top')
fig.text(0.37, 0.26, 'i', fontsize=10, fontweight='bold', va='top')
fig.text(0.74, 0.26, 'j', fontsize=10, fontweight='bold', va='top')

fig.text(0.01, 0.51, 'k', fontsize=10, fontweight='bold', va='top')
fig.text(0.26, 0.51, 'l', fontsize=10, fontweight='bold', va='top')
fig.text(0.5, 0.51, 'm', fontsize=10, fontweight='bold', va='top')
fig.text(0.74, 0.51, 'n', fontsize=10, fontweight='bold', va='top')


#-----------------############################## Functions #################################-----------------------
def compute_window_centered(data, runningwindow,option):
    """
    Computes a rolling average with a length of runningwindow samples.
    """
    performance = []
    start_on=False
    for i in range(len(data)):
        if data['trial'].iloc[i] <= int(runningwindow/2):
            # Store the first index of that session for the first initial trials
            if start_on == False:
                start=i
                start_on=True
            performance.append(round(np.mean(data[option].iloc[start:i + int(runningwindow/2)]), 2))
        elif i < (len(data)-runningwindow):
            if data['trial'].iloc[i] > data['trial'].iloc[i+runningwindow]:
                # Store the last values for the end of the session
                if end == True:
                    end_value = i+runningwindow-1
                    end = False
                performance.append(round(np.mean(data[option].iloc[i:end_value]), 2))
                
            else: # Rest of the session
                start_on=False
                end = True
                performance.append(round(np.mean(data[option].iloc[i - int(runningwindow/2):i+int(runningwindow/2)]), 2))
            
        else:
            performance.append(round(np.mean(data[option].iloc[i:len(data)]), 2))
    return performance
#-----------------########################################################################-----------------------

for subject in df.subject.unique():
    print(len(df.loc[df['subject']==subject].session.unique()))

#-----------------############################## A Panel #################################-----------------------
file_name = 'drug_experiments_df'
df = pd.read_csv(save_path+file_name+'.csv',index_col=0)

panel=a1
df_results = pd.DataFrame()
df_results['accuracy'] = df.loc[(df['drug']=='Saline')|(df['drug']=='NR2B')|(df['drug']=='Rest')].groupby(['subject','drug'])['hit'].mean()
df_results.reset_index(inplace=True)
df_results['drug'] = pd.Categorical(df_results['drug'], categories=['Rest','Saline', 'NR2B'], ordered=True)

palette = sns.color_palette(['black'], len(df_results.subject.unique()))
sns.lineplot(x="drug", y="accuracy",data=df_results, hue='subject', alpha=0.8, ax=panel, marker='', linewidth=0.4, color='lightgrey', palette=palette,legend=False)
boxplot = sns.boxplot(x='drug',y="accuracy", data=df_results, width=0.5, showfliers=False, palette=['black','grey','royalblue'],ax=panel, 
                      linewidth=1, boxprops=dict(edgecolor='white'),
                     whiskerprops=dict(color='black'),
                     capprops=dict(color='black'))

# Customize the color of the mean line
mean_line_color = 'red'
for line in boxplot.lines:
    if line.get_label() == 'Mean':
        line.set_color(mean_line_color)
        line.set_linewidth(2)  # Adjust line width if needed
        
panel.set_xlabel('')
panel.set_ylabel('Accuracy')
panel.set_ylim(0.45,1)
panel.set_xlim(-0.5,2.5)
panel.hlines(y=0.5,xmin=-0.5,xmax=2.5,linestyle=':')
panel.set_xticks([0,1,2],['Rest','Saline','NR2B'])
panel.legend(loc='lower right', ncol=2)

add_stat_annotation(panel, data=df_results, x='drug', y='accuracy',
                    box_pairs=[( "Rest",'Saline'),( "Saline",'NR2B'),( "Rest",'NR2B')],
                    test='t-test_paired', text_format='star', loc='inside', line_offset_to_box=0.05, text_offset=-0.5, line_offset=0, verbose=1, fontsize=6, color='black', linewidth=0.5)


panel=a2
df_results = pd.DataFrame()
df_results['prob_repeat'] = 0.5* df.loc[df['repeat_choice_side']==1].groupby(['drug','subject'])['hit'].count()/df.loc[df.choices == -1].groupby(['drug','subject'])['hit'].count() + 0.5*df.loc[df['repeat_choice_side']==2].groupby(['drug','subject'])['hit'].count()/df.loc[df.choices == 1].groupby(['drug','subject'])['hit'].count()
df_results.reset_index(inplace=True)
df_results['drug'] = pd.Categorical(df_results['drug'], categories=['Rest','Saline', 'NR2B'], ordered=True)

palette = sns.color_palette(['black'], len(df_results.subject.unique()))
sns.lineplot(x="drug", y="prob_repeat",data=df_results, hue='subject', alpha=0.8, ax=panel, linewidth=0.5, markeredgewidth = 0.2,palette=palette,marker='',legend=False)
sns.boxplot(x='drug',y="prob_repeat", data=df_results, width=0.5, showfliers=False, palette=['black','grey','royalblue'],ax=panel, 
            linewidth=1, boxprops=dict(edgecolor='white'),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'))

panel.set_xlabel('')

panel.set_ylabel('Repeating bias')

panel.legend(loc='lower right', ncol=2)
panel.hlines(y=0.5,xmin=-0.5,xmax=2.5,linestyle=':')

add_stat_annotation(panel, data=df_results, x='drug', y='prob_repeat',
                    box_pairs=[("Saline", "Rest"),("Saline", "NR2B"),("Rest", "NR2B")],
                    test='t-test_paired', text_format='star', loc='inside', line_offset_to_box=0.05, text_offset=-0.5, line_offset=0, verbose=1, fontsize=6, linewidth=0.5)
plt.legend([],[], frameon=False)
panel.set_ylim(0.45,0.8)
panel.set_xticks([0,1,2],['Rest','Saline','NR2B'])
# ---------------------------------------------------------------------------------------------------------------

# -----------------############################## B Panel #################################-----------------------
panel=a3
df_results = pd.DataFrame()
df_results['accuracy'] = df.groupby(['subject','drug','delays'])['hit'].mean()
df_results.reset_index(inplace=True)

# sns.lineplot(x='delays',y='accuracy',data=df_results[df_results['drug']=='Rest'],markeredgewidth = 0.2,marker='o',ax=panel, color='black',ci=67)
sns.lineplot(x='delays',y='accuracy',data=df_results[df_results['drug']=='Saline'], markeredgewidth = 0.2,marker='o', linestyle='', err_style='bars', ax=panel, color='grey',ci=67)
sns.lineplot(x='delays',y='accuracy',data=df_results[df_results['drug']=='NR2B'], markeredgewidth = 0.2,marker='o', linestyle='',  err_style='bars', ax=panel, color='royalblue',ci=67)

panel.set_ylim(0.45,1)
panel.tick_params(axis='y', colors='black')
panel.yaxis.label.set_color("black")
panel.hlines(xmin=0, xmax=10, y=0.5, linestyle=':')
panel.locator_params(nbins=3)
panel.set_xlabel('Delay (s)')
panel.set_ylabel('Accuracy')

panel=a4
df_results = pd.DataFrame()
df_results['prob_repeat'] = 0.5* df.loc[df['repeat_choice_side']==1].groupby(['drug','subject','delays'])['hit'].count()/df.loc[df.choices == -1].groupby(['drug','subject','delays'])['hit'].count() + 0.5*df.loc[df['repeat_choice_side']==2].groupby(['drug','subject','delays'])['hit'].count()/df.loc[df.choices == 1].groupby(['drug','subject','delays'])['hit'].count()
df_results.reset_index(inplace=True)

# sns.lineplot(x='delays',y='prob_repeat',data=df_results[df_results['drug']=='Rest'],markeredgewidth = 0.2,marker='o',ax=panel, color='black',ci=67, label='Rest')
sns.lineplot(x='delays',y='prob_repeat',data=df_results[df_results['drug']=='Saline'], markeredgewidth = 0.2,marker='o', linestyle='',  err_style='bars', ax=panel, color='grey',ci=67, label='Saline')
sns.lineplot(x='delays',y='prob_repeat',data=df_results[df_results['drug']=='NR2B'], markeredgewidth = 0.2,marker='o',linestyle='',  err_style='bars', ax=panel, color='royalblue',ci=67, label='NR2B')

# sns.boxplot(x='delays',y="prob_repeat", data=df_results, hue='drug', width=0.5, ax=panel, showfliers=False, color='black',linewidth=1)

panel.set_ylabel('Repeating Bias')
panel.set_ylim(0.45,0.8)
panel.hlines(xmin=0, xmax=10, y=0.5, linestyle=':')
panel.tick_params(axis='y', colors='black')
panel.set_xlabel('Delay (s)')


file_name = 'drug_experiments_synthetic'
df_synthetic = pd.read_csv(save_path+file_name+'.csv',index_col=0)

panel=a3
df_results = pd.DataFrame()
df_results['accuracy'] = df_synthetic.groupby(['subject','drug','delays'])['hit'].mean()
df_results.reset_index(inplace=True)

# sns.lineplot(x='delays',y='accuracy',data=df_results[df_results['drug']=='Rest'],markeredgewidth = 0.2,marker='o',ax=panel, color='black',ci=67)
sns.lineplot(x='delays',y='accuracy',data=df_results[df_results['drug']=='Saline'],markeredgewidth = 0.2, marker='', ax=panel,  err_style=None, color='grey',ci=67)
sns.lineplot(x='delays',y='accuracy',data=df_results[df_results['drug']=='NR2B'],markeredgewidth = 0.2, marker='', ax=panel, err_style=None, color='royalblue',ci=67)

panel=a4
df_results = pd.DataFrame()
df_results['prob_repeat'] = 0.5* df_synthetic.loc[df_synthetic['repeat_choice_side']==1].groupby(['drug','subject','delays'])['hit'].count()/df_synthetic.loc[df_synthetic.choices == -1].groupby(['drug','subject','delays'])['hit'].count() + 0.5*df_synthetic.loc[df_synthetic['repeat_choice_side']==2].groupby(['drug','subject','delays'])['hit'].count()/df_synthetic.loc[df_synthetic.choices == 1].groupby(['drug','subject','delays'])['hit'].count()
df_results.reset_index(inplace=True)

# sns.lineplot(x='delays',y='prob_repeat',data=df_results[df_results['drug']=='Rest'],markeredgewidth = 0.2,marker='o',ax=panel, color='black',ci=67, label='Rest')
sns.lineplot(x='delays',y='prob_repeat',data=df_results[df_results['drug']=='Saline'],markeredgewidth = 0.2, marker='', err_style=None, ax=panel, color='grey',ci=67, label='Saline')
sns.lineplot(x='delays',y='prob_repeat',data=df_results[df_results['drug']=='NR2B'],markeredgewidth = 0.2, marker='', err_style=None, ax=panel, color='royalblue',ci=67, label='NR2B')

# ----------------------------------------------------------------------------------------------------------------

# -----------------######################## C Panel Example session trial index #################################-----------------------
threshold = 0.5

summary_df = pd.DataFrame()
animal = 'N24'
temp_df = df.loc[(df.subject==animal)&(df['session'] == 4)]
# print(temp_df.day.unique())
temp_df['drug'] ='NR2B'
temp_df['WM_roll'] = compute_window_centered(temp_df, 5,'WM')
summary_df = pd.concat([temp_df,summary_df])

panel=b4
panel.fill_between(temp_df['trial'],0 , 1, where=temp_df['WM_roll'] <= threshold,
                 facecolor='indigo', alpha=0.3)
panel.fill_between(temp_df['trial'], 0, 1,  where=temp_df['WM_roll'] >= threshold,
                 facecolor='darkgreen', alpha=0.3)
sns.lineplot(x='trial',y='WM_roll',data=temp_df, ax= panel, color='royalblue',alpha=1, label=False)
panel.set_ylabel('p(WM)')
panel.set_xlabel('Trial index')
panel.hlines(xmin=0, xmax=max(temp_df.trial.unique()), y=threshold, linestyles=':')
panel.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),ncol=2)

print('Total trials for WM: ', temp_df[(temp_df["WM_roll"]>threshold)]["stim"].count()/temp_df["stim"].count())
print('Total trials for RL: ', temp_df[(temp_df["WM_roll"]<threshold)]["stim"].count()/temp_df["stim"].count())

temp_df = df.loc[(df.subject==animal)&(df['session'] == 3)]
temp_df['drug'] = 'Saline'
temp_df['WM_roll'] = compute_window_centered(temp_df, 5,'WM')
summary_df = pd.concat([temp_df,summary_df])

panel=b3
panel.fill_between(temp_df['trial'],0 , 1, where=temp_df['WM_roll'] <= threshold,
                 facecolor='indigo', alpha=0.3)
panel.fill_between(temp_df['trial'], 0, 1,  where=temp_df['WM_roll'] >= threshold,
                 facecolor='darkgreen', alpha=0.3)
sns.lineplot(x='trial',y='WM_roll',data=temp_df, ax= panel, color='grey',alpha=1, label=False)
panel.set_ylabel('Posterior prob p(WM)')
panel.set_xlabel('Trial index')
panel.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),ncol=2)
panel.hlines(xmin=0, xmax=max(temp_df.trial.unique()), y=threshold, linestyles=':')

print('Total trials for WM: ', temp_df[(temp_df["WM_roll"]>threshold)]["stim"].count()/temp_df["stim"].count())
print('Total trials for RL: ', temp_df[(temp_df["WM_roll"]<threshold)]["stim"].count()/temp_df["stim"].count())

# ------------ Line plots of delay versus things ----------
summary_df.reset_index(inplace=True)
panel = b1
sns.lineplot(x='delays',y='hit',data=summary_df, marker='o', markeredgewidth = 0.2, hue='drug',  ci=67, label=False, ax= panel, palette=['grey','royalblue'])
panel.set_ylim(0.45,1)
panel.set_ylabel('Accuracy')
panel.hlines(y=0.5, xmin=0,xmax=10,linestyle=':')
panel.set_xlabel('Delay (s)')
panel.legend('')

panel = b2
sns.lineplot(x='delays',y='repeat',data=summary_df, ci=67, marker='o', markeredgewidth = 0.2, hue = 'drug', ax= panel, palette=['grey','royalblue'])
panel.set_ylim(0.45,0.8)
panel.set_ylabel('Repeating bias')
panel.set_xlabel('Delay (s)')
panel.legend('')

panel.hlines(y=0.5, xmin=0,xmax=10,linestyle=':')  

# ----------------------------------------------------------------------------------------------------------------
# -----------------############################## C Panel - Histogram with and without drug #################################-----------------------
df_summary = pd.read_csv(save_path+'histogram.csv', index_col=0)
panel=c
panel.bar(np.arange(0,10), df_summary.loc[df_summary['10'] == 'Saline'].iloc[:,:-1].astype(float).mean(axis=0), yerr=df_summary.loc[df_summary['10'] == 'Saline'].iloc[:,:-1].astype(float).sem(axis=0),color='grey', width=0.4)
panel.bar(np.arange(0.4,10.4), df_summary.loc[df_summary['10'] == 'NR2B'].iloc[:,:-1].astype(float).mean(axis=0), yerr=df_summary.loc[df_summary['10'] == 'NR2B'].iloc[:,:-1].astype(float).sem(axis=0) ,color='royalblue', width=0.4)
panel.set_ylabel('Probability')
panel.set_xlabel('Posterior prob. p(WM)')
panel.set_yticklabels([0.,0.2,0.4,0.6])
panel.locator_params(axis='x',nbins=6)
panel.set_xticklabels([0,0.2,0.4,0.6,0.8,1.])

# ----------------------------------------------------------------------------------------------------------------

# -----------------############################## D Panel #################################-----------------------
file_name = 'drug_experiments_fit'
full_fit = pd.read_csv(save_path+file_name+'.csv', index_col=0)
full_fit['drug'] = np.where(full_fit.drug=='0', 'Rest', full_fit.drug)
full_fit['drug'] = np.where(full_fit.drug=='drug', 'NR2B', full_fit.drug)
full_fit['drug'] = pd.Categorical(full_fit['drug'], categories=['Rest','Saline', 'NR2B'], ordered=True)

for y, panel in zip(['t11','t22','WM','alfa'], [e,f,g,h]):
    print(y)
    palette = sns.color_palette(['black'], len(full_fit.subject.unique()))
    sns.lineplot(x="drug", y=full_fit[y].astype(float),data=full_fit, hue='subject',  alpha=0.8, ax=panel, linewidth=0.5, palette=palette,marker='',legend=False, color='ligrhtgrey')
    sns.boxplot(x='drug',y=full_fit[y].astype(float), data=full_fit, showfliers=False, width=0.5, palette=['black','grey','royalblue'],ax=panel,
                linewidth=1,  boxprops=dict(edgecolor='white'),
                     whiskerprops=dict(color='black'),
                     capprops=dict(color='black'))

    panel.set_xlabel(y)
    panel.set_xlim(-0.5,2.5)
    panel.locator_params(nbins=3)
    add_stat_annotation(panel, data=full_fit, x='drug', y=y,
                        box_pairs=[("Saline", "NR2B"),("Saline", "Rest"),("Rest", "NR2B")],
                        test='t-test_paired', text_format='star', loc='inside', line_offset_to_box=0.05, text_offset=-0.5, line_offset=0, verbose=1, fontsize=6, linewidth=0.5)
    panel.set_ylim(0,1.25)
    
    plt.legend([],[], frameon=False)
    if y == 'alfa':
        panel.set_ylim(2,3)
        panel.set_xlabel('α')
    elif y == 't11':
        panel.set_xlabel('Transition p(WM🠖WM)')
    elif y == 't22':
        panel.set_xlabel('Transition p(RL🠖RL)')
    elif y == 'WM':
        panel.set_xlabel('p(WM)')


# ----------------------------------------------------------------------------------------------------------------
# Show the figure
sns.despine()
plt.subplots_adjust(left=0.07,
                    bottom=0.07,
                    right=0.97,
                    top=0.97,
                    wspace=1.8,
                    hspace=0.6)

# plt.savefig(save_path+'/Fig 3_Pharma_temporary.svg', bbox_inches='tight',dpi=300)

plt.show()