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
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
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

utilities = 'G:/Mi unidad/WORKING_MEMORY/PAPER/WM_manuscript_FIGURES/'
os.chdir(utilities)
import functions as plots

save_path = 'G:/Mi unidad/WORKING_MEMORY/PAPER/WM_manuscript_FIGURES/Fig. 2 Model/'
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
fig = plt.figure(figsize=(17*cm, 21*cm))
gs = gridspec.GridSpec(nrows=5, ncols=8, figure=fig)

# Create the subplots
a = fig.add_subplot(gs[2, 0:3])
a2 = fig.add_subplot(gs[2, 3:6])

x = fig.add_subplot(gs[1, 4:8])
# x_0 = fig.add_subplot(gs[0, 6:8])
x2 = fig.add_subplot(gs[2, 6:8])

b = fig.add_subplot(gs[3, 0:2])
b2 = fig.add_subplot(gs[3, 2:4])

c = fig.add_subplot(gs[0, 4:6])
c2 = fig.add_subplot(gs[0, 6:8])

d1 = fig.add_subplot(gs[3, 4:6])
d2 = fig.add_subplot(gs[4, 0:1])
d3 = fig.add_subplot(gs[4, 1:2])
d4 = fig.add_subplot(gs[3, 6:8])
d5 = fig.add_subplot(gs[4, 2:3])
d6 = fig.add_subplot(gs[4, 3:4])

# g1 = fig.add_subplot(gs[5, 2:4])
g2 = fig.add_subplot(gs[4, 4:6])
g3 = fig.add_subplot(gs[4, 6:8])

fig.text(0.01, 0.99, 'a', fontsize=10, fontweight='bold', va='top')
fig.text(0.5, 0.99, 'b', fontsize=10, fontweight='bold', va='top')
fig.text(0.01, 0.83, 'c', fontsize=10, fontweight='bold', va='top')
fig.text(0.36, 0.83, 'd', fontsize=10, fontweight='bold', va='top')
fig.text(0.75, 0.83, 'e', fontsize=10, fontweight='bold', va='top')

fig.text(0.01, 0.67, 'f', fontsize=10, fontweight='bold', va='top')
fig.text(0.26, 0.67, 'g', fontsize=10, fontweight='bold', va='top')
fig.text(0.5, 0.67, 'h', fontsize=10, fontweight='bold', va='top')
fig.text(0.75, 0.67, 'i', fontsize=10, fontweight='bold', va='top')

fig.text(0.01, 0.5, 'j', fontsize=10, fontweight='bold', va='top')
fig.text(0.26, 0.5, 'k', fontsize=10, fontweight='bold', va='top')
fig.text(0.4, 0.5, 'l', fontsize=10, fontweight='bold', va='top')
fig.text(0.5, 0.5, 'm', fontsize=10, fontweight='bold', va='top')
fig.text(0.77, 0.5, 'n', fontsize=10, fontweight='bold', va='top')
fig.text(0.89, 0.5, 'o', fontsize=10, fontweight='bold', va='top')

fig.text(0.01, 0.35, 'p', fontsize=10, fontweight='bold', va='top')
fig.text(0.26, 0.35, 'q', fontsize=10, fontweight='bold', va='top')
fig.text(0.5, 0.35, 'r', fontsize=10, fontweight='bold', va='top')

fig.text(0.01, 0.19, 's', fontsize=10, fontweight='bold', va='top')
fig.text(0.27, 0.19, 't', fontsize=10, fontweight='bold', va='top')
fig.text(0.5, 0.19, 'u', fontsize=10, fontweight='bold', va='top')


# -----------------############################## FUNCTIONS #################################-----------------------
def figureplot(new_df_real,new_df,panel):
    Left = 'teal'
    Right = '#FF8D3F'

    # --------------------------------------
    df_results =pd.DataFrame()
    df_results['accuracy'] = new_df_real.groupby(['delays','session','stim'])['hit'].mean()
    df_results.reset_index(inplace=True)      

    sns.lineplot(x='delays',y='accuracy',data=df_results, ci=67, markeredgewidth = 0.2, ax=panel,marker='o',color='black', linestyle = '', err_style="bars")
    sns.lineplot(x='delays',y='accuracy',hue='stim',data=df_results, markeredgewidth = 0.2, ax=panel, marker='o', palette=[Left,Right], linestyle = '', err_style="bars",legend=False)

    df_results =pd.DataFrame()
    df_results['accuracy'] = new_df.groupby(['delays','session'])['hit'].mean()
    df_results.reset_index(inplace=True)   
    sns.lineplot(x='delays',y='accuracy',data=df_results,color='black', ax=panel,  markersize=3)

    df_results =pd.DataFrame()
    df_results['accuracy'] = new_df.groupby(['delays','stim','session'])['hit'].mean()
    df_results.reset_index(inplace=True)   
    sns.lineplot(x='delays',y='accuracy',hue='stim',markeredgewidth = 0.2, data=df_results,  markersize=3, ax=panel, palette=[Left,Right], legend=False)

    panel.set_ylim(0.4,1)
    panel.hlines(xmin=0,xmax=10, y=0.5, linestyles = ':')
    panel.set_xlabel("Delay (s)")
    panel.set_ylabel(" Accuracy")
    panel.locator_params(nbins=3)


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
#____________________________________________________________________________________________________________________

# -----------------############################## B Panel #################################-----------------------
threshold = 0.5
groupings=['subject','delays','state']

file = 'all_data_HMM'
df = pd.read_csv(save_path+ file+'.csv')
df['WM_roll'] = compute_window_centered(df, 3,'WM')
df['state'] = np.where(df.WM_roll > threshold, 1, 0)

file = 'all_data_HMM_model'
df_model = pd.read_csv(save_path+ file+'.csv')

df_model = df_model.loc[df_model.animal_delay == 10]
df = df.loc[df.animal_delay == 10]

# ------------ Real data
df_results = pd.DataFrame()
df_results['hit'] = df.groupby(groupings)['hit'].mean()
df_results['repeat_choice'] = 0.5* df.loc[(df['repeat_choice_side']==1)].groupby(groupings)['choices'].count()/df.loc[df.choices == -1].groupby(groupings)['choices'].count() + 0.5*df.loc[(df['repeat_choice_side']==2)].groupby(groupings)['choices'].count()/df.loc[(df.choices == 1)].groupby(groupings)['choices'].count()
df_results.reset_index(inplace=True)
sns.lineplot(x='delays',y='hit',data=df_results, marker='o', hue='state', markeredgewidth = 0.2, palette=['indigo','darkgreen'], ax=b,  ci=95,linestyle='', legend=False,err_style='bars')
sns.lineplot(x='delays',y='repeat_choice',data=df_results, hue='state',  markeredgewidth = 0.2,palette=['indigo','darkgreen'], marker='o', color='black', ci=95,legend=False, linestyle='', ax=b2, err_style='bars')

groupings=['subject','delays']
df_results = pd.DataFrame()
df_results['repeat_choice'] = 0.5* df.loc[(df['repeat_choice_side']==1)].groupby(groupings)['choices'].count()/df.loc[df.choices == -1].groupby(groupings)['choices'].count() + 0.5*df.loc[(df['repeat_choice_side']==2)].groupby(groupings)['choices'].count()/df.loc[(df.choices == 1)].groupby(groupings)['choices'].count()
df_results['hit'] = df.groupby(groupings)['hit'].mean()
df_results.reset_index(inplace=True)
sns.lineplot(x='delays',y='hit',data=df_results, marker='o', ax=b, markeredgewidth = 0.2,color='black', ci=95,linestyle='', legend=False,err_style='bars')
sns.lineplot(x='delays',y='repeat_choice',data=df_results, marker='o', markeredgewidth = 0.2,color='black', ci=95,legend=False, linestyle='', ax=b2, err_style='bars')

# ------------ HMM model
groupings=['subject','delays','state']
df_results = pd.DataFrame()
df_results['hit'] = df_model.groupby(groupings)['hit'].mean()
df_results['repeat_choice'] = 0.5* df_model.loc[(df_model['repeat_choice_side']==1)].groupby(groupings)['choices'].count()/df_model.loc[df_model.choices == -1].groupby(groupings)['choices'].count() + 0.5*df_model.loc[(df_model['repeat_choice_side']==2)].groupby(groupings)['choices'].count()/df_model.loc[(df_model.choices == 1)].groupby(groupings)['choices'].count()
df_results.reset_index(inplace=True)
sns.lineplot(x='delays',y='hit',data=df_results, hue='state', marker='', ax=b, palette=['indigo','darkgreen'],  ci=95,legend=False,err_style=None)
sns.lineplot(x='delays',y='repeat_choice', hue='state', data=df_results, palette=['indigo','darkgreen'], marker='',  ci=95,ax=b2, err_style=None)
b.set_ylim(0.4,1)
b.hlines(xmin=0, xmax=10, y=0.5, linestyles=':')
b.set_ylabel('Accuracy')
b.set_xlabel('Delay (s)')
b.locator_params(nbins=3)

groupings=['subject','delays']
df_results = pd.DataFrame()
df_results['hit'] = df_model.groupby(groupings)['hit'].mean()
df_results['repeat_choice'] = 0.5* df_model.loc[(df_model['repeat_choice_side']==1)].groupby(groupings)['choices'].count()/df_model.loc[df_model.choices == -1].groupby(groupings)['choices'].count() + 0.5*df_model.loc[(df_model['repeat_choice_side']==2)].groupby(groupings)['choices'].count()/df_model.loc[(df_model.choices == 1)].groupby(groupings)['choices'].count()
df_results.reset_index(inplace=True)
sns.lineplot(x='delays',y='hit',data=df_results, marker='', ax=b, color='black',  ci=95,legend=False,err_style=None)
sns.lineplot(x='delays',y='repeat_choice', data=df_results, color='black', marker='',  ci=95,ax=b2, err_style=None)
b2.set_ylim(0.4,1)
b2.set_ylabel('Repeating bias')
b2.set_xlabel('Delay (s)')
b2.hlines(y=0.5,xmin=0,xmax=10,linestyle=':')
b2.locator_params(nbins=3)

legend_elements = [Line2D([0], [0], color='black', label='All trials'),
                   Line2D([0], [0], color='indigo', label='RL trials'),
                   Line2D([0], [0],  color = 'darkgreen', label='WM trials')]
b2.legend(handles=legend_elements, ncol=1,borderaxespad=0, fontsize=6).get_frame().set_linewidth(0.0)

# ----------------------------------------------------------------------------------------------------------------
# -----------------############################## B Panel - BIC comparison #################################-----------------------
panel=x
# file_name = 'fit_DW_HMM_cross_V5_delay10'
file_name = 'pertrialLL'
full_fit = pd.read_csv(save_path+file_name+'.csv', index_col=0)
color_list = ['black','darkgrey', 'blue','lightgrey','grey']
xA = np.random.normal(0, 0.08, len(full_fit))
sns.stripplot(x='model',y='substracted', data=full_fit, jitter=0.3,size=2, order=["all",'12','9','10','11'], palette = color_list, edgecolor='white', linewidth=0.1, ax=panel)
sns.violinplot(x='model',y='substracted', data=full_fit, saturation=0.7, order=["all",'12','9','10','11'], palette = 'copper',linewidth=0, width = 0.5, ax=panel)
sns.violinplot(x='model',y='substracted', data=full_fit, order=["all",'12','9','10','11'], palette = 'copper',linewidth=1.5, width = 0.5, fliersize=0, ax=panel )

panel.hlines(y=0, xmin=-0.5, xmax=4.5, linestyle=':')
panel.set_xlabel('')
panel.set_ylabel('LL difference (bits/trial)')
# panel.set_ylim(-200,700)
labels = ['DW classic','DW initial\nrepeating','DW delay\n repeating','DW both\n repeating']
panel.set_xticklabels(labels)
# panel.set_yticklabels(['HMM','DW $x_0_r$','DW $mb_r$','DW both','DW'])
# panel.text(x=-1.48,y=1200, s='> 1000 -', fontsize=6)

add_stat_annotation(panel, data=full_fit, x='model', y='substracted',
                    box_pairs=[("all", "11"),("all", "9"),("all", "10"),("all", "12")],
                    test='t-test_paired', text_format='star', loc='outside', verbose=2)

# 9 v.s. all: t-test paired samples with Bonferroni correction, P_val=8.908e-07 stat=6.371e+00
# 12 v.s. all: t-test paired samples with Bonferroni correction, P_val=2.278e-13 stat=1.184e+01
# 11 v.s. all: t-test paired samples with Bonferroni correction, P_val=3.244e-05 stat=5.201e+00
# 10 v.s. all: t-test paired samples with Bonferroni correction, P_val=4.455e-10 stat=8.947e+00
# --------------------------------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------------------------------------
# -----------------############################## B Panel - BIC comparison #################################-----------------------
# panel=x_0
# # file_name = 'fit_DW_HMM_new'
# file_name = 'fit_DW_HMM_cross'

# full_fit = pd.read_csv(save_path+file_name+'.csv', index_col=0)
# color_list = ['blue','lightgrey','grey']
# full_fit = full_fit.loc[(full_fit.model == '9')|(full_fit.model == '10')|(full_fit.model == '11')]
# xA = np.random.normal(0.05, 0.1, len(full_fit))
# sns.stripplot(x='model',y='norm', data=full_fit, jitter=0.3,size=2, order=['9','10','11'], palette = color_list, edgecolor='white', linewidth=0.1, ax=panel)
# # sns.violinplot(x='model',y='norm', data=full_fit, saturation=0.7, order=['all','9','10','11','12'], palette = color_list,linewidth=0, ax=panel)
# sns.violinplot(x='model',y='norm', data=full_fit, legend=False, order=['9','10','11'], palette = color_list,linewidth=1.5, ax=panel)

# panel.hlines(y=1, xmin=-0.5, xmax=2.5, linestyle=':')
# panel.set_xlabel('')
# panel.set_ylabel('BIC difference')

# panel.set_ylim(-250,600)
# labels = ['DW with\n initial \nrepeating','DW with\n repeating \nin Delay','DW\n with both']
# panel.set_xticklabels(labels)
# # panel.set_yticklabels(['HMM','DW $x_0_r$','DW $mb_r$','DW both','DW'])
# # panel.text(x=-1.48,y=1200, s='> 1000 -', fontsize=6)

# # add_stat_annotation(ax, data=full_fit, x='model', y='BIC',
# #                     box_pairs=[("all", "11"),("11", "9"),("9", "10"),("9", "12")],
# #                     test='t-test_paired', text_format='star', loc='outside', verbose=2)
# # --------------------------------------------------------------------------------------------------------------------------------------------

# # -----------------############################## B Panel - BIC comparison #################################-----------------------
# panel=x
# file_name = 'fit_DW_HMM_cross'
# full_fit = pd.read_csv(save_path+file_name+'.csv', index_col=0)
# color_list = ['black','purple']
# full_fit = full_fit.loc[(full_fit.model == 'all')|(full_fit.model == '12')]

# xA = np.random.normal(0, 0.1, len(full_fit))
# sns.stripplot(x='model',y='norm', data=full_fit, jitter=0.3,size=2, order=['all','12'], palette = color_list, edgecolor='white', linewidth=0.1, ax=panel)
# # sns.violinplot(x='model',y='norm', data=full_fit, saturation=0.7, order=['all','9','10','11','12'], palette = color_list,linewidth=0, ax=panel)
# sns.violinplot(x='model',y='norm', data=full_fit, legend=False, order=['all','12'], palette = color_list,linewidth=1.5, ax=panel)

# panel.hlines(y=1, xmin=-0.5, xmax=1.5, linestyle=':')
# panel.set_xlabel('')
# panel.set_ylabel('BIC difference')

# panel.set_ylim(-250,1300)
# labels = ['HMM','classic DW']
# panel.set_xticklabels(labels)
# # panel.set_yticklabels(['HMM','DW $x_0_r$','DW $mb_r$','DW both','DW'])
# panel.text(x=-1.48,y=1200, s='> 1000 -', fontsize=6)

# add_stat_annotation(ax, data=full_fit, x='model', y='BIC',
#                     box_pairs=[("all", "11"),("11", "9"),("9", "10"),("9", "12")],
#                     test='t-test_paired', text_format='star', loc='outside', verbose=2)
# --------------------------------------------------------------------------------------------------------------------------------------------

    
# -----------------############################## C Panel - Model Comparison with long delays #################################-----------------------
# ------------ Real data
groupings=['subject','delays']
color_list = ['black','purple','blue','lightgrey','grey']
# file_name = 'fit_DW_HMM_new'
# full_fit = pd.read_csv(save_path+file_name+'.csv', index_col=0)

df_results = pd.DataFrame()
df_results['hit'] = df.groupby(groupings)['hit'].mean()
df_results['repeat_choice'] = 0.5* df.loc[(df['repeat_choice_side']==1)].groupby(groupings)['choices'].count()/df.loc[df.choices == -1].groupby(groupings)['choices'].count() + 0.5*df.loc[(df['repeat_choice_side']==2)].groupby(groupings)['choices'].count()/df.loc[(df.choices == 1)].groupby(groupings)['choices'].count()
df_results.reset_index(inplace=True)
sns.lineplot(x='delays',y='hit',data=df_results, marker='o',  markeredgewidth = 0.1, ax=c, color='black', ci=95,linestyle='', legend=False,err_style='bars')
sns.lineplot(x='delays',y='repeat_choice',data=df_results, marker='o',  markeredgewidth = 0.1,color='black', ci=95,legend=False, linestyle='', ax=c2, err_style='bars')

color_list = color_list[1:]
for model, color, label in zip(['all','9','10','11','12'],['black','darkblue', 'darkgrey','crimson', 'purple'],['HMM','DW Beta^X0_rep','DW Beta^mub_rep','DW both','DW']):
# for model, color, label in zip(['9','10','11','12'],color_list,['DW Beta^X0_rep','DW Beta^mub_rep','DW both','DW']):
    # ------------ Alternative model
    df_results = pd.read_csv(save_path+'results alternative DW models_'+model+'_cross_V4.csv')
    sns.lineplot(x='delays',y='hit',data=df_results, marker='', ax=c, color=color, ci=95,legend=False,err_style=None)
    sns.lineplot(x='delays',y='repeat_choice', data=df_results, color=color, marker='',  ci=95,ax=c2, err_style=None)

    c.set_ylim(0.45,1)
    c.hlines(xmin=0, xmax=30, y=0.5, linestyles=':')
    c.set_ylabel('Accuracy')
    c.set_xlabel('Delay (s)')

    c2.set_ylim(0.45,0.8)
    c2.set_ylabel('Repeating bias')
    c2.set_xlabel('Delay (s)')
    c2.hlines(y=0.5,xmin=0,xmax=30,linestyle=':')

# ------------- Legend for repetition and alternation plot
legend_elements = [Line2D([0], [0], color=color_list[3], label='DW classic'),
                    Line2D([0], [0], color=color_list[1], label='DW delay repeating'),
                    Line2D([0], [0],  color = color_list[0], label='DW  initial repeating'),
                    Line2D([0], [0],  color=color_list[2],label='DW both repeating')]

c2.legend(handles=legend_elements, ncol=1, fontsize=6,borderaxespad=0).get_frame().set_linewidth(0.0)

c.locator_params(nbins=3)
c2.locator_params(nbins=3)

# ----------------------------------------------------------------------------------------------------------------
# -----------------############################## D Panel - Parameter values for HMM fiting #################################-----------------------
file_name = 'fit_HMM_selected_final'
full_fit = pd.read_csv(save_path+file_name+'.csv', index_col=0)
full_fit = full_fit.loc[full_fit.delay ==10]
full_fit['alfa'] = full_fit['c2']/2

full_fit['const'] = 1
for regressor, panel, color in zip(['P_L', 'P_R','alfa', 'mu_b','WM','RL', 'beta_w', 'beta_bias'],[d1, d1, d2, d3, d4, d4, d5, d6], ['darkgreen','darkgreen','darkgreen','darkgreen','grey','grey','indigo','indigo']):
    xA = np.random.normal(0, 0.1, len(full_fit))
    if regressor == 'P_R':
        plot = pd.DataFrame({'PR': full_fit['P_R'], 'PL': full_fit['P_L']})
        sns.violinplot(data=plot, palette=[color, color],ax=panel, width=0.5)
        sns.violinplot(data=plot, palette=[color, color],ax=panel, width=0.5, linewidth=0)
        xA = np.random.normal(1, 0.1, len(full_fit))
        panel.set_xlabel('$P_{L|L}$             $P_{R|R}$')
    elif regressor == 'RL':
        plot = pd.DataFrame({'p(WM)': full_fit['WM'], 'p(RL)': full_fit['RL']})
        sns.violinplot(data=plot, palette=[color, color],ax=panel, width=0.5)
        sns.violinplot(data=plot, palette=[color, color],ax=panel, width=0.5, linewidth=0)
        xA = np.random.normal(1, 0.1, len(full_fit))  
        panel.set_xlabel('p(WM)          p(RL)')
    elif regressor == 'P_L' or regressor == 'WM':
        pass
    else:
        xA = np.random.normal(0, 0.20, len(full_fit))
        sns.violinplot(x=full_fit['const'].astype(float),y=full_fit[regressor].astype(float), data=full_fit, width=1,color=color,ax=panel, saturation=0.6,linewidth=0)
        sns.violinplot(x=full_fit['const'].astype(float),y=full_fit[regressor].astype(float), data=full_fit, width=1, color=color,ax=panel, legend=False,linewidth=1)

    sns.scatterplot(x=xA,y=regressor,data=full_fit,ax=panel,alpha=0.7,style='delay', color=color,legend=False, size=2)
    panel.set_ylabel('')
    panel.set_xticks([])
    panel.set_xlim(-0.7,0.7)

    if regressor == 'alfa':
        panel.set_xlabel('α')
        panel.set_ylim(1.5,3.1)
    elif regressor == 'mu_b':
        panel.set_xlabel('$m_b$')
    elif regressor == 'beta_w':
        panel.set_xlabel('$β_a$')
    elif regressor == 'beta_bias':
        panel.set_xlabel('$β_{bias}$')
    elif regressor == 'P_L' or regressor == 'P_R' or regressor == 'WM' or regressor == 'RL':
        panel.set_ylim(-0.1,1.1)
        panel.set_xlim(-0.5,1.5)

    panel.hlines(y=0,xmin=-1,xmax=1.5,linestyle=':')
    panel.locator_params(axis='y', nbins=5)
    y_min, y_max = panel.get_ylim()  
    panel.locator_params(nbins=3)

# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------
# -----------------############################## X2 Panel -  Histogram #################################---------------------
df_summary = pd.read_csv(save_path+'histogram_all.csv', index_col=0)

panel = x2
patches = panel.bar(np.arange(0,10), df_summary.astype(float).mean(axis=0), yerr=df_summary.sem(axis=0),color='grey', alpha=0.3)
for i in range(5,10):
    patches[i].set_facecolor('darkgreen')
for i in range(0,5):    
    patches[i].set_facecolor('indigo')

panel.set_ylabel('Probability')
panel.set_xlabel('Posterior p(WM)')
panel.vlines(x=threshold*10-0.5, ymax=max(df_summary.astype(float).mean(axis=0)),ymin=0, linestyle=':')
panel.locator_params(axis='x',nbins=18)
panel.set_xticklabels([0.0,'',0.2,'',0.4,'',0.6,'',0.8,'',1])

# -----------------############################## X Panel -  Example poor man's and p(WM) #################################---------------------

# animal = 'E12_10'
# session = 19

# animal = 'N24_10'
# session = 17

# animal = 'C38'
# session = 11
file = 'final data from HMM (delays 10)'
df = pd.read_csv(save_path+ file+'.csv')
df['WM_roll'] = compute_window_centered(df, 3,'WM')
df['state'] = np.where(df.WM_roll > threshold, 1, 0)

animal = 'C38'
session = 8

temp_df = df.loc[(df['session'] == session)&(df.subject==animal)]
temp_df['accuracy'] = compute_window_centered(temp_df, 20,'hit')
temp_df['repeat_choice'] = compute_window_centered(temp_df, 20,'repeat')
temp_df['WM_roll'] = compute_window_centered(temp_df, 3,'WM')
print(temp_df.day.unique())

panel=a2
panel.fill_between(temp_df['trial'],0 , 1, where=temp_df['WM'] <= threshold,
                 facecolor='indigo', alpha=0.3)
panel.fill_between(temp_df['trial'], 0, 1,  where=temp_df['WM'] >= threshold,
                 facecolor='darkgreen', alpha=0.3)
sns.lineplot(x='trial',y='accuracy',data=temp_df, ax= panel, color='black', label='Accuracy')
sns.lineplot(x='trial',y='repeat_choice',data=temp_df, ax= panel, color='darkgrey', label= 'Repeating bias')
panel.set_ylabel('Running percentage')
panel.set_xlim(0,max(temp_df.trial.unique())-3)
panel.set_xlabel('Trial index')
panel.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),ncol=2)
panel.hlines(xmin=0, xmax=len(temp_df),y=0.5, linestyle=':')
panel.set_title('Session C38-2021-07-03')

panel=a
panel.fill_between(temp_df['trial'],0 , 1, where=temp_df['WM'] <= threshold,
                 facecolor='indigo', alpha=0.3)
panel.fill_between(temp_df['trial'], 0, 1,  where=temp_df['WM'] >= threshold,
                 facecolor='darkgreen', alpha=0.3)
sns.lineplot(x='trial',y='WM_roll',data=temp_df, ax= panel, color='black', label='pWM')
panel.set_ylabel('p(WM)')
panel.set_xlim(0,max(temp_df.trial.unique())-3)
panel.set_xlabel('Trial index')
panel.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),ncol=2)
panel.hlines(xmin=0, xmax=len(temp_df),y=0.5, linestyle=':')

# panel.arrow(x=1,y=0.5,dx=0,dy=0.1, width=0.1, color='darkgreen',head_width=0.5)
# panel.arrow(x=1,y=0.5,dx=0,dy=-0.1, width=0.1, color='indigo',head_width=0.5)

# print('Total trials for WM: ', temp_df[(temp_df["WM"]>0.5)]["stim"].count()/temp_df["stim"].count())
# print('Total trials for RL: ', temp_df[(temp_df["RL"]>0.5)]["stim"].count()/temp_df["stim"].count())

np.mean(df.loc[(df["WM"]>0.6)].groupby('subject')["stim"].count()/df.groupby('subject')["stim"].count())
np.std(df.loc[(df["WM"]>0.6)].groupby('subject')["stim"].count()/df.groupby('subject')["stim"].count())

np.mean(df.loc[(df["RL"]>0.4)].groupby('subject')["stim"].count()/df.groupby('subject')["stim"].count())
np.std(df.loc[(df["RL"]>0.4)].groupby('subject')["stim"].count()/df.groupby('subject')["stim"].count())

# ----------------------------------------------------------------------------------------------------------------

# -----------------############################## G Panel - Individual animal examples #################################-----------------------

# animal='N27_10'
# new_df_real = pd.read_csv(save_path+animal+'_data.csv')
# new_df =  pd.read_csv(save_path+animal+'_model.csv')
# figureplot(new_df_real,new_df,g1)

# animal='C10b'
animal='N11_10'
new_df_real = pd.read_csv(save_path+animal+'_data.csv', index_col = 0)
new_df =  pd.read_csv(save_path+animal+'_model.csv', index_col = 0)
figureplot(new_df_real,new_df,g2)

animal='C37_10'
new_df_real = pd.read_csv(save_path+animal+'_data.csv', index_col = 0)
new_df =  pd.read_csv(save_path+animal+'_model.csv', index_col = 0)
figureplot(new_df_real,new_df,g3)

# Show the figure
sns.despine()
plt.subplots_adjust(left=0.07,
                    bottom=0.07,
                    right=0.97,
                    top=0.97,
                    wspace=1.5,
                    hspace=0.5)

plt.savefig(save_path+'/Fig 2_panel_V6.1.svg', bbox_inches='tight',dpi=300)
plt.show()