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

save_path = r'G:\Mi unidad\WORKING_MEMORY\PAPER\WM_manuscript_FIGURES\SUPPLEMENTARY_FIGURES_WM_MS\Supp. 3. Extra results from model'
analysis_path = 'G:/Mi unidad/WORKING_MEMORY/PAPER/ANALYSIS_figures/'
path = 'G:/Mi unidad/WORKING_MEMORY/PAPER/WM_manuscript_FIGURES/Fig. 2 Model/'

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
fig = plt.figure(figsize=(17*cm, 10*cm))
gs = gridspec.GridSpec(nrows=3, ncols=8, figure=fig)

a = fig.add_subplot(gs[0, 0:3])
b = fig.add_subplot(gs[0, 3:5])
c = fig.add_subplot(gs[0, 6:8])

d1 = fig.add_subplot(gs[1, 0:2])
d2 = fig.add_subplot(gs[1, 2:4])
e1 = fig.add_subplot(gs[1, 4:6])
e2 = fig.add_subplot(gs[1, 6:8])

f = fig.add_subplot(gs[2, 0:2])
g = fig.add_subplot(gs[2, 2:4])

fig.text(0.01, 0.99, 'a', fontsize=10, fontweight='bold', va='top')
fig.text(0.37, 0.99, 'b', fontsize=10, fontweight='bold', va='top')
fig.text(0.75, 0.99, 'c', fontsize=10, fontweight='bold', va='top')

fig.text(0.01, 0.66, 'd', fontsize=10, fontweight='bold', va='top')
fig.text(0.5, 0.66, 'e', fontsize=10, fontweight='bold', va='top')

fig.text(0.01, 0.37, 'f', fontsize=10, fontweight='bold', va='top')
fig.text(0.25, 0.37, 'g', fontsize=10, fontweight='bold', va='top')

# -----------------############################## FUNCTIONS #################################-----------------------
def exponential_decay(x, A, tau, c):
    return A * np.exp(-x/tau) + c

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


# ----------------------------------------------------------------------------------------------------------------
# -----------------############################## D Panel - Parameter values for HMM fiting #################################-----------------------
file_name = 'fit_HMM_selected_final'
full_fit = pd.read_csv(analysis_path+file_name+'.csv', index_col=0)

full_fit['const'] = 1
full_fit = full_fit.loc[full_fit.delay == 10]

for regressor, panel, color in zip(['pi', 't11','t22'],[a, a, a], ['darkgreen','grey','grey']):
    if regressor == 'pi':
        xA = np.random.normal(0, 0.1, len(full_fit))  
        sns.scatterplot(x=xA,y=regressor,data=full_fit, ax=panel, color=color, legend=False, alpha=0.7, size=1)

    elif regressor == 't11':
        xA = np.random.normal(1, 0.1, len(full_fit))  
        sns.scatterplot(x=xA,y=regressor,data=full_fit, ax=panel, color=color, legend=False,alpha=0.7, size=1)

    elif regressor == 't22':
        xA = np.random.normal(2, 0.1, len(full_fit))  
        sns.scatterplot(x=xA,y=regressor,data=full_fit, ax=panel, color=color, legend=False, alpha=0.7, size=1)
    
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

    panel.hlines(y=0,xmin=-1,xmax=2.5,linestyle=':')
    panel.locator_params(axis='y', nbins=5)
    y_min, y_max = panel.get_ylim()  
    panel.locator_params(nbins=3)

panel = a
plot = pd.DataFrame({'pi': full_fit['pi'], 't11': full_fit['t11'], 't22': full_fit['t22']})
sns.violinplot(data=plot, palette=['darkgreen','grey','grey'],ax=panel, width=1)
sns.violinplot(data=plot, palette=['darkgreen','grey','grey'],ax=panel, width=1, linewidth=0)
panel.set_xlabel(r'$\pi         t11         t22')

# ----------------------------------------------------------------------------------------------------------------
# -----------------############################## E Panel - X-Y #################################-----------------------
panel=b

df_results = pd.read_csv(path+'\X-Y.csv')
cmap = sns.diverging_palette(15,250, s=100, l=60, n=len(df_results.loc[df_results.streak != 0].streak.unique()), center="dark")

sns.lineplot(x='delays',y='accuracy_model',hue='streak', data=df_results, ci=67, legend=False,ax=panel,palette=cmap)
sns.lineplot(x='delays',y='accuracy_data',hue='streak', data=df_results, ci=67, linestyle='',ax=panel,  markeredgewidth = 0.2, marker='o', err_style="bars",legend=False, palette=cmap)

panel.hlines(xmin=0,xmax=10, y=0.5, linestyles = ':')
panel.set_ylim(0.4,1)
panel.set_xlabel("Delay (s)")
panel.set_ylabel(" Accuracy")
panel.locator_params(nbins=3)

# ------------- Legend for repetition and alternation plot
legend_elements = [Line2D([0], [0], color=cmap[-1], label='XXX'),
                    Line2D([0], [0], color=cmap[-2], label='XX'),
                    Line2D([0], [0],  color = cmap[-3], label='X'),
                    Line2D([0], [0],  color=cmap[2],label='Y'),
                    Line2D([0], [0], color=cmap[1], label='YY'),
                    Line2D([0], [0], color=cmap[0], label='YYY')]
panel.legend(handles=legend_elements, ncol=1, fontsize=6, bbox_to_anchor=(1, 1),borderaxespad=0).get_frame().set_linewidth(0.0)
# ----------------------------------------------------------------------------------------------------------------

animal='N27_10'
new_df_real = pd.read_csv(path+animal+'_data.csv')
new_df =  pd.read_csv(path+animal+'_model.csv')
figureplot(new_df_real,new_df,c)

# -----------------############################## F Panel - After correct #################################-----------------------
panel=d1
df_results = pd.read_csv(path+'after_correct.csv')
sns.lineplot(x='delays',y='accuracy_data',hue='after_correct',data=df_results.loc[df_results.after_correct!=-1], markeredgewidth = 0.2, ax=panel, marker='o', palette=['crimson','darkgreen'], linestyle = '', err_style="bars", legend=False)
sns.lineplot(x='delays',y='accuracy_model',hue='after_correct',data=df_results.loc[df_results.after_correct!=-1], ax=panel,marker='', palette=['crimson','darkgreen'],  legend=False)
panel.hlines(xmin=0,xmax=10, y=0.5, linestyles = ':')
panel.set_xlabel("Delay (s)")
panel.set_ylim(0.45,1)
panel.hlines(xmin=0,xmax=10, y=0.5, linestyles = ':')
panel.set_ylabel(" Accuracy")

panel=d2
sns.lineplot(x='delays',y='repeat_data',data=df_results, marker='o', hue='after_correct', palette=['crimson','darkgreen'], markeredgewidth = 0.2, ax=panel, legend=False,err_style="bars", linestyle = '')
sns.lineplot(x='delays',y='repeat_model',data=df_results,hue='after_correct', palette=['crimson','darkgreen'],ax=panel, legend=False)
panel.hlines(xmin=0,xmax=10, y=0.5, linestyles = ':')
panel.set_ylim(0.45,1)
legend_elements = [Line2D([0], [0], color='crimson', label='After incorrect'),
                    Line2D([0], [0], color='darkgreen', label='After correct')]

panel.legend(handles=legend_elements, fontsize=6, ncol=1).get_frame().set_linewidth(0.0)
panel.set_xlabel("Delay (s)")
panel.set_ylabel("Repeating bias")
panel.locator_params(nbins=3)
# ----------------------------------------------------------------------------------------------------------------

panel=e1
df_results = pd.read_csv(path+'\previous_correct.csv')
sns.lineplot(x='delays',y='accuracy_data',hue='previous_correct',data=df_results.loc[df_results.previous_correct!=-1], markeredgewidth = 0.2, ax=panel, marker='o', palette=['crimson','darkgreen'], linestyle = '', err_style="bars", legend=False)
sns.lineplot(x='delays',y='accuracy_model',hue='previous_correct',data=df_results.loc[df_results.previous_correct!=-1], ax=panel,marker='', palette=['crimson','darkgreen'],  legend=False)
panel.hlines(xmin=0,xmax=10, y=0.5, linestyles = ':')
panel.set_xlabel("Delay (s)")
panel.set_ylim(0.45,1)
panel.hlines(xmin=0,xmax=10, y=0.5, linestyles = ':')
panel.set_ylabel(" Accuracy")

panel=e2
sns.lineplot(x='delays',y='repeat_data',data=df_results, marker='o', hue='previous_correct', palette=['crimson','darkgreen'], markeredgewidth = 0.2, ax=panel, legend=False,err_style="bars", linestyle = '')
sns.lineplot(x='delays',y='repeat_model',data=df_results,hue='previous_correct', palette=['crimson','darkgreen'],ax=panel, legend=False)
panel.hlines(xmin=0,xmax=10, y=0.5, linestyles = ':')
panel.set_ylim(0.45,1)
legend_elements = [Line2D([0], [0], color='crimson', label='Before incorrect trial'),
                   Line2D([0], [0], color='darkgreen', label='Before correct trial')]

panel.legend(handles=legend_elements, fontsize=6, ncol=1).get_frame().set_linewidth(0.0)
panel.set_xlabel("Delay (s)")
panel.set_ylabel("Repeating bias")
panel.locator_params(nbins=3)


# ------------------------------------------------------------------------------------------------------------

cumulative_autocorrelation_hit_model=pd.read_csv(analysis_path+'hit_autocorrelation_10_model_V2.csv', index_col=0)
cumulative_autocorrelation_repeat_model=pd.read_csv(analysis_path+'repeat_autocorrelation_10_model_V2.csv', index_col=0)

cumulative_autocorrelation_hit_data =pd.read_csv(analysis_path+'hit_autocorrelation.csv', index_col=0)
cumulative_autocorrelation_repeat_data =pd.read_csv(analysis_path+'repeat_autocorrelation.csv', index_col=0)

panel = f
color = 'darkgreen'
corr = cumulative_autocorrelation_hit_model.mean(axis=1)

panel.plot(np.arange(1,len(corr)+1), corr ,marker='.', linestyle='', color='grey')
params, cov = curve_fit(exponential_decay, np.arange(len(corr)), corr.values)
print(params[1])

corr = cumulative_autocorrelation_hit_data[:25]
df_lower= pd.DataFrame()
df_upper= pd.DataFrame()

for timepoint in range(len(corr)):
    mean_surr = []

    # recover the values for that specific timepoint
    array = corr.iloc[timepoint].to_numpy()

    array = array[~np.isnan(array)]
    # iterate several times with resampling: chose X time among the same list of values
    for iteration in range(1000):
        x = np.random.choice(array, size=len(array), replace=True)
        # recover the mean of that new distribution
        mean_surr.append(np.mean(x))

    df_lower.at[0,timepoint] = np.percentile(mean_surr, 2.5)
    df_upper.at[0,timepoint] = np.percentile(mean_surr, 97.5)

lower =  df_lower.iloc[0].values[:25]
upper =  df_upper.iloc[0].values[:25]

mean=corr.mean(axis=1)[:25]
lower = corr.quantile(q=0.025, axis=1, numeric_only=True)
upper = corr.quantile(q=0.975, axis=1, numeric_only=True)

# It could also be used sns.lineplot, but this way is faster. Plots mean and sem curves, positive and negative, and fills the space between
panel.plot(np.arange(1,len(corr)+1), mean ,marker='', color=color)

# panel.fill_between(np.arange(1,len(corr)+1),  lower, upper, alpha=0.2, color='grey')
panel.hlines(y=0,xmin=0,xmax=25,linestyles=':')
panel.set_xlabel('Trial indexes')
panel.set_xlim(0,25)

params, cov = curve_fit(exponential_decay, np.arange(len(corr.mean(axis=1))), corr.mean(axis=1).values)
print(params[1])
# ------------------------------------------------------------------------------------------------------------

os.chdir(path)

panel = g
color = 'indigo'
corr = cumulative_autocorrelation_repeat_model.mean(axis=1)

panel.plot(np.arange(1,len(corr)+1), corr ,marker='.', linestyle='', color='grey')
params, cov = curve_fit(exponential_decay, np.arange(len(corr)), corr.values)
print(params[1])

corr = cumulative_autocorrelation_repeat_data[:25]
df_lower= pd.DataFrame()
df_upper= pd.DataFrame()

for timepoint in range(len(corr)):
    mean_surr = []

    # recover the values for that specific timepoint
    array = corr.iloc[timepoint].to_numpy()

    array = array[~np.isnan(array)]
    # iterate several times with resampling: chose X time among the same list of values
    for iteration in range(1000):
        x = np.random.choice(array, size=len(array), replace=True)
        # recover the mean of that new distribution
        mean_surr.append(np.mean(x))

    df_lower.at[0,timepoint] = np.percentile(mean_surr, 2.5)
    df_upper.at[0,timepoint] = np.percentile(mean_surr, 97.5)

lower =  df_lower.iloc[0].values[:25]
upper =  df_upper.iloc[0].values[:25]

mean=corr.mean(axis=1)[:25]
lower = corr.quantile(q=0.025, axis=1, numeric_only=True)
upper = corr.quantile(q=0.975, axis=1, numeric_only=True)

# It could also be used sns.lineplot, but this way is faster. Plots mean and sem curves, positive and negative, and fills the space between
panel.plot(np.arange(1,len(corr)+1), mean ,marker='', color=color)
# panel.fill_between(np.arange(1,len(corr)+1),  lower, upper, alpha=0.2, color='grey',linewidth=0)
panel.hlines(y=0,xmin=0,xmax=25,linestyles=':')
panel.set_xlabel('Trial indexes')
panel.set_xlim(0,25)
params, cov = curve_fit(exponential_decay, np.arange(len(corr.mean(axis=1))), corr.mean(axis=1).values)
print(params[1])

# ------------------------------------------------------------------------------------------------------------


# Show the figure
sns.despine()
plt.subplots_adjust(left=0.07,
                    bottom=0.07,
                    right=0.97,
                    top=0.97,
                    wspace=1.5,
                    hspace=0.5)

plt.savefig(save_path+'/Fig. Supp. 3. Extra results from model_v3.svg', bbox_inches='tight',dpi=300)
plt.show()