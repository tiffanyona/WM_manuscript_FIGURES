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

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore', 'FigureCanvasAgg is non-interactive')
#Import all needed libraries
from neo.core import SpikeTrain
from quantities import ms, s

from neo.core import SpikeTrain
from quantities import ms, s
from elephant.statistics import time_histogram

import numpy as np
import pandas as pd
import seaborn as sns


from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import ROOT, FIGURES_OUT, DATA_DIR
sys.path.insert(0, str(ROOT / 'src'))
from functions import add_stat_annotation, trials_synch, synch_trial, distribution

save_path = str(DATA_DIR / 'supp_figures' / 'supp_fig_19_synch_session') + '/'  # input data
fig_out_path = str(FIGURES_OUT / 'supp_figures' / 'supp_fig_19_synch_session') + '/'
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
fig = plt.figure(figsize=(12*cm, 5*cm))
gs = gridspec.GridSpec(nrows=1, ncols=3, figure=fig)

a = fig.add_subplot(gs[0, 0:2])
b = fig.add_subplot(gs[0, 2:3])


fig.text(0.01, 1, 'a', fontsize=10, fontweight='bold', va='top')
fig.text(0.65, 1, 'b', fontsize=10, fontweight='bold', va='top')


# #######################################################################################
# Panel A: Example trial
# #######################################################################################

file_name = 'panel_a_per_trial_synchrony_and_behavioral_state_all_sessions'
df_final = pd.read_csv(save_path+file_name+'.csv', index_col=0)

# df_final = df_final.loc[df_final.T_norm < 0.6 ]
df_final['state'] = np.where(df_final['WM_roll']>0.5, 1, 0)

df_final['T_norm'] = np.around(df_final['T_norm'],2)

df_results = pd.DataFrame()
df_results['synch_window'] = df_final.groupby(['T_norm','animal'])['synch_window'].mean()
df_results.reset_index(inplace=True)

panel = a
sns.lineplot(x='T_norm',y='synch_window',data=df_results,ax=panel, color='black', errorbar=('ci', 95))

panel.hlines(xmin=0,xmax=1,y=1, linestyle=':')
panel.set_ylim(0.9,2.5)
panel.set_ylabel('Synch')
panel.set_xlabel('Normalized trial index')


# #######################################################################################
# Panel B: Synchrony depending on state
# #######################################################################################

file_name = 'panel_b_mean_prestimulus_firing_rate_by_HMM_state_per_session'
df_final = pd.read_csv(save_path+file_name+'.csv', header=None)
df_final.columns = ['RL', 'WM']

panel = b

melted_data = pd.melt(df_final)
sns.boxplot(data=melted_data, x='variable', y='value', hue='variable',
            legend=False, ax=b, order=['WM', 'RL'], palette=['darkgreen', 'indigo'],
                medianprops=dict(color='white', linewidth=1.5),
                boxprops=dict(linewidth=0),
                whiskerprops=dict(color='black', lw=1),
                capprops=dict(color='black', lw=0))
panel.plot([1, 0], df_final.T.values, color='black', alpha=0.3, marker='')
panel.set_ylabel('Mean rate (spks/s)')
panel.set_title('Mean pre-stim rate', fontsize=7)
panel.set_xticks([0,1],['WM','RepL'])

add_stat_annotation(panel, data=melted_data, x='variable', y='value',
                    box_pairs=[( 'WM','RL')],
                    test='t-test_paired', text_format='star', loc='inside', line_offset_to_box=0.05, text_offset=-0.5, line_offset=0, verbose=1, fontsize=6, linewidth=0.5)


# #######################################################################################
# Save Figure
# #######################################################################################

# Show the figure
sns.despine(offset = 10)
plt.subplots_adjust(left=0.07,
                    bottom=0.07,
                    right=0.97,
                    top=0.97,
                    wspace=0.5,
                    hspace=0.5)
plt.tight_layout()

# plt.savefig(fig_out_path+'/Fig 7_Brain state_V3.svg', bbox_inches='tight',dpi=1000)
# plt.savefig(fig_out_path+'/Fig 7_Brain state_V2.png', bbox_inches='tight',dpi=1000)

with warnings.catch_warnings():
    warnings.simplefilter('ignore', UserWarning)
    plt.show()
