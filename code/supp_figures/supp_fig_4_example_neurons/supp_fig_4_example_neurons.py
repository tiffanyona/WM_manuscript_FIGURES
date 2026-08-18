
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
import seaborn as sns

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import ROOT, FIGURES_OUT, DATA_DIR
sys.path.insert(0, str(ROOT / 'src'))
from functions import COLORLEFT, COLORRIGHT, new_convolve, convolveandplot

save_path = str(FIGURES_OUT / 'supp_figures' / 'supp_fig_4_example_neurons')
Path(save_path).mkdir(parents=True, exist_ok=True)
path = str(DATA_DIR/ 'supp_figures' / 'supp_fig_4_example_neurons')

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
fig = plt.figure(figsize=(18*cm, 20*cm))
gs = gridspec.GridSpec(nrows=8, ncols=8, figure=fig, height_ratios=[2,1,2,1,2,1,2,1])

# Create the subplots
a1 = fig.add_subplot(gs[0, 0:2])
a2 = fig.add_subplot(gs[1, 0:2])

b1 = fig.add_subplot(gs[0, 2:4])
b2 = fig.add_subplot(gs[1, 2:4])

c1 = fig.add_subplot(gs[0, 4:6])
c2 = fig.add_subplot(gs[1, 4:6])

d1 = fig.add_subplot(gs[0, 6:8])
d2 = fig.add_subplot(gs[1, 6:8])

e1 = fig.add_subplot(gs[2, 0:2])
e2 = fig.add_subplot(gs[3, 0:2])

f1 = fig.add_subplot(gs[2, 2:4])
f2 = fig.add_subplot(gs[3, 2:4])

g1 = fig.add_subplot(gs[2, 4:6])
g2 = fig.add_subplot(gs[3, 4:6])

h1 = fig.add_subplot(gs[2, 6:8])
h2 = fig.add_subplot(gs[3, 6:8])

i1 = fig.add_subplot(gs[4, 0:2])
i2 = fig.add_subplot(gs[5, 0:2])

j1 = fig.add_subplot(gs[4, 2:4])
j2 = fig.add_subplot(gs[5, 2:4])

k1 = fig.add_subplot(gs[4, 4:6])
k2 = fig.add_subplot(gs[5, 4:6])

l1 = fig.add_subplot(gs[4, 6:8])
l2 = fig.add_subplot(gs[5, 6:8])

m1 = fig.add_subplot(gs[6, 0:2])
m2 = fig.add_subplot(gs[7, 0:2])

n1 = fig.add_subplot(gs[6, 2:4])
n2 = fig.add_subplot(gs[7, 2:4])

o1 = fig.add_subplot(gs[6, 4:6])
o2 = fig.add_subplot(gs[7, 4:6])

p1 = fig.add_subplot(gs[6, 6:8])
p2 = fig.add_subplot(gs[7, 6:8])

# fig.text(0.01, 0.99, 'a', fontsize=10, fontweight='bold', va='top')
# fig.text(0.5, 0.99, 'b', fontsize=10, fontweight='bold', va='top')
# fig.text(0.01, 0.75, 'c', fontsize=10, fontweight='bold', va='top')
# fig.text(0.5, 0.75, 'f', fontsize=10, fontweight='bold', va='top')
# fig.text(0.75, 0.75, 'g', fontsize=10, fontweight='bold', va='top')

# fig.text(0.01, 0.51, 'd', fontsize=10, fontweight='bold', va='top')
# fig.text(0.01, 0.4, 'e', fontsize=10, fontweight='bold', va='top')
# fig.text(0.01, 0.28, 'f', fontsize=10, fontweight='bold', va='top')
# fig.text(0.01, 0.17, 'g', fontsize=10, fontweight='bold', va='top')

# fig.text(0.51, 0.51, 'i', fontsize=10, fontweight='bold', va='top')
# fig.text(0.51, 0.4, 'j', fontsize=10, fontweight='bold', va='top')
# fig.text(0.51, 0.28, 'k', fontsize=10, fontweight='bold', va='top')
# fig.text(0.51, 0.17, 'l', fontsize=10, fontweight='bold', va='top')

# #########################################################################################
# Setup: convolveandplot — spike raster and firing rate plotter
# #########################################################################################

# #########################################################################################
# Panel (unused): crossdecoder for 3s delay
# #########################################################################################

# file_name = 'crossdecoder_WM_roll_1_3s_alignedstimulus_nosubstract'

# #########################################################################################
# Panel A: single neuron example — E20_2022-02-13, neuron 81 (subplot a, delay=10s)
# #########################################################################################

file_name = 'E20_2022-02-13_15-10-51_neuron_81'
df = pd.read_csv(path+f'\{file_name}.csv', index_col=0)

delay = 10
cluster_id = df.cluster_id.unique()[0]

colors = [COLORRIGHT,COLORLEFT]
# colors = ['darkgreen','crimson']
labels = ['Right stimulus','Left stimulus']
# labels = ['Correct','Incorrect']
align = 'Stimulus_ON'

j=1
temp_df = df.loc[(df.WM_roll >0.6)&(df.hit ==1)]
j = convolveandplot(temp_df, a1, a2, variable='reward_side', cluster_id=cluster_id, delay=delay, j=j, cue_off=0.35, start=-2.5, kernel=200, show_xlabel=False)
a1.set_title(file_name.split('_neuron_')[0], fontsize=7)

# #########################################################################################
# Panel B: single neuron example — E14_2021-04-02, neuron 361 (subplot b, delay=10s)
# #########################################################################################

file_name = 'E14_2021-04-02_12-53-42_neuron_361'
df = pd.read_csv(path+f'\{file_name}.csv', index_col=0)

delay = 10
cluster_id = df.cluster_id.unique()[0]

j=1
temp_df = df.loc[(df.WM_roll >0.6)&(df.hit ==1)]
j = convolveandplot(temp_df, b1, b2, variable='reward_side', cluster_id=cluster_id, delay=delay, j=j, cue_off=0.35, start=-2.5, kernel=200, show_xlabel=False)
b1.set_title(file_name.split('_neuron_')[0], fontsize=6)

# #########################################################################################
# Panel C: single neuron example — E17_2022-02-01, neuron 304 (subplot c, delay=10s)
# #########################################################################################

file_name = 'E17_2022-02-01_17-02-16_neuron_304'
df = pd.read_csv(path+f'\{file_name}.csv', index_col=0)

delay = 10
cluster_id = df.cluster_id.unique()[0]

j=1
temp_df = df.loc[(df.WM_roll >0.6)&(df.hit ==1)]
j = convolveandplot(temp_df, c1, c2, variable='reward_side', cluster_id=cluster_id, delay=delay, j=j, cue_off=0.35, start=-2.5, kernel=200, show_xlabel=False)
c1.set_title(file_name.split('_neuron_')[0], fontsize=6)

# #########################################################################################
# Panel D: single neuron example — E17_2022-02-02, neuron 33 (subplot d, delay=10s)
# #########################################################################################

file_name = 'E17_2022-02-02_17-13-06_neuron_33'
df = pd.read_csv(path+f'\{file_name}.csv', index_col=0)

delay = 10
cluster_id = df.cluster_id.unique()[0]

j=1
temp_df = df.loc[(df.WM_roll >0.6)&(df.hit ==1)]
j = convolveandplot(temp_df, d1, d2, variable='reward_side', cluster_id=cluster_id, delay=delay, j=j, cue_off=0.35, start=-2.5, kernel=200, show_xlabel=False)
d1.set_title(file_name.split('_neuron_')[0], fontsize=6)

# #########################################################################################
# Panel E: single neuron example — E17_2022-02-02, neuron 265 (subplot e, delay=1s)
# #########################################################################################

file_name = 'E17_2022-02-02_17-13-06_neuron_265'
df = pd.read_csv(path+f'\{file_name}.csv', index_col=0)

delay = 1
cluster_id = df.cluster_id.unique()[0]

j=1
temp_df = df.loc[(df.WM_roll >0.6)&(df.hit ==1)]
j = convolveandplot(temp_df, e1, e2, variable='reward_side', cluster_id=cluster_id, delay=delay, j=j, cue_off=0.35, start=-2.5, kernel=200, show_xlabel=False)
e1.set_title(file_name.split('_neuron_')[0], fontsize=7)

# #########################################################################################
# Panel F: single neuron example — E17_2022-02-02, neuron 233 (subplot f, delay=3s)
# #########################################################################################

file_name = 'E17_2022-02-02_17-13-06_neuron_233'
df = pd.read_csv(path+f'\{file_name}.csv', index_col=0)

delay = 3
cluster_id = df.cluster_id.unique()[0]

j=1
temp_df = df.loc[(df.WM_roll >0.6)&(df.hit ==1)]
j = convolveandplot(temp_df, f1, f2, variable='reward_side', cluster_id=cluster_id, delay=delay, j=j, cue_off=0.35, start=-2.5, kernel=200, show_xlabel=False)
f1.set_title(file_name.split('_neuron_')[0], fontsize=7)

# #########################################################################################
# Panel G: single neuron example — E17_2022-02-02, neuron 288 (subplot g, delay=3s)
# #########################################################################################

file_name = 'E17_2022-02-02_17-13-06_neuron_288'
df = pd.read_csv(path+f'\{file_name}.csv', index_col=0)

delay = 3
cluster_id = df.cluster_id.unique()[0]

j=1
temp_df = df.loc[(df.WM_roll >0.6)&(df.hit ==1)]
j = convolveandplot(temp_df, g1, g2, variable='reward_side', cluster_id=cluster_id, delay=delay, j=j, cue_off=0.35, start=-2.5, kernel=200, show_xlabel=False)
g1.set_title(file_name.split('_neuron_')[0], fontsize=7)

# #########################################################################################
# Panel H: single neuron example — E04_2021-03-30, neuron 169 (subplot h, delay=10s)
# #########################################################################################

file_name = 'E04_2021-03-30_11-20-16_neuron_169'
df = pd.read_csv(path+f'\{file_name}.csv', index_col=0)

delay = 10
cluster_id = df.cluster_id.unique()[0]

j=1
temp_df = df.loc[(df.WM_roll >0.6)&(df.hit ==1)]
j = convolveandplot(temp_df, h1, h2, variable='reward_side', cluster_id=cluster_id, delay=delay, j=j, cue_off=0.35, start=-2.5, kernel=200, show_xlabel=False)

h1.get_xaxis().set_visible(False)
h1.set_title(file_name.split('_neuron_')[0], fontsize=6)

# #########################################################################################
# Panel I: single neuron example — E14_2021-04-02, neuron 511 (subplot i, delay=10s)
# #########################################################################################

file_name = 'E14_2021-04-02_12-53-42_neuron_511'
df = pd.read_csv(path+f'\{file_name}.csv', index_col=0)

delay = 10
cluster_id = df.cluster_id.unique()[0]

j=1
temp_df = df.loc[(df.WM_roll >0.6)&(df.hit ==1)]
j = convolveandplot(temp_df, i1, i2, variable='reward_side', cluster_id=cluster_id, delay=delay, j=j, cue_off=0.35, start=-2.5, kernel=200, show_xlabel=False)
i1.set_title(file_name.split('_neuron_')[0], fontsize=6)

# #########################################################################################
# Panel J: single neuron example — E22_2022-01-22, neuron 246 (subplot j, delay=10s)
# #########################################################################################

file_name = 'E22_2022-01-22_17-09-15_neuron_246'
df = pd.read_csv(path+f'\{file_name}.csv', index_col=0)

delay = 10
cluster_id = df.cluster_id.unique()[0]

j=1
temp_df = df.loc[(df.WM_roll >0.6)&(df.hit ==1)]
j = convolveandplot(temp_df, j1, j2, variable='reward_side', cluster_id=cluster_id, delay=delay, j=j, cue_off=0.35, start=-2.5, kernel=200, show_xlabel=False)
j1.set_title(file_name.split('_neuron_')[0], fontsize=6)

# #########################################################################################
# Panel K: single neuron example — E22_2022-01-13, neuron 381 (subplot k, delay=10s)
# #########################################################################################

file_name = 'E22_2022-01-13_16-34-24_neuron_381'
df = pd.read_csv(path+f'\{file_name}.csv', index_col=0)

delay = 10
cluster_id = df.cluster_id.unique()[0]

j=1
temp_df = df.loc[(df.WM_roll >0.6)&(df.hit ==1)]
j = convolveandplot(temp_df, k1, k2, variable='reward_side', cluster_id=cluster_id, delay=delay, j=j, cue_off=0.35, start=-2.5, kernel=200, show_xlabel=False)
k1.set_title(file_name.split('_neuron_')[0], fontsize=6)

# #########################################################################################
# Panel L: single neuron example — E22_2022-01-14, neuron 16 (subplot l, delay=10s)
# #########################################################################################

file_name = 'E22_2022-01-14_16-50-37_neuron_16'
df = pd.read_csv(path+f'\{file_name}.csv', index_col=0)

delay = 10
cluster_id = df.cluster_id.unique()[0]

j=1
temp_df = df.loc[(df.WM_roll >0.6)&(df.hit ==1)]
j = convolveandplot(temp_df, l1, l2, variable='reward_side', cluster_id=cluster_id, delay=delay, j=j, cue_off=0.35, start=-2.5, kernel=200)
l1.set_title(file_name.split('_neuron_')[0], fontsize=6)

# #########################################################################################
# Panel M: single neuron example — E20_2022-02-14, neuron 204 (subplot m, delay=10s)
# #########################################################################################

file_name = 'E20_2022-02-14_16-01-30_neuron_204'
df = pd.read_csv(path+f'\{file_name}.csv', index_col=0)

delay = 10
cluster_id = df.cluster_id.unique()[0]

j=1
temp_df = df.loc[(df.WM_roll >0.6)&(df.hit ==1)]
j = convolveandplot(temp_df, m1, m2, variable='reward_side', cluster_id=cluster_id, delay=delay, j=j, cue_off=0.35, start=-2.5, kernel=200)
m1.set_title(file_name.split('_neuron_')[0], fontsize=6)

# #########################################################################################
# Panel N: single neuron example — E20_2022-03-01, neuron 88 (subplot n, delay=10s)
# #########################################################################################

file_name = 'E20_2022-03-01_16-11-01_neuron_88'
df = pd.read_csv(path+f'\{file_name}.csv', index_col=0)

delay = 10
cluster_id = df.cluster_id.unique()[0]

j=1
temp_df = df.loc[(df.WM_roll >0.6)&(df.hit ==1)]
j = convolveandplot(temp_df, n1, n2, variable='reward_side', cluster_id=cluster_id, delay=delay, j=j, cue_off=0.35, start=-2.5, kernel=200)
n1.set_title(file_name.split('_neuron_')[0], fontsize=6)

# #########################################################################################
# Panel O: single neuron example — E20_2022-03-01, neuron 95 (subplot o, delay=10s)
# #########################################################################################

file_name = 'E20_2022-03-01_16-11-01_neuron_95'
df = pd.read_csv(path+f'\{file_name}.csv', index_col=0)

delay = 10
cluster_id = df.cluster_id.unique()[0]

j=1
temp_df = df.loc[(df.WM_roll >0.6)&(df.hit ==1)]
j = convolveandplot(temp_df, o1, o2, variable='reward_side', cluster_id=cluster_id, delay=delay, j=j, cue_off=0.35, start=-2.5, kernel=200)
o1.set_title(file_name.split('_neuron_')[0], fontsize=6)

# #########################################################################################
# Panel P: single neuron example — E22_2022-01-14, neuron 56 (subplot p, delay=10s)
# #########################################################################################

file_name = 'E22_2022-01-14_16-50-37_neuron_56'
df = pd.read_csv(path+f'\{file_name}.csv', index_col=0)

delay = 10
cluster_id = df.cluster_id.unique()[0]

j=1
temp_df = df.loc[(df.WM_roll >0.6)&(df.hit ==1)]
j = convolveandplot(temp_df, p1, p2, variable='reward_side', cluster_id=cluster_id, delay=delay, j=j, cue_off=0.35, start=-2.5, kernel=200)
p1.set_title(file_name.split('_neuron_')[0], fontsize=6)

# 'E22_2022-01-14_16-50-37_neuron_56' Delay
# 'E22_2022-01-14_16-50-37_neuron_47' Choice
# 'E22_2022-01-14_16-50-37_neuron_23' Choice

# Show the figure
plt.subplots_adjust(left=0.07,
                    bottom=0.07,
                    right=0.97,
                    top=0.97,
                    wspace=1.0,
                    hspace=0.95)

# plt.savefig(save_path+'/supp_fig_4_example_neurons.svg', bbox_inches='tight',dpi=300)
# plt.savefig(save_path+'/supp_fig_4_example_neurons.png', bbox_inches='tight',dpi=300)

plt.show()
