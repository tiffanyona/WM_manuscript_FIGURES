# -*- coding: utf-8 -*-
"""
Fig 4 panel — original code + panels i and j added from supp_fig_XX_reversals.ipynb
Changes vs original:
  1. figsize height increased: 18cm -> 28cm
  2. GridSpec expanded: 6 rows -> 8 rows, height_ratios added
  3. Four new subplot axes: i_heat, i_line, j_heat, j_line
  4. Two new panel labels: 'i' and 'j'
  5. Panel i code (stimulus-aligned heatmap + lineplot) appended
  6. Panel j code (reversal-aligned heatmap + lineplot) appended
Dead code removed:
  - Duplicate 'from pathlib import Path' import
  - Unused R imports: Rstats, scales, lmerTest
  - Intermediate variables: 'variables', 'hits' (inlined into variables_combined)
  - Unused variables: labels, align, j (convolveandplot return)
  - Silent wilcoxon calls (added print)
  - normalize=True flag (inlined unconditional normalization)
  - plot_df = new_df.copy() (use new_df directly)
  - Redundant ax.set_xlabel('') before ax.xaxis.set_visible(False)
"""
COLORLEFT = 'teal'
COLORRIGHT = '#FF8D3F'

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from pathlib import Path
os.environ.setdefault('R_LIBS_USER',
    str(Path.home() / 'Documents' / 'R' / 'win-library' / '4.6'))
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

base = importr('base')
car  = importr('car')
lme4 = importr('lme4')

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import ROOT, FIGURES_OUT, DATA_DIR
sys.path.insert(0, str(ROOT / 'src'))
import functions as plots

save_path = str(FIGURES_OUT / 'fig_4_ephys_errors') + '/'
path = str(DATA_DIR / 'fig_4_ephys_errors') + '/'
os.chdir(path)

cm = 1/2.54
sns.set_context('paper', rc={'axes.labelsize': 7,
                              'lines.linewidth': 1,
                              'lines.markersize': 2,
                              'legend.fontsize': 7,
                              'xtick.major.size': 1,
                              'xtick.labelsize': 6,
                              'ytick.major.size': 1,
                              'ytick.labelsize': 6,
                              'xtick.major.pad': 0,
                              'ytick.major.pad': 0,
                              'xlabel.labelpad': -10})

fig = plt.figure(figsize=(12*cm, 25*cm))
gs = gridspec.GridSpec(nrows=8, ncols=2, figure=fig,
                       height_ratios=[1, 1, 1, 1, 1, 1, 2.5, 0.8])

# Original subplots
# g and h: nested GridSpec with hspace=0 so the 3 subpanels are joined
gs_g = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[3:6, 1:2], hspace=0.05)
a1 = fig.add_subplot(gs_g[0])
a2 = fig.add_subplot(gs_g[1])
a3 = fig.add_subplot(gs_g[2])

gs_h = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[3:6, 0:1], hspace=0.05)
b1 = fig.add_subplot(gs_h[0])
b2 = fig.add_subplot(gs_h[1])
b3 = fig.add_subplot(gs_h[2])
g2 = fig.add_subplot(gs[0, 1:2])
j0 = fig.add_subplot(gs[2, 0:1])
j1 = fig.add_subplot(gs[0, 0:1])
j2 = fig.add_subplot(gs[1, 0:1])
j3 = fig.add_subplot(gs[1, 1:2])
k  = fig.add_subplot(gs[2, 1:2])

# New panels i and j
i_heat = fig.add_subplot(gs[6, 0:1])
i_line = fig.add_subplot(gs[7, 0:1])

j_heat = fig.add_subplot(gs[6, 1:2])
j_line = fig.add_subplot(gs[7, 1:2])

# Panel labels — y positions match row tops in the 8-row GridSpec
# height_ratios=[1,1,1,1,1,1,2.5,0.8] total=9.3, span=0.92 (top=0.97,bot=0.05)
# row tops: r0=0.97, r1=0.87, r2=0.77, r3=0.67, r4=0.57, r5=0.46, r6=0.36
fig.text(0.01, 1.00, 'a', fontsize=10, fontweight='bold', va='top')  # j1 row0
fig.text(0.01, 0.9, 'b', fontsize=10, fontweight='bold', va='top')  # j2 row1
fig.text(0.01, 0.77, 'c', fontsize=10, fontweight='bold', va='top')  # j0 row2
fig.text(0.52, 1.00, 'd', fontsize=10, fontweight='bold', va='top')  # g2 row0
fig.text(0.52, 0.9, 'e', fontsize=10, fontweight='bold', va='top')  # j3 row1
fig.text(0.52, 0.77, 'f', fontsize=10, fontweight='bold', va='top')  # k  row2
fig.text(0.01, 0.67, 'g', fontsize=10, fontweight='bold', va='top')  # b1 rows3-5
fig.text(0.52, 0.67, 'h', fontsize=10, fontweight='bold', va='top')  # a1 rows3-5
fig.text(0.01, 0.30, 'i', fontsize=10, fontweight='bold', va='top')  # i_heat row6
fig.text(0.52, 0.30, 'j', fontsize=10, fontweight='bold', va='top')  # j_heat row6

# #######################################################################################
# Panels a / b / c: stimulus, response, and delay decoder accuracy
# #######################################################################################

colors = ['crimson', 'darkgreen']
variables_combined = ['WM_roll_0', 'WM_roll_1']

file_name = '/panel_a_stimulus_decoding_accuracy_STM_correct_vs_incorrect'
df_cum_sti = pd.read_csv(path + file_name + '.csv', index_col=0)
scores = df_cum_sti.loc[df_cum_sti.delay == 10].groupby('session').score.mean().reset_index()
list_exclude = scores.loc[scores.score < 0.60].session.unique()
df_cum_sti = df_cum_sti.loc[df_cum_sti.delay == 10][~df_cum_sti['session'].isin(list_exclude)]
delay = 10
plots.plotsingledelay(df_cum_sti, j1, colors, variables_combined, delay,
                      baseline=0, invert_list=[True, False])
j1.set_xlim(-2, 14)
j1.set_title('Stimulus code', fontweight='bold', fontsize=6)
j1.set_xlabel('Testing time from Stimulus onset (s)')
sns.despine(ax=j1)
file_name = '/panel_b_response_decoding_accuracy_STM_correct_vs_incorrect'
df_cum_sti = pd.read_csv(path + file_name + '.csv', index_col=0)
scores = df_cum_sti.loc[df_cum_sti.delay == 10].groupby('session').score.mean().reset_index()
list_exclude = scores.loc[scores.score < 0.60].session.unique()
df_cum_sti = df_cum_sti.loc[df_cum_sti.delay == 10][~df_cum_sti['session'].isin(list_exclude)]
plots.plotsingledelay(df_cum_sti, j2, colors, variables_combined, delay, baseline=0)
j2.set_xlim(-2, 14)
j2.set_title('Response code', fontweight='bold', fontsize=6)
j2.set_xlabel('Testing time from Stimulus onset (s)')
sns.despine(ax=j2)

file_name = '/panel_c_delay_decoding_accuracy_STM_correct_vs_incorrect'
df_cum_sti = pd.read_csv(path + file_name + '.csv', index_col=0)
plots.plotsingledelay(df_cum_sti, j0, colors, variables_combined, delay, baseline=0)
j0.set_xlim(-2, 14)
j0.set_title('Delay code', fontweight='bold', fontsize=6)
j0.set_xlabel('Testing time from Stimulus onset (s)')
sns.despine(j0)

# #######################################################################################
# Panel d: delay-code log-odds by trial outcome and epoch (boxplot + stats)
# #######################################################################################

file_name = 'panel_d_delay_decoder_log_odds_by_epoch_STM_correct_vs_incorrect'
df_final = pd.read_csv(path + file_name + '.csv', index_col=0)

panel = g2
df_results = df_final.groupby(['session','trial_type','epoch']).log_odds.mean().reset_index()

sns.boxplot(x='trial_type', y='log_odds', hue='epoch',
            order=['WM_roll_1','WM_roll_0'], showcaps=False, showfliers=False,
            palette=['darkgreen','lightgreen','crimson','lightcoral'],
            medianprops=dict(color="white", linewidth=1), gap =0.25,
            linewidth=0, ax=panel, data=df_results, width=1,
            legend=False)

# Scatter dots — filled circles; Incorrect Early = open circles to match target
for xpos, tt, ep, col, fc in [
    (-0.25, 'WM_roll_1', 'early', 'darkgreen',  'darkgreen'),
    ( 0.25, 'WM_roll_1', 'late',  'lightgreen', 'lightgreen'),
    ( 0.75, 'WM_roll_0', 'early', 'crimson',    'red'),      # open circles
    ( 1.25, 'WM_roll_0', 'late',  'lightcoral', 'lightcoral'),
]:
    df_plots = df_results.loc[(df_results.trial_type==tt) & (df_results.epoch==ep)]
    xA = np.random.normal(xpos, 0.05, len(df_plots))
    panel.scatter(xA, df_plots['log_odds'].values,
                  color=fc, edgecolors='white', s=8, linewidths=0.5, zorder=3)

panel.set_ylim(-2.5, 5)
panel.axhline(y=0, linestyle=':', color='black')
panel.set_ylabel('Decoding accuracy \n (log odds)')
panel.set_xlabel('')

# x-axis: 4 ticks at box positions, labelled Early/Late/Early/Late
panel.set_xticks([-0.25, 0.25, 0.75, 1.25])
panel.set_xticklabels(['Early', 'Late', 'Early', 'Late'], fontsize=6)
panel.set_xlim(-0.6, 1.6)

# Group labels below x-axis
panel.text(0.0, 5, 'Correct',   ha='center', fontsize=6, color='darkgreen')
panel.text(1.0, 5, 'Incorrect', ha='center', fontsize=6, color='crimson')
sns.despine(ax=panel)

# Significance stars above each box (ttest_1samp vs 0)
star_y = 4.2
for xpos, tt, ep in [(-0.25,'WM_roll_1','early'),(0.25,'WM_roll_1','late'),
                      (0.75,'WM_roll_0','early'),(1.25,'WM_roll_0','late')]:
    vals = df_final.loc[(df_final.trial_type==tt) & (df_final.epoch==ep)
                        ].groupby('session').log_odds.mean().values
    _, p = stats.ttest_1samp(vals, 0)
    star = plots.p_to_stars(p)
    panel.text(xpos, star_y, star, ha='center', va='bottom', fontsize=7)

with localconverter(ro.default_converter + pandas2ri.converter):
    r_df = ro.conversion.py2rpy(df_final)

formula = "log_odds ~ hit*epoch + (hit+epoch+1|session:fold)"
model = lme4.lmer(formula, data=r_df)
for i, v in enumerate(list(base.summary(model).names)):
    if v == 'coefficients':
        print(base.summary(model).rx2(v))
print(car.Anova(model))

print(stats.ttest_1samp(
    df_final.loc[(df_final.trial_type=='WM_roll_0')&(df_final.epoch=='early')]
    .groupby('session').log_odds.mean().values, 0))
print(stats.ttest_1samp(
    df_final.loc[(df_final.trial_type=='WM_roll_0')&(df_final.epoch=='late')]
    .groupby('session').log_odds.mean().values, 0))

print(stats.wilcoxon(
    df_final.loc[(df_final.trial_type=='WM_roll_0')&(df_final.epoch=='early')]
    .groupby('session')['log_odds'].mean(), alternative='less'))
print(stats.wilcoxon(
    df_final.loc[(df_final.trial_type=='WM_roll_0')&(df_final.epoch=='late')]
    .groupby('session')['log_odds'].mean()))
print(stats.wilcoxon(
    df_final.loc[(df_final.trial_type=='WM_roll_1')&(df_final.epoch=='early')]
    .groupby('session')['log_odds'].mean(), alternative='less'))
print(stats.wilcoxon(
    df_final.loc[(df_final.trial_type=='WM_roll_1')&(df_final.epoch=='late')]
    .groupby('session')['log_odds'].mean()))

# #######################################################################################
# Panel e: single-session decoder example (correct vs incorrect)
# #######################################################################################

delays = [10]
file_name = r'\panel_e_example_session_delay_decoding_correct_vs_incorrect'
df_cum_sti    = pd.read_csv(path + file_name + '.csv', index_col=0)
df_cum_shuffle = pd.read_csv(path + r'\panel_e_example_session_delay_decoding_correct_vs_incorrect_shuffle.csv', index_col=0)

for delay in delays:
    df_sti  = df_cum_sti.loc[df_cum_sti.delay == delay]
    df_iter = df_cum_shuffle.loc[df_cum_shuffle.delay == delay]

    for color, variable in zip(colors, variables_combined):
        real  = np.array(np.mean(df_sti.loc[df_sti['trial_type'] == variable]
                                 .groupby('session').median()
                                 .drop(columns=['fold','delay','score'])
                                 .dropna(axis=1)))
        times = df_sti.loc[df_sti['trial_type'] == variable].dropna(axis=1)
        times = np.array(times.drop(
            columns=['fold','score','trial_type','delay','session'], axis=1
        ).columns.astype(float))

        df_new = pd.DataFrame()
        for iteration in np.arange(1, 50):
            df_new[iteration] = (df_iter.loc[df_iter.trial_type == variable]
                                 .groupby('times').mean()[str(float(iteration))])

        y_mean = df_new.mean(axis=1).values
        lower  = df_new.quantile(q=0.975, interpolation='linear', axis=1).values - y_mean
        upper  = df_new.quantile(q=0.025, interpolation='linear', axis=1).values - y_mean

        j3.set_xlabel('Testing time from Stimulus onset (s)')
        try:
            j3.plot(times, real - 0.5, color=color)
            j3.plot(times, lower + real - 0.5, color=color, linestyle='', alpha=0.6, linewidth=0)
            j3.plot(times, upper + real - 0.5, color=color, linestyle='', alpha=0.6, linewidth=0)
            j3.fill_between(times, lower + real - 0.5, upper + real - 0.5, alpha=0.2, color=color, linewidth=0)
            j3.set_ylim(-0.4, 0.5)
            j3.axhline(y=0, linestyle=':', color='black')
            j3.fill_betweenx(np.arange(-0.1, 0.6, 0.1), 0, 0.4,
                             color='lightgrey', alpha=1, linewidth=0)
            j3.fill_betweenx(np.arange(-0.1, 0.6, 0.1), delay + 0.3, delay + 0.5,
                             color='grey', alpha=.5, linewidth=0)
            j3.set_xlim(-2, 14)
            j3.text(5, -0.3, 'Session E20_2022_02_27', ha='center', fontsize=5, color = 'grey')
            j3.set_ylabel('Excess decoding \n accuracy')
        except Exception:
            print('not this condition for this delay')
            continue
sns.despine(ax=j3)

# #######################################################################################
# Panel f: single neuron firing rate — correct vs incorrect trials
# #######################################################################################

file_name = r'\panel_f_neuron153_spike_times_all_10s_delay_trials'
df = pd.read_csv(path + file_name + '.csv', index_col=0)
delay = 10; cluster_id = 153

# Correct trials in grey (background)
temp_df = df.loc[(df.WM_roll > 0.6) & (df.hit == 1)]
plots.convolveandplot(temp_df, k, k, variable='reward_side',
                      cluster_id=cluster_id, delay=delay, j=1,
                      alpha=0.3, colors=['grey','grey'], spikes=False)
# Correct trials in COLORRIGHT/COLORLEFT
temp_df = df.loc[(df.WM_roll > 0.6) & (df.hit == 0)]
plots.convolveandplot(temp_df, k, k, variable='reward_side',
                      cluster_id=cluster_id, delay=delay, j=1,
                      labels=['Right stimulus','Left stimulus'],
                      colors=[COLORRIGHT, COLORLEFT],
                      spikes=False, kernel=100)
k.set_xlim(-2, 14)
k.legend([False])
k.text(5, 40, 'Right Stimulus', ha='center', fontsize=5, color = COLORRIGHT)
k.text(5, 45, 'Right Left', ha='center', fontsize=5, color = COLORLEFT)

k.set_xlabel('Testing time from Stimulus onset (s)')
k.set_ylabel('Firing rate \n (spks/s)')
sns.despine(k)
# #######################################################################################
# Panels g / h: single-trial decoder traces (incorrect and correct example)
# #######################################################################################

path_repl = str(DATA_DIR / 'fig_5_ephys_repl') + '/'

T = 83
df_decoder = pd.read_csv(path_repl + 'fig4_panel_g_delay_decoder_output_trial83_incorrect_STM_session_E17.csv', index_col=0)
df_t       = pd.read_csv(path_repl + 'fig4_panel_g_spike_times_trial83_incorrect_STM_session_E17.csv', index_col=0)
big_data   = pd.read_csv(path_repl + 'fig4_panel_g_population_firing_rates_trial83_incorrect_STM_session_E17.csv', index_col=0)
plots.single_trial_with_decoder(df_t, df_decoder, big_data, 'E17_2022-02-02_17-13-06.csv', T, panels=[a1,a2,a3])
# Hide x-axis labels on top two subpanels so they appear joined
plt.setp(a1.get_xticklabels(), visible=False)
plt.setp(a2.get_xticklabels(), visible=False)
a1.set_xlabel('')
a2.set_xlabel('')
a1.set_title("Incorrect Right Stimulus trial (T83)", fontsize=6)
a2.xaxis.set_visible(False)
a1.xaxis.set_visible(False)
sns.despine(ax=a1, bottom=True)
sns.despine(ax=a2, bottom=True)
sns.despine(ax=a3)

T = 185
df_decoder = pd.read_csv(path_repl + 'fig4_panel_h_delay_decoder_output_trial185_correct_STM_session_E17.csv', index_col=0)
df_t       = pd.read_csv(path_repl + 'fig4_panel_h_spike_times_trial185_correct_STM_session_E17.csv', index_col=0)
big_data   = pd.read_csv(path_repl + 'fig4_panel_h_population_firing_rates_trial185_correct_STM_session_E17.csv', index_col=0)
plots.single_trial_with_decoder(df_t, df_decoder, big_data, 'E17_2022-02-02_17-13-06.csv', T,
                                panels=[b1,b2,b3], show_y=True)
plt.setp(b1.get_xticklabels(), visible=False)
plt.setp(b2.get_xticklabels(), visible=False)
b1.set_xlabel('')
b2.set_xlabel('')
b3.text(5, -10, 'Session E17_2022_02_02', ha='center', fontsize=5, color = 'grey')
b1.set_title("Correct Right Stimulus trial (T185)", fontsize=6)
b2.xaxis.set_visible(False)
b1.xaxis.set_visible(False)
sns.despine(ax=b1, bottom=True)
sns.despine(ax=b2, bottom=True)
sns.despine(ax=b3)
# #######################################################################################
# Panel i: stimulus-aligned single-trial log-odds heatmap and mean lineplot
# #######################################################################################

path_rev = str(DATA_DIR / 'fig_4_ephys_errors') + '/'
df = pd.read_csv(path_rev + 'panels_ij_single_trial_delay_decoder_log_odds_timeseries.csv', index_col=0)

aggregated_df = df.groupby(['trial','trial_change', 'times']).agg({'log_odds': 'mean'}).reset_index()


time_interval = [0.125, 0.375, 0.625, 0.875, 1.125, 1.375, 1.625, 1.875,
                     2.125, 2.375, 2.625, 2.875, 3.125, 3.375, 3.625, 3.875,
                     4.125, 4.375, 4.625, 4.875, 5.125, 5.375, 5.625, 5.875,
                     6.125, 6.375, 6.625, 6.875, 7.125, 7.375, 7.625, 7.875,
                     8.125, 8.375, 8.625, 8.875, 9.125, 9.375, 9.625, 9.875,
                     10.125]

search_df = aggregated_df[aggregated_df['times'].isin(time_interval)]

early_window = search_df['times'] < 1.5

trial_stats = (
    search_df
    .groupby('trial')
    .apply(lambda x: pd.Series({
        'early_mean': x.loc[early_window.loc[x.index], 'log_odds'].mean(),
    }))
)

valid_trials = trial_stats[
    (trial_stats['early_mean'] < 0)
].index

search_df = search_df[search_df['trial'].isin(valid_trials)]
aggregated_df = aggregated_df[aggregated_df['trial'].isin(valid_trials)]

def find_first_change_point(values, neg_count, pos_count):
    for i in range(len(values) - (neg_count + pos_count - 1)):
        if all(values[i:i+neg_count] < 0.25) and all(values[i+neg_count:i+neg_count+pos_count] > -0.25):
            return i + neg_count
    return None

change_points = []
for trial in search_df['trial_change'].unique():
    trial_data = search_df[search_df['trial_change'] == trial]
    values = trial_data['log_odds'].values
    point = find_first_change_point(values, 3, 3)
    if point is not None:
        change_points.append((trial, trial_data['times'].iloc[point]))

for trial_change, change_time in change_points:
    mask = (
        (aggregated_df['trial_change'] == trial_change) &
        (aggregated_df['times'] >= change_time) &
        (aggregated_df['times'] <= 10.375)
    )

    mask = (
        (aggregated_df['trial_change'] == trial_change) &
        (aggregated_df['times'] <= change_time) &
        (aggregated_df['times'] >= 0)
    )


sorted_trials = sorted(change_points, key=lambda x: x[1])
sorted_trial_change = [trial for trial, _ in sorted_trials]

heatmap_data = aggregated_df.pivot(index="trial_change", columns='times', values="log_odds")
heatmap_data = heatmap_data.loc[sorted_trial_change]

ax = i_heat
sns.heatmap(heatmap_data, center=0, vmin=-2, vmax=2, cmap='coolwarm', rasterized=True, ax=ax)

for trial, time in change_points:
    ax.scatter(heatmap_data.columns.get_loc(time) + 0.5, heatmap_data.index.get_loc(trial) + 0.5, color='white', s=8, edgecolors='black', linewidths=0.5)

ax.axvline(x=heatmap_data.columns.get_loc(0.125) - 0.75, color='black', linestyle='--')
ax.axvline(x=heatmap_data.columns.get_loc(10.375), color='black', linestyle='--')

y_ticks = np.arange(0, heatmap_data.shape[0], 20)
ax.set_yticks(y_ticks)
ax.set_yticklabels([f"{y}" for y in y_ticks])
ax.set_xlim(-2, 18)
ax.set_ylabel("Trials")
nice_times = [-2, 0, 2, 4, 6, 8, 10, 12, 14]
x_positions = [np.argmin(np.abs(heatmap_data.columns - t)) for t in nice_times]
cbar = ax.collections[0].colorbar
cbar.set_ticks([-2, -1, 0, 1, 2])
cbar.set_ticklabels(['-2', '-1', '0', '1', '2'])
cbar.set_label('Log Odds')
ax.set_xticks(x_positions)
ax.set_xticklabels([f"{t:}" for t in nice_times], rotation=360)
ax.set_xlabel("Time from stimulus onset (s)")
ax.xaxis.set_visible(False)
sns.despine(ax=i_line)

sns.lineplot(x='times', y='log_odds', data=aggregated_df, color='black',
             errorbar='ci', err_style='band', err_kws={"edgecolor": "none"}, ax=i_line)
i_line.axvline(x=0, color='black', linestyle='--')
i_line.axvline(x=10.4, color='black', linestyle='--')
i_line.axhline(0, linestyle='--', color='grey')
i_line.set_xlim(-2, 14)
i_line.set_ylabel("Log Odds")
i_line.set_xlabel("Time from stimulus onset (s)")
i_line.set_xticks(np.arange(-2, 15, 2))
i_line.set_ylim(-1, 1)

# #######################################################################################
# Panel j: reversal-aligned single-trial log-odds heatmap and mean lineplot
# #######################################################################################

df_change_points = pd.DataFrame(change_points, columns=['trial_change', 'time'])
new_df = aggregated_df.merge(df_change_points, on='trial_change', how='left')
new_df['time_reversal_aligned'] = new_df['times'] - new_df['time']

plot_df = new_df.copy()

simple_df = plot_df[['trial_change', 'time_reversal_aligned', 'log_odds']].dropna()
heatmap_data = simple_df.pivot(index="trial_change", columns='time_reversal_aligned', values="log_odds")
heatmap_data = heatmap_data.loc[sorted_trial_change]

ax = j_heat
sns.heatmap(heatmap_data, center=0, vmin=-2, vmax=2, cmap='coolwarm', rasterized=True, ax=ax)
sns.despine(ax=ax, bottom=True)

y_ticks = np.arange(0, heatmap_data.shape[0], 20)
ax.set_yticks(y_ticks)
ax.set_yticklabels([f"{y}" for y in y_ticks])
ax.set_ylabel("Trials")
nice_times = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]
x_positions = [np.argmin(np.abs(heatmap_data.columns - t)) for t in nice_times]

for i_row, trial in enumerate(heatmap_data.index):
    test = plot_df[plot_df.trial_change == trial]
    for t_mark in [0.125, 10.625]:
        rows = test.loc[test.times == t_mark, 'time_reversal_aligned']
        if not rows.empty:
            x_pos = heatmap_data.columns.get_loc(rows.iloc[0])
            ax.scatter(x_pos, i_row + 0.5, color='black', s=2, marker='|', linewidths=0.5)

ax.set_xticks(x_positions)
ax.set_xticklabels([f"{t:}" for t in nice_times], rotation=360)
ax.set_xlim(4, 92)
x0 = np.argmin(np.abs(heatmap_data.columns - 0))
y_positions_arr = np.arange(heatmap_data.shape[0])
ax.scatter(np.full_like(y_positions_arr, x0), y_positions_arr + 0.5, color='black', s=10)
cbar = ax.collections[0].colorbar
cbar.set_ticks([-2, -1, 0, 1, 2])
cbar.set_ticklabels(['-2', '-1', '0', '1', '2'])
ax.set_xlabel("Time from reversal (s)")
ax.xaxis.set_visible(False)

plot_df = new_df.loc[new_df.times.isin([0.125, 0.375, 0.625, 0.875, 1.125, 1.375, 1.625, 1.875,
                     2.125, 2.375, 2.625, 2.875, 3.125, 3.375, 3.625, 3.875,
                     4.125, 4.375, 4.625, 4.875, 5.125, 5.375, 5.625, 5.875,
                     6.125, 6.375, 6.625, 6.875, 7.125, 7.375, 7.625, 7.875,
                     8.125, 8.375, 8.625, 8.875, 9.125, 9.375, 9.625, 9.875,
                     10.125])]
simple_df = plot_df[['trial_change', 'time_reversal_aligned', 'log_odds']].dropna()
simple_df = simple_df.loc[(simple_df.time_reversal_aligned < 8) & (simple_df.time_reversal_aligned > -8)]

sns.lineplot(x='time_reversal_aligned', y='log_odds', data=simple_df, color='black',
             errorbar='ci', err_kws={"edgecolor": "none"}, ax=j_line)
sns.despine(ax=j_line)
j_line.axvline(x=0, color='black', linestyle='--')
j_line.axhline(0, linestyle='--', color='grey')
j_line.set_ylabel("Log Odds")
j_line.set_xlabel("Time from reversal (s)")
j_line.set_xticks(np.arange(-10, 12, 2))
j_line.set_xlim(-10, 10)
j_line.set_ylim(-1.25, 1.25)

# #######################################################################################
# Finalise: adjust layout and save / show figure
# #######################################################################################

plt.subplots_adjust(left=0.1, bottom=0.05, right=0.9, top=0.97,
                    wspace=0.35, hspace=0.75)
# plt.savefig(save_path+'/Fig_4_panel_revised.svg', bbox_inches='tight', dpi=300)
plt.show()
