# -*- coding: utf-8 -*-
COLORLEFT  = 'teal'
COLORRIGHT = '#FF8D3F'
COLOR_STM  = 'darkgreen'
COLOR_REPL = 'indigo'

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import pandas as pd
import numpy as np
import seaborn as sns
import sys

try:
    from rpy2.robjects.packages import importr
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    base     = importr('base')
    car      = importr('car')
    lme4     = importr('lme4')
    pandas2ri.activate()
    HAS_RPY2 = True
except Exception as e:
    print(f"rpy2 unavailable: {e}")
    HAS_RPY2 = False

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import ROOT, FIGURES_OUT, DATA_DIR
sys.path.insert(0, str(ROOT / 'src'))

from functions import convolveandplot
import functions as plots


# #########################################################################################
# Setup — imports, paths, and global style
# #########################################################################################

save_path = str(FIGURES_OUT / 'fig_5_ephys_repl') + '/'
Path(save_path).mkdir(parents=True, exist_ok=True)
path      = str(DATA_DIR   / 'fig_5_ephys_repl') + '/'
os.chdir(path)

cm = 1 / 2.54
sns.set_context('paper', rc={
    'axes.labelsize':   7,
    'lines.linewidth':  1,
    'lines.markersize': 3,
    'legend.fontsize':  7,
    'xtick.major.size': 1,
    'xtick.labelsize':  6,
    'ytick.major.size': 1,
    'ytick.labelsize':  6,
    'xtick.major.pad':  0,
    'ytick.major.pad':  0,
})


# #########################################################################################
# Figure layout and GridSpec
# #########################################################################################

fig = plt.figure(figsize=(25 * cm, 22 * cm))

outer = gridspec.GridSpec(
    2, 1, figure=fig,
    height_ratios=[3, 2],
    hspace=0.2,
    left=0.07, right=0.98, top=0.93, bottom=0.05,
)

top = gridspec.GridSpecFromSubplotSpec(
    3, 4, subplot_spec=outer[0],
    hspace=0.5, wspace=0.5,
    height_ratios=[1, 1, 1],
)

a  = fig.add_subplot(top[0, 0])
b  = fig.add_subplot(top[0, 1])
d_outer = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=top[0, 2:4], wspace=0)
d  = fig.add_subplot(d_outer[0, 0:3])   # only 3/4 width

c_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=top[1, 0:2], wspace=0.15)
c1 = fig.add_subplot(c_gs[0, 0])
c2 = fig.add_subplot(c_gs[0, 1])

e_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=top[2, 0:2], wspace=0.15)
e1 = fig.add_subplot(e_gs[0, 0])
e2 = fig.add_subplot(e_gs[0, 1])

f_gs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=top[1:3, 2], hspace=0.3)
f1 = fig.add_subplot(f_gs[0])
f2 = fig.add_subplot(f_gs[1])
f3 = fig.add_subplot(f_gs[2])

g_gs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=top[1:3, 3], hspace=0.3)
g1 = fig.add_subplot(g_gs[0])
g2 = fig.add_subplot(g_gs[1])
g3 = fig.add_subplot(g_gs[2])

bot = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[1], wspace=0.5)

h_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=bot[0, 0], height_ratios=[1, 3], hspace=0)
h1 = fig.add_subplot(h_gs[0])
h = fig.add_subplot(h_gs[1])

i_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=bot[0, 1], hspace=0.3, height_ratios=[3, 2])
i1 = fig.add_subplot(i_gs[0])
i2 = fig.add_subplot(i_gs[1])

j_outer = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=bot[0, 2:4], height_ratios=[2, 3], hspace=0)
j0 = fig.add_subplot(j_outer[0])
j_gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=j_outer[1], wspace=0.08)
j1 = fig.add_subplot(j_gs[0])
j2 = fig.add_subplot(j_gs[1])
j3 = fig.add_subplot(j_gs[2])


# #########################################################################################
# Helper functions
# #########################################################################################

def label_panel(ax, letter, x=-0.25, y=1.25):
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=9, fontweight='bold', va='top', ha='left')

def sig_bar(ax, x0, x1, lw=1.5):
    ax.annotate(
        '', xy=(x1, 1.02), xytext=(x0, 1.02),
        xycoords=('data', 'axes fraction'),
        textcoords=('data', 'axes fraction'),
        arrowprops=dict(arrowstyle='-', color='black', lw=lw),
        annotation_clip=False,
    )

def hide_inner(left_ax, right_ax):
    left_ax.spines['right'].set_visible(False)
    right_ax.spines['left'].set_visible(False)
    right_ax.tick_params(left=False, labelleft=False)

def hide_yaxis(ax):
    ax.set_ylabel('')
    ax.tick_params(left=False, labelleft=False)
    ax.spines['left'].set_visible(False)

def no_right_spine(ax):
    ax.spines['right'].set_visible(False)

def no_left_spine(ax):
    ax.spines['left'].set_visible(False)
    ax.tick_params(left=False, labelleft=False)
    ax.set_ylabel('')


# #########################################################################################
# Panel a: stimulus-aligned decoding (STM vs RepL)
# #########################################################################################

y_range = [-0.1, 0.45]

variable  = 'WM_roll_1'
df_sti    = pd.read_csv(path + 'panel_a_stimulus_aligned_choice_decoding_STM_vs_RepL.csv', index_col=0)
df_sti    = df_sti.loc[df_sti.trial_type == variable]
df_shuf   = pd.read_csv(path + 'panel_a_stimulus_aligned_choice_decoding_STM_shuffle.csv', index_col=0)
sess      = df_shuf.session.unique()
df_sti    = df_sti[df_sti.session.isin(sess)]
plots.plot_results_session_summary(fig, a, df_sti, [COLOR_STM], [variable], shuffle_df=df_shuf,
    y_range=y_range, x_range=[-2, 1.3], baseline=0)

variable  = 'RL_roll_1'
df_sti    = pd.read_csv(path + 'panel_a_stimulus_aligned_choice_decoding_STM_vs_RepL.csv', index_col=0)
df_sti    = df_sti.loc[df_sti.trial_type == variable]
df_shuf2  = pd.read_csv(path + 'panel_a_stimulus_aligned_choice_decoding_RepL_shuffle.csv', index_col=0)
df_shuf2  = df_shuf2[df_shuf2.session.isin(sess)]
plots.plot_results_session_summary(fig, a, df_sti, [COLOR_REPL], [variable], shuffle_df=df_shuf2,
    y_range=y_range, x_range=[-2, 1.3], baseline=0)

a.set_title('Stimulus code', fontsize=7, fontweight='bold')
a.set_ylabel('Excess decoding\naccuracy')
label_panel(a, 'a')


# #########################################################################################
# Panel b: response-aligned decoding (STM vs RepL)
# #########################################################################################

variable = 'WM_roll_1'
df_res   = pd.read_csv(path + 'panel_b_response_aligned_choice_decoding_STM_vs_RepL.csv', index_col=0)
df_res   = df_res.loc[df_res.trial_type == variable]
df_shuf  = pd.read_csv(path + 'panel_b_response_aligned_choice_decoding_STM_shuffle.csv', index_col=0)
plots.plot_results_session_summary(fig, b, df_res, [COLOR_STM], [variable], shuffle_df=df_shuf,
    y_range=y_range, x_range=[-1, 3], baseline=0)

variable = 'RL_roll_1'
df_res   = pd.read_csv(path + 'panel_b_response_aligned_choice_decoding_STM_vs_RepL.csv', index_col=0)
df_res   = df_res.loc[df_res.trial_type == variable]
df_shuf  = pd.read_csv(path + 'panel_b_response_aligned_choice_decoding_RepL_shuffle.csv', index_col=0)
plots.plot_results_session_summary(fig, b, df_res, [COLOR_REPL], [variable], shuffle_df=df_shuf,
    y_range=y_range, x_range=[-1, 3], baseline=0)

b.set_title('Response code', fontsize=7, fontweight='bold')
b.set_xlabel('Time from Go cue onset (s)')
hide_yaxis(b)
label_panel(b, 'b')


# #########################################################################################
# Panel c: delay code decoding (STM vs RepL, stimulus and response aligned)
# #########################################################################################

for variable, color in [('WM_roll_1', COLOR_STM), ('RL_roll_1', COLOR_REPL)]:
    df_sti  = pd.read_csv(path + 'panel_c_stimulus_aligned_delay_decoding_STM_vs_RepL.csv', index_col=0)
    df_res  = pd.read_csv(path + 'panel_c_response_aligned_delay_decoding_STM_vs_RepL.csv', index_col=0)
    df_sti  = df_sti.loc[df_sti.trial_type == variable]
    df_res  = df_res.loc[df_res.trial_type == variable]
    shuf_sti = 'panel_c_stimulus_aligned_delay_decoding_STM_shuffle.csv' if variable == 'WM_roll_1' else 'panel_c_stimulus_aligned_delay_decoding_RepL_shuffle.csv'
    shuf_res = 'panel_c_response_aligned_delay_decoding_STM_shuffle.csv' if variable == 'WM_roll_1' else 'panel_c_response_aligned_delay_decoding_RepL_shuffle.csv'
    df_ss   = pd.read_csv(path + shuf_sti, index_col=0)
    df_rs   = pd.read_csv(path + shuf_res, index_col=0)
    plots.plot_results_session_summary(fig, c1, df_sti, [color], [variable], shuffle_df=df_ss,
        y_range=y_range, x_range=[-2, 1.3], baseline=0)
    plots.plot_results_session_summary(fig, c2, df_res, [color], [variable], shuffle_df=df_rs,
        y_range=y_range, x_range=[-1, 3], baseline=0)

c1.set_title('Delay code', fontsize=7, fontweight='bold')
c1.set_ylabel('Excess decoding\naccuracy')
c1.set_xlabel('Time to stimulus onset (s)')
c2.set_xlabel('Time from Go cue onset (s)')
label_panel(c1, 'c')
hide_yaxis(c2)
hide_inner(c1, c2)

c2.text(0.95, 0.95, 'STM',  transform=c2.transAxes, ha='right', va='top',
        fontsize=6, color=COLOR_STM,  fontweight='bold')
c2.text(0.95, 0.82, 'RepL', transform=c2.transAxes, ha='right', va='top',
        fontsize=6, color=COLOR_REPL, fontweight='bold')

grey_y_bot = y_range[0]
grey_y_top = grey_y_bot + 0.03
a.fill_betweenx( [grey_y_bot, grey_y_top], 0, 0.35, color='lightgrey', alpha=1, linewidth=0, zorder=0)
b.fill_betweenx( [grey_y_bot, grey_y_top], 0, 0.2,  color='lightgrey', alpha=1, linewidth=0, zorder=0)
c1.fill_betweenx([grey_y_bot, grey_y_top], 0, 0.35, color='lightgrey', alpha=1, linewidth=0, zorder=0)
c2.fill_betweenx([grey_y_bot, grey_y_top], 0, 0.2,  color='lightgrey', alpha=1, linewidth=0, zorder=0)


# #########################################################################################
# Panel d: ramp code decoding (delay vs pre-stimulus probability)
# #########################################################################################

baseline_d = 0.5
df_ramp = pd.read_csv(path + 'panel_d_delay_vs_prestimulus_ramp_probability_STM_vs_RepL.csv')

plots.plot_results_session_summary(None, d, df_ramp, [COLOR_STM, COLOR_REPL],
                                   ['WM_roll_1', 'RL_roll_1'],
                                   y_range=[0, 1.0],
                                   x_range=[-2, 10],
                                   epoch='Stimulus_ON',
                                   baseline=baseline_d)

d.set_title('Ramp code', fontsize=7, fontweight='bold')
d.set_ylabel('Prob(Delay vs pre-stim)')
d.set_xlabel('Time to stimulus onset (s)')
d.set_xlim(-2, 10)
d.set_ylim(0, 1.0)
label_panel(d, 'd', x=-0.08)


# #########################################################################################
# Panel e: example session single-trial decoding (stimulus and response aligned)
# #########################################################################################

df_sti_e  = pd.read_csv(path + 'panel_e_example_session_stimulus_aligned_delay_decoding.csv',         index_col=0)
df_res_e  = pd.read_csv(path + 'panel_e_example_session_response_aligned_delay_decoding.csv',         index_col=0)
df_ss_e   = pd.read_csv(path + 'panel_e_example_session_stimulus_aligned_delay_decoding_shuffle.csv', index_col=0)
df_rs_e   = pd.read_csv(path + 'panel_e_example_session_response_aligned_delay_decoding_shuffle.csv', index_col=0)

for color, variable in [(COLOR_STM,'WM_roll_1'), (COLOR_REPL,'RL_roll_1')]:
    real   = np.array(np.mean(
        df_sti_e.loc[df_sti_e.trial_type==variable]
        .groupby('session').median(numeric_only=True)
        .drop(columns=['score','fold'])))
    times  = np.array(
        df_sti_e.loc[df_sti_e.trial_type==variable]
        .drop(columns=['session','fold','score','subject','trial_type'])
        .columns.astype(float))
    df_new = pd.DataFrame()
    for it in np.arange(1,100):
        try:    df_new[it] = df_ss_e.loc[df_ss_e.trial_type==variable].groupby('times').mean(numeric_only=True)[float(it)]
        except: df_new[it] = df_ss_e.loc[df_ss_e.trial_type==variable].groupby('times').mean(numeric_only=True)[str(float(it))]
    ym = df_new.mean(axis=1).values
    up = df_new.quantile(0.975,interpolation='linear',axis=1)-ym
    lo = df_new.quantile(0.025,interpolation='linear',axis=1)-ym
    e1.plot(times, real, color=color)
    e1.fill_between(times, lo+real, up+real, alpha=0.2, color=color, linewidth=0)
    e1.axhline(0, linestyle=':', color='black')
    e1.set_ylim(-0.2, 0.5)

    real2  = np.array(np.mean(
        df_res_e.loc[df_res_e.trial_type==variable]
        .groupby('session').median(numeric_only=True)
        .drop(columns=['score','fold'])))
    times2 = np.array(
        df_res_e.loc[df_res_e.trial_type==variable]
        .drop(columns=['session','fold','score','subject','trial_type'])
        .columns.astype(float))
    df_new2 = pd.DataFrame()
    for it in np.arange(1,100):
        try:    df_new2[it] = df_rs_e.loc[df_rs_e.trial_type==variable].groupby('times').mean(numeric_only=True)[it]
        except: df_new2[it] = df_rs_e.loc[df_rs_e.trial_type==variable].groupby('times').mean(numeric_only=True)[str(float(it))]
    ym2 = df_new2.mean(axis=1).values
    up2 = df_new2.quantile(0.975,interpolation='linear',axis=1)-ym2
    lo2 = df_new2.quantile(0.025,interpolation='linear',axis=1)-ym2
    e2.plot(times2, real2, color=color)
    e2.fill_between(times2, lo2+real2, up2+real2, alpha=0.2, color=color, linewidth=0)
    e2.axhline(0, linestyle=':', color='grey')
    e2.set_ylim(-0.2, 0.5)

e_grey_bot = -0.2
e_grey_top = 0.5
e1.fill_betweenx([e_grey_bot, e_grey_top], 0, 0.35, color='lightgrey', alpha=1, linewidth=0, zorder=0)
e2.fill_betweenx([e_grey_bot, e_grey_top], 0, 0.2,  color='lightgrey', alpha=1, linewidth=0, zorder=0)

e1.set_xlabel('Time from stimulus onset (s)')
e1.set_ylabel('Excess decoding\naccuracy')
e1.set_xlim(-2, 1)
e2.set_xlabel('Time from Go cue onset (s)')
e2.set_xlim(-1, 2)
e1.text(0.04, 0.05, 'Session E20_2022_02_26', transform=e1.transAxes, fontsize=5, color='grey')
hide_yaxis(e2)
hide_inner(e1, e2)
label_panel(e1, 'e')


# #########################################################################################
# Panels f & g: single-trial raster and decoder (STM and RepL example trials)
# #########################################################################################

T = 223
plots.single_trial_with_decoder(
    pd.read_csv(path + 'panel_f_spike_times_trial223_correct_STM.csv',    index_col=0),
    pd.read_csv(path + 'panel_f_delay_decoder_output_trial223_correct_STM.csv',   index_col=0),
    pd.read_csv(path + 'panel_f_population_firing_rates_trial223_correct_STM.csv', index_col=0),
    'E17_2022-01-31_16-30-44.csv', T, panels=[f1, f2, f3])
f1.set_title('Correct Right stimulus STM trial\n(trial 223)', fontsize=5.5, pad=2)
f3.set_ylim(-10, 10)
label_panel(f1, 'f')
f1.tick_params(labelbottom=False); f1.set_xlabel('')
f2.tick_params(labelbottom=False); f2.set_xlabel('')
f1.set_xticks([0, 5, 10]); f2.set_xticks([0, 5, 10]); f3.set_xticks([0, 5, 10])
for fax in [f1, f2, f3]:
    fax.axvspan(0, 0.35,  color='lightgrey', alpha=0.8, linewidth=0, zorder=0)
    fax.axvspan(10.4,10.6,color='lightgrey', alpha=0.8, linewidth=0, zorder=0)
# neuron labels only on f2 (left plot), in flat region
f2.text(0.40, 0.90, 'Right prefering neurons', transform=f2.transAxes, fontsize=5, color=COLORRIGHT)
f2.text(0.40, 0.55, 'Left prefering neurons',  transform=f2.transAxes, fontsize=5, color=COLORLEFT)

T = 21
plots.single_trial_with_decoder(
    pd.read_csv(path + 'panel_g_spike_times_trial21_RepL.csv',    index_col=0),
    pd.read_csv(path + 'panel_g_delay_decoder_output_trial21_RepL.csv',   index_col=0),
    pd.read_csv(path + 'panel_g_population_firing_rates_trial21_RepL.csv', index_col=0),
    'E17_2022-01-31_16-30-44.csv', T, panels=[g1, g2, g3])
g1.set_title('Correct Right stimulus RepL trial\n(trial 21)', fontsize=5.5, pad=2)
g3.set_ylim(-10, 10)
label_panel(g1, 'g')
g1.tick_params(labelbottom=False); g1.set_xlabel('')
g2.tick_params(labelbottom=False); g2.set_xlabel('')
g1.set_xticks([0, 5, 10]); g2.set_xticks([0, 5, 10]); g3.set_xticks([0, 5, 10])
for gax in [g1, g2, g3]:
    gax.axvspan(0, 0.35,  color='lightgrey', alpha=0.8, linewidth=0, zorder=0)
    gax.axvspan(10.4,10.6,color='lightgrey', alpha=0.8, linewidth=0, zorder=0)


# #########################################################################################
# Panel h: epoch-split decoding log-odds (early vs late delay, STM vs RepL)
# #########################################################################################

df_final   = pd.read_csv(path + 'panel_h_delay_decoder_log_odds_by_epoch_STM_vs_RepL.csv', index_col=0)
df_results = df_final.groupby(['session','trial_type','epoch']).log_odds.mean().reset_index()

positions    = [-0.25, 0.2, 0.75, 1.2]
group_colors = [COLOR_STM, '#90EE90', COLOR_REPL, '#B39DDB']
groups = [('WM_roll_1','early'),('WM_roll_1','late'),('RL_roll_1','early'),('RL_roll_1','late')]
for (tt, ep), pos, col in zip(groups, positions, group_colors):
    sub = df_results.loc[(df_results.trial_type==tt)&(df_results.epoch==ep), 'log_odds']
    h.boxplot(sub, positions=[pos], widths=0.35, patch_artist=True,
              showmeans=False, showfliers=False,
              medianprops=dict(color='white', linewidth=1.5),
              boxprops=dict(facecolor=col, alpha=0.8, linewidth=0),
              whiskerprops=dict(color='black', linewidth=0.7),
              capprops=dict(color='black', linewidth=0.7))
    dot_col = COLOR_STM if tt == 'WM_roll_1' else COLOR_REPL
    h.scatter(np.random.normal(pos, 0.05, len(sub)), sub,
              color=dot_col, alpha=0.8, s=8, zorder=5,
              edgecolors='white', linewidths=0.3)

h.set_xlim(-0.6, 1.5)
h.set_ylim(-1, 6)
h.axhline(0, linestyle=':', color='grey')
h.set_ylabel('Decoding accuracy\n(log odds)')
h.set_xlabel('')
h.set_xticks(positions)
h.set_xticklabels(['Early','Late','Early','Late'], fontsize=5)
h.text(0.02, 0.97, 'RepL trials', transform=h.transAxes, va='top', fontsize=6, color=COLOR_REPL, fontweight='bold')
h.text(0.02, 0.86, 'STM trials',  transform=h.transAxes, va='top', fontsize=6, color=COLOR_STM,  fontweight='bold')
label_panel(h1, 'h')
h1.axis('off')

if HAS_RPY2:
    formula = "log_odds ~ state*epoch + (1|session:fold)"
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_df = ro.conversion.py2rpy(df_final.reset_index(drop=True))
    model = lme4.lmer(formula, data=r_df)
    print(base.summary(model))


# #########################################################################################
# Panel i: example neuron firing rate by WM state
# #########################################################################################

df_neuron = pd.read_csv(path + 'panel_i_neuron138_spike_times_all_10s_delay_trials.csv')
temp_df   = df_neuron.loc[df_neuron.cluster_id==138].copy()
temp_df['state'] = np.where(temp_df['WM_roll']>0.6, 1, 0)

j_val = convolveandplot(
    temp_df.loc[(temp_df.vector_answer==0)&(temp_df.hit==1)],
    i1, i2, variable='state', cluster_id=138, delay=10, j=1,
    labels=['WM left','HB left'], colors=[COLOR_STM, COLOR_REPL],
    kernel=200, add_state=True)
i1.set_ylim(0, 19)
i1.set_xlim(-2, 12); i1.set_xticks([0, 5, 10])
i1.axvspan(0,    0.35, color='lightgrey', alpha=0.8, linewidth=0, zorder=0)
i1.axvspan(10.4, 10.6, color='lightgrey', alpha=0.8, linewidth=0, zorder=0)
i1.set_title('Cluster 138', fontsize=6, color='grey', pad=2)
i1.tick_params(labelbottom=False)
i2.set_xlim(-2, 12); i2.set_xticks([0, 5, 10])
i2.axvspan(0,    0.35, color='lightgrey', alpha=0.8, linewidth=0, zorder=0)
i2.axvspan(10.4, 10.6, color='lightgrey', alpha=0.8, linewidth=0, zorder=0)
i2.set_title('Session_E22_2022-01-13', fontsize=5, color='grey', pad=2)
try:    i2.get_legend().remove()
except: pass
try:    i1.get_legend().remove()
except: pass
label_panel(i1, letter="i",y=1.05)


# #########################################################################################
# Panel j: previous-response decoding across trial epochs (j1 prev-Go, j2 Stim, j3 Go)
# #########################################################################################

y_range_j = [-0.1, 0.45]

# j1
variable = 'WM_roll_1'
df_sj  = pd.read_csv(path + 'panel_j_stimulus_aligned_previous_response_decoding_STM.csv', index_col=0)
df_sjs = pd.read_csv(path + 'panel_j_stimulus_aligned_previous_response_decoding_STM_shuffle.csv', index_col=0)
df_sjs['trial_type'] = variable
sess_js = df_sjs.session.unique()
df_sj   = df_sj[df_sj.session.isin(sess_js)]
plots.plot_results_session_summary(fig, j2, df_sj, [COLOR_STM], [variable], shuffle_df=df_sjs,
    y_range=y_range_j, x_range=[-4, 1], baseline=0)

variable = 'RL_roll_1'
df_sj2  = pd.read_csv(path + 'panel_j_stimulus_aligned_previous_response_decoding_RepL.csv', index_col=0)
df_sjs2 = pd.read_csv(path + 'panel_j_stimulus_aligned_previous_response_decoding_RepL_shuffle.csv', index_col=0)
df_sjs2['trial_type'] = variable
df_sj2  = df_sj2[df_sj2.session.isin(sess_js)]
plots.plot_results_session_summary(fig, j2, df_sj2, [COLOR_REPL], [variable], shuffle_df=df_sjs2,
    y_range=y_range_j, x_range=[-4, 1], baseline=0)

# j2 — data already pre-subtracted, plot directly without shuffle
df_ss1 = pd.read_csv(path + 'panel_j_gocue_previous_response_decoding_STM_shuffle_subtracted.csv', index_col=0)
plots.plot_results_session_summary(fig, j1, df_ss1, colors=[COLOR_STM],
    variables_combined=['WM_roll_1'], y_range=y_range_j, x_range=[-1, 4], baseline=0)

df_ss2 = pd.read_csv(path + 'panel_j_gocue_previous_response_decoding_RepL_shuffle_subtracted.csv', index_col=0)
plots.plot_results_session_summary(fig, j1, df_ss2, colors=[COLOR_REPL],
    variables_combined=['RL_roll_1'], y_range=y_range_j, x_range=[-1, 4], baseline=0)

# j3
variable = 'WM_roll_1'
df_rj  = pd.read_csv(path + 'panel_j_gocue_aligned_previous_response_decoding_STM.csv', index_col=0)
df_rjs = pd.read_csv(path + 'panel_j_gocue_aligned_previous_response_decoding_STM_shuffle.csv', index_col=0)
df_rjs['trial_type'] = variable
df_rj  = df_rj[df_rj.session.isin(sess_js)]
plots.plot_results_session_summary(fig, j3, df_rj, [COLOR_STM], [variable], shuffle_df=df_rjs,
    y_range=y_range_j, x_range=[-1, 3], baseline=0)

variable = 'RL_roll_1'
df_rj2  = pd.read_csv(path + 'panel_j_gocue_aligned_previous_response_decoding_RepL.csv', index_col=0)
df_rjs2 = pd.read_csv(path + 'panel_j_gocue_aligned_previous_response_decoding_RepL_shuffle.csv', index_col=0)
df_rjs2['trial_type'] = variable
df_rj2  = df_rj2[df_rj2.session.isin(sess_js)]
plots.plot_results_session_summary(fig, j3, df_rj2, [COLOR_REPL], [variable], shuffle_df=df_rjs2,
    y_range=y_range_j, x_range=[-1, 3], baseline=0)

j1.set_xlabel('Time from Go (s)')
j1.set_ylabel('Excess decoding\naccuracy')
j1.set_xlim(-1, 4)
j2.set_xlabel('Time from Stim. (s)')
j2.set_xlim(-4, 1)
j3.set_xlabel('Time from Go (s)')
j3.set_xlim(-1, 3)
no_right_spine(j1); no_left_spine(j2); no_right_spine(j2); no_left_spine(j3)
j1.xaxis.get_major_ticks()[-1].label1.set_visible(False)
j2.xaxis.get_major_ticks()[0].label1.set_visible(False)
j2.xaxis.get_major_ticks()[-1].label1.set_visible(False)
j3.xaxis.get_major_ticks()[0].label1.set_visible(False)
label_panel(j0, 'j', x=-0.12)
j0.axis('off')


# #########################################################################################
# Significance bars
# #########################################################################################

sig_bar(a,  0, 0.4)
sig_bar(b,  0, 0.2)
sig_bar(c1, 0, 0.4)
sig_bar(c2, 0, 0.2)


# #########################################################################################
# Final polish — despine, align axes, spine cleanup
# #########################################################################################

sns.despine()

# After despine: align b, c2, e2 to same height/y0 as a, c1, e1
fig.canvas.draw()
for left_ax, right_ax in [(a, b), (c1, c2), (e1, e2)]:
    lpos = left_ax.get_position()
    rpos = right_ax.get_position()
    right_ax.set_position([rpos.x0, lpos.y0, rpos.width, lpos.height])

for ax in [b, c2, e2, j2, j3]:
    ax.spines['left'].set_visible(False)
    ax.yaxis.set_visible(False)
for ax in [c1, e1, j1]:
    ax.spines['right'].set_visible(False)
j2.spines['right'].set_visible(False)

# j ylims: push data down, leave room for labels at top
for jax in [j1, j2, j3]:
    jax.set_ylim(-0.12, 0.65)
    jax.set_yticks([0, 0.2, 0.4])

for ax in [f1, f2, g1, g2]:
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(bottom=False)

# i1: no bottom spine (despine style, keep ticks)
i1.spines['bottom'].set_visible(False)

b.spines['left'].set_visible(False)
b.tick_params(left=False, labelleft=False)


# #########################################################################################
# Panel j annotations — arrows, labels, and trial boundary line
# #########################################################################################

fig.canvas.draw()


j1.annotate('', xy=(1.0, 1.22), xytext=(0.0, 1.22),
    xycoords='axes fraction', textcoords='axes fraction',
    arrowprops=dict(arrowstyle='<->', color='black', lw=0.8),
    annotation_clip=False)
j1.text(0.5, 1.225, 'trial  t −1', transform=j1.transAxes,
    ha='center', va='bottom', fontsize=7, style='italic')

l = fig.transFigure.inverted().transform(j2.transAxes.transform([0.0, 1.22]))
r = fig.transFigure.inverted().transform(j3.transAxes.transform([1.0, 1.22]))
j2.annotate('', xy=(r[0], r[1]), xytext=(l[0], l[1]),
    xycoords='figure fraction', textcoords='figure fraction',
    arrowprops=dict(arrowstyle='<->', color='black', lw=0.8),
    annotation_clip=False)
fig.text((l[0]+r[0])/2, l[1]+0.005, 'trial  t',
    ha='center', va='bottom', fontsize=7, style='italic',
    transform=fig.transFigure)


lj = fig.transFigure.inverted().transform(j1.transAxes.transform([0.0, 1.02]))
rj = fig.transFigure.inverted().transform(j3.transAxes.transform([1.0, 1.02]))
fig.text((lj[0]+rj[0])/2, lj[1]+0.06, 'Previous response decoding',
    ha='center', va='bottom', fontsize=8, fontweight='bold',
    transform=fig.transFigure)

j1.axvline(x=j1.get_xlim()[1], ymin=-0.15, ymax=1.20,
    color='black', linestyle='--', linewidth=0.7, clip_on=False)

j1.axvline(x=0, color='lightgrey', linestyle='--', linewidth=0.5, zorder=1)
j2.axvline(x=0, color='lightgrey', linestyle='--', linewidth=0.5, zorder=1)
j2.axvline(x=0.4, color='lightgrey', linestyle='--', linewidth=0.5, zorder=1)
j3.axvline(x=0, color='lightgrey', linestyle='--', linewidth=0.5, zorder=1)

j1.text(0.04, 0.98, 'Go cue',             transform=j1.transAxes, fontsize=6, color='grey', va='top', clip_on=True)
j1.text(0.22, 0.98, 'Resp. window + ITI', transform=j1.transAxes, fontsize=6, color='grey', va='top', clip_on=True)
j2.text(0.04, 0.98, 'Pre-stim.',          transform=j2.transAxes, fontsize=6, color='grey', va='top', clip_on=True)
j2.text(0.68, 0.98, 'Stim.',              transform=j2.transAxes, fontsize=6, color='grey', va='top', clip_on=True)
j3.text(0.02, 0.98, 'Delay',              transform=j3.transAxes, fontsize=6, color='grey', va='top', clip_on=True)
j3.text(0.38, 0.98, 'Go cue',             transform=j3.transAxes, fontsize=6, color='grey', va='top', clip_on=True)
j3.text(0.58, 0.98, 'Resp. window + ITI', transform=j3.transAxes, fontsize=6, color='grey', va='top', clip_on=True)

# plt.savefig(save_path + 'fig_5_ephys_repl_v5.svg', bbox_inches='tight', dpi=300)
# plt.savefig(save_path + 'fig_5_ephys_repl_v5.png', bbox_inches='tight', dpi=300)
plt.show()