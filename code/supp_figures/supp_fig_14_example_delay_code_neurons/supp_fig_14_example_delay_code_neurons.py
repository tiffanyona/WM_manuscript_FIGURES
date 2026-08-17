import json

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.ticker import MaxNLocator
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import ROOT, FIGURES_OUT, DATA_DIR
sys.path.insert(0, str(ROOT / 'src'))
from functions import COLORLEFT, COLORRIGHT

data_path  = str(DATA_DIR    / 'supp_figures' / 'supp_fig_14_example_delay_code_neurons')
final_path = str(FIGURES_OUT / 'supp_figures' / 'supp_fig_14_example_delay_code_neurons')


# #######################################################################################
# Sessions & trials
# #######################################################################################

sessions = [
    {'filename_base': 'E14_2021-04-02_12-53-42', 'trials': [57, 201, 137, 141]},
#   {'filename_base': 'E19_2022-01-14_14-42-13', 'trials': [261, 266, 284, 371]},
    {'filename_base': 'E17_2022-02-01_17-02-16', 'trials': [75, 77, 93, 118]},
    {'filename_base': 'E20_2022-02-14_16-01-30', 'trials': [88, 126, 139, 209]},
    {'filename_base': 'E22_2022-01-14_16-50-37', 'trials': [207, 321, 332, 192]},
]


# #######################################################################################
# Plotting function
# #######################################################################################

def replot_multi_session_grid(
        data_path,
        sessions,
        save_path=None,
        show_neuron_ids=False,
        show_decoder=True):

    ncols         = max(len(s['trials']) for s in sessions)
    n_sessions    = len(sessions)
    n_subrows     = 3 if show_decoder else 2
    height_ratios = [2.5, 1.25, 1.25] if show_decoder else [2.5, 1.25]

    row_height  = 7 if show_decoder else 6
    gap_height  = 2.5
    fig_height  = n_sessions * row_height + n_sessions * gap_height

    fig = plt.figure(figsize=(ncols * 5.5, fig_height))

    outer_heights = []
    for _ in range(n_sessions):
        outer_heights.append(gap_height)
        outer_heights.append(row_height)

    outer_gs = gridspec.GridSpec(
        n_sessions * 2, 1,
        figure=fig,
        hspace=0.0,
        height_ratios=outer_heights,
    )

    session_info = []

    for session_idx in range(n_sessions):
        gap_row  = session_idx * 2
        plot_row = session_idx * 2 + 1

        ax_label = fig.add_subplot(outer_gs[gap_row, 0])
        ax_label.axis('off')
        ax_label.text(
            0.5, 0.5,
            sessions[session_idx]['filename_base'],
            ha='center', va='center',
            fontsize=12, fontweight='bold',
            transform=ax_label.transAxes,
            bbox=dict(facecolor='white', edgecolor='lightgrey',
                      boxstyle='round,pad=0.3', alpha=0.9),
        )

        inner_gs = GridSpecFromSubplotSpec(
            n_subrows, ncols,
            subplot_spec=outer_gs[plot_row, 0],
            hspace=0.08,
            wspace=0.35,
            height_ratios=height_ratios,
        )

        trial_axes = []
        for trial_idx in range(len(sessions[session_idx]['trials'])):
            ax_raster  = fig.add_subplot(inner_gs[0, trial_idx])
            ax_fr      = fig.add_subplot(inner_gs[1, trial_idx])
            ax_decoder = fig.add_subplot(inner_gs[2, trial_idx]) if show_decoder else None
            trial_axes.append((ax_raster, ax_fr, ax_decoder))

        session_info.append((inner_gs, trial_axes))

    # ── Populate each session ─────────────────────────────────────────────────
    for session_idx, session in enumerate(sessions):
        filename_base        = session['filename_base']
        trials_list          = session['trials']
        inner_gs, trial_axes = session_info[session_idx]

        for trial_idx, T in enumerate(trials_list):
            print(f"Loading {filename_base} trial {T}...")

            base = rf"{data_path}//{filename_base}_trial_{T}"
            with open(base + "_metadata.json", "r") as f:
                metadata = json.load(f)

            FR_ordered = pd.read_csv(base + "_FR_ordered.csv")
            dft        = pd.read_csv(base + "_spikes.csv")
            df_results = pd.read_csv(base + "_results.csv")
            df_decoder = pd.read_csv(base + "_decoder.csv") if show_decoder else None
            mean_all   = pd.read_csv(base + "_mean_all.csv", index_col=0)['firing']

            delay                     = metadata["delay"]
            stop                      = metadata["stop"]
            significant_right_neurons = np.array(metadata["significant_right_neurons"])
            significant_left_neurons  = np.array(metadata["significant_left_neurons"])
            markersize                = max(1, 30 / np.sqrt(len(FR_ordered)))

            ax_raster, ax_fr, ax_decoder = trial_axes[trial_idx]

            # Raster ──────────────────────────────────────────────────────────
            for _, row_data in FR_ordered.iterrows():
                cid    = row_data.cluster_id
                color  = COLORRIGHT if cid in significant_right_neurons else COLORLEFT
                spikes = dft.loc[dft.cluster_id == cid, 'a_Stimulus_ON'].values
                ax_raster.plot(
                    spikes,
                    np.repeat(row_data.raster_y, len(spikes)),
                    '|', markersize=markersize, color=color, markeredgewidth=0.5,
                )

            if show_neuron_ids:
                ax_raster.set_yticks(FR_ordered['raster_y'])
                ax_raster.set_yticklabels(FR_ordered['cluster_id'])

            ax_raster.set_yticks(np.arange(0, 11, 5))
            ax_raster.set_xlim(-2.5, stop)

            if trial_idx == 0:
                ax_raster.set_ylabel("Units")
                ax_fr.set_ylabel("FR (sp/s)")
                if show_decoder:
                    ax_decoder.set_ylabel("Log odds")
            else:
                ax_raster.set_ylabel("")
                ax_fr.set_ylabel("")
                if show_decoder:
                    ax_decoder.set_ylabel("")

            # Population FR ───────────────────────────────────────────────────
            for neurons_, color in [
                (significant_right_neurons, COLORRIGHT),
                (significant_left_neurons,  COLORLEFT),
            ]:
                d = (df_results[df_results.neuron.isin(neurons_)]
                     .groupby("time_centered").firing.mean())
                ax_fr.plot(d.index, d.values, color=color, linewidth=1.5)

            trace_right = (df_results[df_results.neuron.isin(significant_right_neurons)]
                           .groupby("time_centered").firing.mean())
            trace_left  = (df_results[df_results.neuron.isin(significant_left_neurons)]
                           .groupby("time_centered").firing.mean())

            y_max = max(
                mean_all.max(),
                trace_right.max() if not trace_right.empty else 0,
                trace_left.max()  if not trace_left.empty  else 0,
            ) + 2

            ax_fr.set_ylim(0, y_max)
            ax_fr.set_xlim(-2.5, stop)
            ax_fr.yaxis.set_major_locator(MaxNLocator(nbins='auto'))

            # Decoder ─────────────────────────────────────────────────────────
            if show_decoder:
                ax_decoder.plot(df_decoder["times"], df_decoder["real"],
                                color="black", linewidth=1.5)
                ax_decoder.hlines(0, -2.5, stop, linestyle=':', color='grey')
                ax_decoder.set_xlim(-2.5, stop)
                ax_decoder.set_xticks(np.arange(0, stop + 1, 5))

            # Grey event bars ─────────────────────────────────────────────────
            for ax in ([ax_raster, ax_fr, ax_decoder] if show_decoder else [ax_raster, ax_fr]):
                ax.axvspan(0, 0.38, color='grey', alpha=0.3,
                           zorder=0, linewidth=0, edgecolor='none')
                ax.axvspan(0.38 + delay, 0.38 + delay + 0.2, color='grey', alpha=0.3,
                           zorder=0, linewidth=0, edgecolor='none')

            # Cosmetics ───────────────────────────────────────────────────────
            ax_raster.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
            if show_decoder:
                ax_fr.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
                ax_decoder.set_xlabel("Time from stimulus onset (s)")
            else:
                ax_fr.set_xticks(np.arange(0, stop + 1, 5))

            side = "Right" if metadata['reward_side'] == '1.0' else "Left"
            ax_raster.set_title(f"T{T} | {side}")

            sns.despine(ax=ax_raster, bottom=True)
            sns.despine(ax=ax_fr, bottom=True if show_decoder else False)
            if show_decoder:
                sns.despine(ax=ax_decoder)

        # Hide unused columns
        for empty_col in range(len(trials_list), ncols):
            for k in range(n_subrows):
                fig.add_subplot(inner_gs[k, empty_col]).axis("off")

    if save_path:
        svg_path = rf"{save_path}//multi_session_trial_grid.svg"
        plt.savefig(svg_path, format='svg', dpi=300, bbox_inches='tight')
        print(f"Saved grid to {svg_path}")

    plt.show()
    plt.close()


# #######################################################################################
# Run
# #######################################################################################

replot_multi_session_grid(
    data_path=data_path,
    sessions=sessions,
    save_path=final_path,
    show_decoder=True,
    show_neuron_ids=False,
)
