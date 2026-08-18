# Working Memory Manuscript — Repository Wiki

**Episodic recruitment of attractor dynamics in frontal cortex reveals distinct mechanisms for forgetting and lack of cognitive control in short-term memory**

Oña-Jodar T, Prat-Ortega G, Pastor-Ciurana J, Li C, Dalmau J, Compte A\*, de la Rocha J\* — *Nature Neuroscience* (in revision)
Preprint: [bioRxiv 2024.02.18.579447](https://www.biorxiv.org/content/10.1101/2024.02.18.579447v1) · Repository: [github.com/tiffanyona/WM_manuscript_FIGURES](https://github.com/tiffanyona/WM_manuscript_FIGURES) · Contact: tiffany.ona@alleninstitute.org

---

## Overview

This repository contains the code to reproduce all main and supplementary figures in the manuscript from pre-processed data files. Each figure has its own subfolder under `code/` with a self-contained script or notebook. Running a script writes the corresponding outputs to `figures/`.

---

## Repository structure

```
WM_manuscript_FIGURES/
├── code/
│   ├── fig_1_behavior/        # Fig 1 — behavior, autocorrelations, GLMM weights
│   ├── fig_2_model/           # Fig 2 — HMM / drift-walk behavioral model fits
│   ├── fig_3_ephys_wm/        # Fig 3 — cross-temporal decoding during delay
│   ├── fig_4_ephys_errors/    # Fig 4 — decoding on error trials, log-odds trajectories
│   ├── fig_5_ephys_repl/      # Fig 5 — STM vs. RepL decoder generalization
│   ├── fig_6_synch/           # Fig 6 — population synchrony, PSDs, oscillations
│   ├── supp_figures/          # Supplementary figures (supp_fig_1 … supp_fig_22)
│   └── table_animals.py       # Animal summary table
├── data/                      # Pre-processed CSVs, one folder per figure
├── figures/                   # Rendered outputs (PNG / SVG / PDF)
├── src/
│   ├── functions.py           # Shared plotting utilities and stats helpers
│   └── trial_examples.py      # Trial-level visualization helpers
├── config.py                  # Path constants (DATA_DIR, FIGURES_OUT, ANALYSIS_DATA)
├── config.toml                # Submission-specific output paths
└── pyproject.toml             # Project metadata and pinned dependencies
```

---

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for environment management. Python 3.9+ is required.

```bash
# Install uv if you don't have it
pip install uv

# Create the virtual environment and install all dependencies
uv sync
```

Activate the environment:

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### R dependency

Some analyses (GLMM in Fig 1, `data/fig_1_behavior/R-code for GLM.r`) require R and use `rpy2` to call R from Python. Install R separately and ensure it is on your `PATH` before running those scripts.

---

## Configuration

`config.py` defines the three key paths used by every script:

| Variable | Default (relative to repo root) | Purpose |
|---|---|---|
| `DATA_DIR` | `./data/` | Pre-processed panel CSVs |
| `FIGURES_OUT` | `./figures/` | Rendered figure outputs |
| `ANALYSIS_DATA` | `../general_data/` | Decoded session files not bundled in `data/` |

If your data lives elsewhere (e.g. on a different drive), edit the `ANALYSIS_DATA` line in `config.py`.

---

## Data

Pre-processed data files (`data/`) are **not included in this repository**. They are distributed separately through this Open Research Framework (ORF) publication.

> **Important:** Scripts read data files immediately on startup. If the data files are missing, the script will crash with a `FileNotFoundError`. Download and place the data before running any script.

Download the data from this ORF publication, then place the extracted `data/` folder at the repository root so the structure matches what `config.py` expects:

```
WM_manuscript_FIGURES/
├── data/
│   ├── fig_1_behavior/
│   ├── fig_2_model/
│   ├── fig_3_ephys_wm/
│   ├── fig_4_ephys_errors/
│   ├── fig_5_ephys_repl/
│   ├── fig_6_synch/
│   └── supp_figures/
```

---

## Reproducing figures

Each `code/fig_N_*/` folder contains one primary script (`.py`) or notebook (`.ipynb`). Run scripts from the repository root so that relative imports resolve correctly.

```bash
# Example — Fig 3
python code/fig_3_ephys_wm/fig_3_panel.py

# Example — Supplementary figure (notebook)
jupyter notebook code/supp_figures/supp_fig_1_tau_exponential/supp_fig_1_tau_exponential.ipynb
```

Output files are written to the corresponding subfolder under `figures/`. All output directories are created automatically when the script runs.

---

## Dependencies

Key packages (all pinned in `pyproject.toml`):

| Package | Version | Role |
|---|---|---|
| numpy | 1.26.4 | Numerical computing |
| pandas | 1.5.3 | Data wrangling |
| matplotlib | 3.10.3 | Plotting |
| scipy | 1.16.0 | Statistics, signal processing |
| scikit-learn | 1.7.0 | Decoding / classification |
| neo + elephant | 0.14.5 / 1.1.1 | Spike-train analysis |
| rpy2 | 3.6.7 | R integration (GLMM) |
| seaborn | 0.13.2 | Statistical plots |
| statsmodels | 0.14.6 | Statistical modeling |
| pingouin | 0.5.5 | Statistical tests |

---

# Figure Reference Guide

This guide maps each panel of the six main figures to the pre-processed data file(s) that generate it. All files are distributed through this ORF publication and belong under `data/` in the repository root — see the README for the expected folder structure and setup instructions.

Files whose names begin with `panels_a-h_` or similar span multiple panels that share the same source. Where a data file lives in a different figure's folder (e.g., Fig. 4 panels g–h draw from `data/fig_5_ephys_repl/`), this is noted in the Source column. Schematics have no associated data file.

---

## Figure 1 — Behavioral characterization of delayed auditory working memory

`code/fig_1_behavior/` · `data/fig_1_behavior/`

*How does performance on a delayed go/no-go auditory task degrade with increasing delay, what history effects drive errors, and how do generalized linear model weights quantify the influence of prior stimuli and choices?*

| Panel | Content | Source file(s) |
|---|---|---|
| a | Task schematic | *Schematic — no data file* |
| b | Per-animal accuracy vs. delay (0, 1, 3, 10 s) — line plot | `panels_a-h_behavior_all_trials_all_animals.csv` |
| c | Mean ± SEM accuracy vs. delay with linear regression fit and reported slope/p-value | `panels_a-h_behavior_all_trials_all_animals.csv` |
| d | Rate of accuracy decay (slope) vs. accuracy at 10 s delay — scatter with regression | `panels_a-h_behavior_all_trials_all_animals.csv` |
| e | Lapse rate vs. overall accuracy per animal — scatter with regression | `panels_a-h_behavior_all_trials_all_animals.csv` |
| f | Per-animal repeating bias vs. delay — line plot | `panels_a-h_behavior_all_trials_all_animals.csv` |
| g | Mean repeating bias vs. delay with linear fit and statistics | `panels_a-h_behavior_all_trials_all_animals.csv` |
| h | Lapse rate vs. repeating bias at 0.01 s delay — scatter with regression | `panels_a-h_behavior_all_trials_all_animals.csv` |
| i | Trial-outcome autocorrelogram across lags 1–25 with exponential decay fit (τ) | `panel_i_trial_outcome_autocorrelation_by_lag_per_animal.csv` |
| j | Choice-repetition autocorrelogram across lags with exponential decay fit | `panel_j_repeat_choice_autocorrelation_by_lag_per_animal.csv` |
| k | GLM schematic | *Schematic — no data file* |
| l | GLM weight distributions for current stimulus S_t (Left and Right) — violin + boxplot | `panels_l-o_GLMM_stimulus_history_weights_per_animal.csv` |
| m | GLM weight distributions for stimulus × delay interaction S_t·D_t | `panels_l-o_GLMM_stimulus_history_weights_per_animal.csv` |
| n | GLM weight distribution for previous choices C_{t−k} (exponential history) | `panels_l-o_GLMM_stimulus_history_weights_per_animal.csv` |
| o | GLM weight distribution for previous choices × delay C_{t−k}·D_t | `panels_l-o_GLMM_stimulus_history_weights_per_animal.csv` |

---

## Figure 2 — Hidden Markov Model of behavioral states

`code/fig_2_model/` · `data/fig_2_model/`

*Does a two-state Hidden Markov Model (STM = working memory state; RepL = repeating-left state) capture trial-by-trial behavioral dynamics better than Drift-Walk family models, and what are the fitted parameters across animals?*

| Panel | Content | Source file(s) |
|---|---|---|
| a | Task schematic | *Schematic — no data file* |
| b | DW model family vs. real data — accuracy vs. delay (DW, DW-L, DW-B, DW-M) | `panels_bc_DW_model{9,10,11,12}_predicted_accuracy_and_repeat_bias.csv` |
| c | DW model family vs. real data — repeating bias vs. delay | `panels_bc_DW_model{9,10,11,12}_predicted_accuracy_and_repeat_bias.csv` |
| d | HMM diagram | *Schematic — no data file* |
| e | Per-trial log-likelihood: HMM vs. DW model variants — violin + box + strip | `panel_e_per_trial_log_likelihood_HMM_vs_DW_models.csv` |
| f | Example session (C38-2021-07-03): posterior p(STM) over trials with state shading | `panels_fg_example_session_10s_delay_with_HMM_state_posteriors.csv` |
| g | Same session: 20-trial running repeating bias with state shading | `panels_fg_example_session_10s_delay_with_HMM_state_posteriors.csv` |
| h | Histogram of posterior p(STM) across all sessions, colored by state (<0.5 = RepL, >0.5 = STM) | `panel_h_HMM_state_probability_histogram_all_sessions.csv` |
| i | Accuracy vs. delay by HMM state (STM / RepL / All) — data + model lines | `panels_fg_ij_behavior_all_trials_with_HMM_state_posteriors.csv`<br>`panels_ij_HMM_model_simulated_trials_with_state_posteriors.csv` |
| j | Repeating bias vs. delay by HMM state with inline state labels | `panels_fg_ij_behavior_all_trials_with_HMM_state_posteriors.csv`<br>`panels_ij_HMM_model_simulated_trials_with_state_posteriors.csv` |
| k–p | HMM fitted parameters per animal: loading probabilities (k), state posteriors (l), DW barrier α (m), DW asymmetry μ (n), repeating strength β_C (o), choice bias β_bias (p) | `panels_k-p_HMM_fitted_parameters_all_animals.csv` |
| q | Individual fit — Mouse N11: real vs. HMM-simulated accuracy and repeating bias | `panel_q_mouse_N11_10s_delay_behavioral_data.csv`<br>`panel_q_mouse_N11_10s_delay_HMM_model_predictions.csv` |
| r | Individual fit — Mouse C37: real vs. HMM-simulated accuracy and repeating bias | `panel_r_mouse_C37_10s_delay_behavioral_data.csv`<br>`panel_r_mouse_C37_10s_delay_HMM_model_predictions.csv` |

---

## Figure 3 — Prefrontal cortex encodes working memory across stimulus, delay, and response epochs

`code/fig_3_ephys_wm/` · `data/fig_3_ephys_wm/`

*Does prefrontal population activity maintain a stable, generalizable choice representation throughout the delay period, and do the decoder weight vectors for different task epochs overlap?*

| Panel | Content | Source file(s) |
|---|---|---|
| a | Population raster (neurons sorted by delay-period rate) + PSTH — trial 237, session E20 | `panel_a_population_firing_rates_trial237_session_E20.csv`<br>`panel_a_population_spike_times_trial237_session_E20.csv` |
| b | Neuron 153 spike raster + PSTH — STM correct trials, split Left vs. Right, session E20 | `panel_b_neuron153_spike_times_all_10s_delay_trials.csv` |
| c | Cross-temporal decoding heatmap: excess accuracy at all train × test time pairs; epoch labels Stimulus / Delay / Response | `panels_c-f_cross_temporal_choice_decoding_3s_delay.csv`<br>`panels_c-f_cross_temporal_choice_decoding_3s_delay_shuffle_control.csv` |
| d | Decoder time course trained on stimulus epoch (0.0–0.25 s) — "Stimulus code" | `panels_c-f_cross_temporal_choice_decoding_3s_delay.csv`<br>`panels_c-f_cross_temporal_choice_decoding_3s_delay_shuffle_control.csv` |
| e | Decoder time course trained on delay epoch (3.0–3.25 s) — "Delay code" | `panels_c-f_cross_temporal_choice_decoding_3s_delay.csv`<br>`panels_c-f_cross_temporal_choice_decoding_3s_delay_shuffle_control.csv` |
| f | Decoder time course trained on response epoch (3.75–4.0 s) — "Response code" | `panels_c-f_cross_temporal_choice_decoding_3s_delay.csv`<br>`panels_c-f_cross_temporal_choice_decoding_3s_delay_shuffle_control.csv` |
| g | Decoder weight-vector dot products between epoch pairs — boxplot + strip; significance bars | `panel_g_decoder_weight_vectors_epoch_overlap.csv` |

---

## Figure 4 — Delay-period representation breaks down on working memory error trials

`code/fig_4_ephys_errors/` · `data/fig_4_ephys_errors/`

*When and how does the prefrontal delay-period choice representation differ between correct and incorrect STM trials, and can single-trial decoder trajectories reveal the timing of memory failure?*

| Panel | Content | Source file(s) |
|---|---|---|
| a | Stimulus decoding time course — STM correct (darkgreen) vs. incorrect (crimson) trials | `panel_a_stimulus_decoding_accuracy_STM_correct_vs_incorrect.csv` |
| b | Response-epoch decoding time course — correct vs. incorrect STM trials | `panel_b_response_decoding_accuracy_STM_correct_vs_incorrect.csv` |
| c | Delay-epoch decoding time course — correct vs. incorrect STM trials | `panel_c_delay_decoding_accuracy_STM_correct_vs_incorrect.csv` |
| d | Delay decoder log-odds by epoch (early vs. late) for correct vs. incorrect — grouped boxplot | `panel_d_delay_decoder_log_odds_by_epoch_STM_correct_vs_incorrect.csv` |
| e | Example session E20_2022_02_27: delay decoding, correct vs. incorrect with shuffle baseline | `panel_e_example_session_delay_decoding_correct_vs_incorrect.csv`<br>`panel_e_example_session_delay_decoding_correct_vs_incorrect_shuffle.csv` |
| f | Neuron 153 PSTH — correct vs. incorrect STM trials, split by Left/Right stimulus | `panel_f_neuron153_spike_times_all_10s_delay_trials.csv` |
| g | Single trial T185 (correct Right, session E17): population raster → sorted firing rates → delay decoder log-odds | `fig4_panel_g_*` files in **data/fig_5_ephys_repl/** |
| h | Single trial T83 (incorrect Right, session E17): same three-panel layout as g | `fig4_panel_h_*` files in **data/fig_5_ephys_repl/** |
| i | Single-trial log-odds heatmap (stimulus-aligned) with trials sorted by reversal time; mean log-odds below | `panels_ij_single_trial_delay_decoder_log_odds_timeseries.csv` |
| j | Same trials as i, realigned to the log-odds sign-reversal time point | `panels_ij_single_trial_delay_decoder_log_odds_timeseries.csv` |

---

## Figure 5 — STM and RepL behavioral states correspond to distinct neural representations

`code/fig_5_ephys_repl/` · `data/fig_5_ephys_repl/`

*Do STM and RepL behavioral states produce measurably different patterns of prefrontal activity — in choice decoding, delay-period ramping, single-trial dynamics, and decoding of the previous trial's response?*

| Panel | Content | Source file(s) |
|---|---|---|
| a | Stimulus-aligned choice decoding — STM (darkgreen) vs. RepL (indigo); excess decoding accuracy | `panel_a_stimulus_aligned_choice_decoding_STM_vs_RepL.csv`<br>`panel_a_*_shuffle*.csv` (STM and RepL shuffle controls) |
| b | Go-cue-aligned choice decoding — STM vs. RepL; excess decoding accuracy | `panel_b_response_aligned_choice_decoding_STM_vs_RepL.csv`<br>`panel_b_*_shuffle*.csv` |
| c | Delay-code decoding — STM vs. RepL; stimulus-aligned (c1) and response-aligned (c2) | `panel_c_stimulus_aligned_delay_decoding_STM_vs_RepL.csv`<br>`panel_c_response_aligned_delay_decoding_STM_vs_RepL.csv` + shuffles |
| d | Ramp code: probability that population pattern resembles delay vs. pre-stimulus epoch — STM vs. RepL | `panel_d_delay_vs_prestimulus_ramp_probability_STM_vs_RepL.csv` |
| e | Example session E20_2022_02_26: delay decoding, stimulus-aligned (e1) and response-aligned (e2), STM vs. RepL | `panel_e_example_session_stimulus_aligned_delay_decoding.csv`<br>`panel_e_example_session_response_aligned_delay_decoding.csv` + shuffles |
| f | Single trial T223 (correct STM, session E17): population raster → sorted firing rates (colored by L/R preference) → delay log-odds | `panel_f_spike_times_trial223_*.csv`<br>`panel_f_decoder_output_trial223_*.csv`<br>`panel_f_population_firing_rates_trial223_*.csv` |
| g | Single trial T21 (correct RepL, session E17): same three-panel layout as f | `panel_g_spike_times_trial21_*.csv`<br>`panel_g_decoder_output_trial21_*.csv`<br>`panel_g_population_firing_rates_trial21_*.csv` |
| h | Early vs. late delay log-odds — grouped boxplot by state (STM/RepL × early/late) | `panel_h_delay_decoder_log_odds_by_epoch_STM_vs_RepL.csv` |
| i | Neuron 138 spike raster (i1) + PSTH (i2) — STM vs. RepL, correct left-choice trials, session E22_2022-01-13 | `panel_i_neuron138_spike_times_all_10s_delay_trials.csv` |
| j | Previous-response decoding across three trial epochs (Go cue t−1 / Stimulus t / Go cue t) — STM vs. RepL; excess decoding accuracy | `panel_j_*_previous_response_decoding_*.csv` (multiple, one per epoch × state × shuffle) |

---

## Figure 6 — Population synchrony tracks behavioral state and reveals rhythmic structure

`code/fig_6_synch/` · `data/fig_6_synch/`

*Does trial-by-trial population synchrony co-vary with HMM-defined behavioral state, and does the power spectral structure of population firing rates differ between STM and RepL — pointing to an oscillatory mechanism?*

| Panel | Content | Source file(s) |
|---|---|---|
| a | Example trial 432, session E20 — population raster sorted by rate (a2) + PSTH (a1) + synchrony metric time course (a3) | `panel_a_population_firing_rates_trial432_session_E20.csv`<br>`panel_a_population_spike_times_trial432_session_E20.csv` |
| b | Session E22_2022-01-13: mini-rasters for annotated trials (T153, T212, T340) with continuous synchrony + HMM state overlay across all trials | `panel_b_per_trial_synchrony_and_behavioral_state_all_sessions.csv`<br>`panel_b_spike_times_trial{153,212,340}_for_synchrony_display.csv` |
| c | Synchrony by HMM state — per-session boxplot (STM vs. RepL) with paired per-animal lines | `panel_b_per_trial_synchrony_and_behavioral_state_all_sessions.csv` |
| d | Pearson correlation of synchrony with p(STM), accuracy, and repeating bias — boxplot across sessions | `panel_d_synchrony_correlation_with_behavior_per_session.csv` |
| e | Example session E11_2021-05-12: population rate autocorrelogram STM vs. RepL (left); PSD ratio RepL/STM on log-log axes (right) | `panel_e_left_spike_autocorrelogram_STM_vs_RepL_example_session.csv`<br>`panel_e_right_PSD_ratio_RepL_over_STM_example_session.csv` |
| f | All sessions: mean PSD — STM vs. RepL, log-log axes (left); mean PSD ratio RepL/STM with 95% CI, peak highlighted at 4.4 Hz (right) | `panel_f_left_mean_power_spectrum_STM_vs_RepL_all_sessions.csv`<br>`panel_f_right_mean_PSD_ratio_with_confidence_interval.csv` |
