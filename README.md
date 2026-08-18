# Working Memory Manuscript Figures

Code to reproduce all main and supplementary figures for:

**Episodic recruitment of attractor dynamics in frontal cortex reveals distinct mechanisms for forgetting and lack of cognitive control in short-term memory**

Tíffany Oña-Jodar, Genís Prat-Ortega, Jordi Pastor-Ciurana, Chengyu Li, Josep Dalmau, Albert Compte\*, and Jaime de la Rocha\* — *Nature Neuroscience* (in revision)

\* Co-senior authorship

Preprint: https://www.biorxiv.org/content/10.1101/2024.02.18.579447v1

---

## Overview

This repository generates every figure in the manuscript from pre-processed data files. Each figure has its own subfolder under `code/` containing a self-contained script or notebook. Running a script produces the corresponding panel outputs in `figures/`.

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
│   ├── fig_6_synch/           # Fig 6 — LFP synchrony, PSDs, oscillations
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

# Create the virtual environment and install dependencies
uv sync
```

To activate the environment:

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

`config.py` defines the three key paths:

| Variable | Default (relative to repo root) | Purpose |
|---|---|---|
| `DATA_DIR` | `./data/` | Pre-processed panel CSVs |
| `FIGURES_OUT` | `./figures/` | Rendered figure outputs |
| `ANALYSIS_DATA` | `../general_data/` | Decoded session files not bundled in `data/` |

If your data lives elsewhere (e.g. on a different drive), edit the `ANALYSIS_DATA` line in `config.py`.

---

## Reproducing figures

Each `code/fig_N_*/` folder contains one primary script (`.py`) or notebook (`.ipynb`). Run them from the repo root so that relative imports resolve correctly.

**Example — Fig 3:**

```bash
python code/fig_3_ephys_wm/fig_3_panel.py
```

**Example — Supplementary figure (notebook):**

```bash
jupyter notebook code/supp_figures/supp_fig_1_tau_exponential/supp_fig_1_tau_exponential.ipynb
```

Output files are written to `figures/fig_3_ephys_wm/` (or the equivalent subfolder). All output directories are created automatically when the script runs.

---

## Data

Pre-processed data files (`data/`) are **not included in this repository**. They are distributed separately through the Open Research Framework (ORF) publication associated with this paper.

> **Important:** Scripts read data files immediately on startup. If the data files are missing, the script will crash with a `FileNotFoundError`. Download and place the data before running any script.

Download the data from the ORF publication, then place the extracted `data/` folder at the root of this repository so the structure matches what `config.py` expects:

```
WM_manuscript_FIGURES/
├── data/
│   ├── fig_1_behavior/
│   ├── fig_2_model/
│   ...
```

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

## Contact

Tiffany Ona — [tiffany.ona@alleninstitute.org](mailto:tiffany.ona@alleninstitute.org)
