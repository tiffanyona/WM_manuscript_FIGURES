from pathlib import Path

ROOT = Path(__file__).parent

# Processed analysis outputs (decoder files, session summaries not bundled in data/)

# Adjust this one line if the directory lives elsewhere on a new machine
ANALYSIS_DATA = ROOT.parent / 'general_data'

# Raw ephys session CSVs (iterated in table_animals and run_decoder_errors)
EPHYS_SESSIONS = ROOT.parent / 'Ephys' / 'summary_complete'

FIGURES_OUT = ROOT / 'figures'
DATA_DIR    = ROOT / 'data'
