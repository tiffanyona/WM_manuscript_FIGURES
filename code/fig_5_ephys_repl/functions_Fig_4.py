# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 15:14:01 2023

@author: Tiffany
"""

COLORLEFT = 'teal'
COLORRIGHT = '#FF8D3F'

import pandas as pd
import numpy as np
import seaborn as sns
#Import all needed libraries
from neo.core import SpikeTrain
from quantities import ms
from elephant.statistics import time_histogram, instantaneous_rate
from elephant.kernels import GaussianKernel
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import ROOT, DATA_DIR
sys.path.insert(0, str(ROOT / 'src'))
from functions import add_stat_annotation, single_trial_with_decoder, convolveandplot, plot_decoder_shuffle, new_convolve, plotsingledelay

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

