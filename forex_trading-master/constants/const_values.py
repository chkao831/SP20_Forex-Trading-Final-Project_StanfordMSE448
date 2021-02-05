import os
from pathlib import Path

GC_HOME = Path(f"{os.getcwd()}/../../") # parent directory of the cloned repo
SOURCE_PATH = GC_HOME / 'forex_trading'

PROJECT_NAME = 'FX_Trading'
PROJECT_ID = 'fxtrading-275204 '
GC_BUCKET = 'integral_data'
CREDENTIAL_PATH = SOURCE_PATH / \
                    'bash_scripts' / 'fxtrading-275204-b70b76c93f8c.json'

EXP_STORAGE = 'experiments'
# DATA_STORAGE = 'organized_data'  # stores raw csv
DATA_STORAGE = 'pickled_data'    # stores pickles

TEMP_FOLDER = GC_HOME / 'temp_store'
TEMP_FOLDER.mkdir(exist_ok=True)


# JBY: Sunday evenings are rather quite, do not use them
ALL_DAYS = [                                                                      "2019-02-01", 
            "2019-02-03", "2019-02-04", "2019-02-05", "2019-02-06", "2019-02-07", "2019-02-08",
            "2019-02-10", "2019-02-11", "2019-02-12", "2019-02-13", "2019-02-14", "2019-02-15",
            "2019-02-17", "2019-02-18", "2019-02-19", "2019-02-20", "2019-02-21", "2019-02-22",
            "2019-02-24", "2019-02-25", "2019-02-26", "2019-02-27", "2019-02-28", "2019-03-01"]

'''
TRAIN_DAYS = ["2019-02-04", "2019-02-05", "2019-02-06", "2019-02-07", "2019-02-08",
              "2019-02-11", "2019-02-12", "2019-02-13", "2019-02-14", "2019-02-15",
              "2019-02-18", "2019-02-19", "2019-02-20", "2019-02-21", "2019-02-22"]
'''
TRAIN_DAYS = ["2019-02-04", "2019-02-05", "2019-02-06", "2019-02-07", "2019-02-08",
              "2019-02-11", "2019-02-12", "2019-02-13",  "2019-02-15",
              "2019-02-18", "2019-02-19", "2019-02-20", "2019-02-21", "2019-02-22"]

VALID_DAYS = ["2019-02-25", "2019-02-26", "2019-02-27", "2019-02-28", "2019-03-01"]

DAY_HOUR_LOOKUP = \
            {                                                                                                                   "2019-02-01": [0, 22], 
            "2019-02-03": [18, 21], "2019-02-04": [0, 24], "2019-02-05": [0, 24], "2019-02-06": [0, 24], "2019-02-07": [0, 24], "2019-02-08": [0, 22],
            "2019-02-10": [18, 21], "2019-02-11": [0, 24], "2019-02-12": [0, 24], "2019-02-13": [0, 24], "2019-02-14": [0, 24], "2019-02-15": [0, 22],
            "2019-02-17": [18, 21], "2019-02-18": [0, 24], "2019-02-19": [0, 24], "2019-02-20": [0, 24], "2019-02-21": [0, 24], "2019-02-22": [0, 22],
            "2019-02-24": [18, 21], "2019-02-25": [0, 24], "2019-02-26": [0, 24], "2019-02-27": [0, 24], "2019-02-28": [0, 24], "2019-03-01": [0, 22]
            }

TRADING_LP = list(range(1, 2))# TODO (JBY): Change back to 5 when things are sorted out!

SUPPORTED_MODELS = ['DummyModel' # Simpliest model, 1 layer LSTM with regression head
                    ]
SUPPORTED_LOSSES = ['mse', 'ce']

# TRADING_PAIRS = ['AUDUSD', 'EURUSD', 'GBPUSD', 'USDCAD', 'USDCHF']# Missing USDSEK and USDJPY
TRADING_PAIRS = ['EURUSD', 'USDCAD']# Missing USDSEK and USDJPY

# EVAL_METRICS = {'mse': {'func': 'metric_mse', 'format': ':6.2f', 'args': []}}
EVAL_METRICS = {'accuracy': {'func': 'metric_accuracy', 'format': ':6.2f', 'args': []}}

NUM_CHANNELS = len(TRADING_PAIRS) * len(TRADING_LP) * 5 + 1       # (5 LP x OHLC+Volume) + time of day