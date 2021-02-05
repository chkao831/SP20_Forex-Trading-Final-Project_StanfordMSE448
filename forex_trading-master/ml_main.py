'''
A Python script version for the notebook ml_main.ipynb.
For faster debugging support functions.
To tune parameters, please use the Jupyter notebook.
'''

import sys
sys.path.append('..')

import os
import pprint as pp

import pandas as pd
import tqdm

from arguments import *
from constants import *
from data import *
import models as supported_models
from training_tools import *
from utils import *


storage = GCStorage.get_CloudFS(project_name=PROJECT_NAME,
                                bucket_name=GC_BUCKET,
                                credential_path=CREDENTIAL_PATH)

all_args = parse_args()
# all_args = parse_from_string('--exp_name=jupyter --device=cpu --num_candles=20 --num_workers=1 --fast_debug=True --candle_interval=30 --log_level=4')

logger = all_args.misc_args.logger

train_loader, valid_loader = get_dataloaders(all_args)
loaders = {'train': train_loader, 'valid': valid_loader}

logger.log_stdout('Loaders defined')

if ALL_SETTINGS[all_args.misc_args.exp_setting]['type'] == 'regression':
    regressors_list = supported_models.regressors
    metric_list = supported_models.regression_metrics
    all_models = {model_name: regressors_list[model_name] \
                                                for model_name in regressors_list}
elif ALL_SETTINGS[all_args.misc_args.exp_setting]['type'] == 'binary_classification':
    classifier_list = supported_models.classifiers
    metric_list = supported_models.binary_classification_metrics
    all_models = {model_name: classifier_list[model_name] \
                                                for model_name in classifier_list}
else:
    print(f'Task type {all_args.misc_args.exp_setting["type"]} not supported yet')
    exit(-1)


def gether_data(loader):
    all_x, all_gt = [], []
    total_len = len(loader.dataset)
    for i in range(total_len):
        x, gt = loader.dataset[i]
        if i % 500 == 0:
            logger.log_stdout(f'# [{i + 1}]/{total_len}')

        x = x.flatten()
        # gt = gt

        all_x.append(x)
        all_gt.append(gt)
    
    return all_x, all_gt

train_x, train_gt = gether_data(train_loader)
valid_x, valid_gt = gether_data(valid_loader)

# print(valid_gt)

train_result = {}
valid_result = {}

# Note that we do not perform GridSearchCV here, we only train, then eval
for model_name in all_models:
    print(f'model: {model_name}')
    model = all_models[model_name]
    print(model)

    model.fit(train_x, train_gt)

    train_pred = model.predict(train_x)
    valid_pred = model.predict(valid_x)

    tr = {}
    vr = {}
    for metric in metric_list:
        train_score = metric_list[metric](train_gt, train_pred)
        valid_score = metric_list[metric](valid_gt, valid_pred)
        print(f'{metric}: {valid_score}')

        tr[metric] = train_score
        vr[metric] = valid_score

    train_result[model_name] = tr
    valid_result[model_name] = vr

tpd = pd.DataFrame(train_result)
vpd = pd.DataFrame(valid_result)

print(tpd)
print(vpd)
        




