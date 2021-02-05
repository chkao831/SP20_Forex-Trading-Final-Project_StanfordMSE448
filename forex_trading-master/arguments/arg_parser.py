'''
Adapted for Spring 2020 Stanford MS&E 448

Adapted from Stanford AI for Healthcare Bootcamp deep learning infrastructure
Adapted from Driver2vec as part of CS341
Created for Winter 2019 Stanford CS224W
'''

import argparse
import copy
import datetime
import getpass
import json
import pathlib
from pathlib import Path
import pprint as pp
from pytz import timezone
import sys

from utils import str_to_bool, GCOpen, GCStorage
from constants import *
from logger import *


def define_parser():
    parser = argparse.ArgumentParser(
        description='Pytorch implementation of KGAT')

    # Logistics (more later)
    parser.add_argument('--exp_name',
                        dest='misc_args.exp_name',
                        default='mse448',
                        help='Experiment name')
    parser.add_argument('--exp_setting',
                        dest='misc_args.exp_setting',
                        default='simple_regression',
                        choices=SUPPORTED_SETTINGS,
                        help='Experiment setting (dataset and evaluation type)')

    # Data
    parser.add_argument('--candle_interval',
                        dest='data_args.candle_interval',
                        type=int, default=10,
                        help='Number of seconds per candle')
    parser.add_argument('--num_candles',
                        dest='data_args.num_candles',
                        type=int, default=64,
                        help='Number of candles per data chunk')
    parser.add_argument('--num_iterval_ahead',
                        dest='data_args.num_iterval_ahead',
                        type=int, default=4,
                        help='Number of candle intervals to lookahead')
    parser.add_argument('--currency_pair',
                        dest='data_args.currency_pair',
                        type=str, default='USDCAD',
                        help='Currency pair')
    parser.add_argument('--num_workers',
                        dest='data_args.num_workers',
                        type=int, default=8,
                        help='Number of workers for data loader')

    # More miscellaneous arguments
    parser.add_argument('--log_level',
                        dest='misc_args.log_level',
                        type=int, default=LOG_ALL,
                        help='Log level')
    parser.add_argument('--fast_debug',
                        dest='misc_args.fast_debug',
                        type=str_to_bool, default=False,
                        help='Shorten DL cycle to cover code faster')

    # Training setup
    '''
    parser.add_argument('--do_train',
                        dest='train_args.do_train',
                        type=str_to_bool, default=True,
                        help='Do training')
    parser.add_argument('--do_test',
                        dest='train_args.do_test',
                        type=str_to_bool, default=True,
                        help='Do testing')
    '''
    parser.add_argument('--loss_func',
                        dest='train_args.loss_func',
                        type=str, default='ce',
                        choices=SUPPORTED_LOSSES,
                        help='Loss functions')
    parser.add_argument('--device',
                        dest='train_args.device',
                        type=str, default='cuda',
                        choices=['cpu', 'cuda'],
                        help='Pytorch device')
    parser.add_argument('--disp_steps',
                        dest='train_args.disp_steps',
                        type=int, default=10,
                        help='Display step')
    parser.add_argument('--eval_steps',
                        dest='train_args.eval_steps',
                        type=int, default=99999,
                        help='Steps between evaluation')
    parser.add_argument('--max_epochs',
                        dest='train_args.max_epochs',
                        type=int, default=100,
                        help='Number of epoch.')
    parser.add_argument('--batch_size',
                        dest='train_args.batch_size',
                        type=int, default=256,
                        help='Batch size')
    parser.add_argument('--learning_rate',
                        dest='train_args.learning_rate',
                        type=float, default=1e-4,
                        help='Learning rate.')
    parser.add_argument('--weight_decay',
                        dest='train_args.weight_decay',
                        type=float, default=0.9,
                        help='Weight decay')
    parser.add_argument('--clipping_value',
                        dest='train_args.clipping_value',
                        type=float, default=1.0,
                        help='Model gradient clipping value')

    # Model setup
    '''
    parser.add_argument('--ckpt_path',
                        dest='model_args.ckpt_path',
                        default='',
                        help='Store model path.')
    parser.add_argument('--load_model', type=str_to_bool, default=False,
                        dest='model_args.load_model',
                        help='Load model')
    '''
    parser.add_argument('--model_type',
                        dest='model_args.model_type',
                        default='DummyModel',
                        choices=SUPPORTED_MODELS,
                        help='Specify a model type')
    parser.add_argument('--emb_size',
                        dest='model_args.emb_size',
                        type=int,
                        default=32,
                        help='Embedding size')
    parser.add_argument('--hidden_size',
                        dest='model_args.hidden_size',
                        type=int,
                        default=64,
                        help='Size of hidden layer')
    parser.add_argument('--num_layers',
                        dest='model_args.num_layers',
                        type=int, default=3,
                        help='Number of layers in stack')
    '''
    parser.add_argument('--dropout',
                        dest='model_args.dropout',
                        type=float, default=0.25,
                        help='Dropout ratio')
    parser.add_argument('--use_att',
                        dest='model_args.use_att',
                        type=str_to_bool, default=True,
                        help='whether using attention mechanism')
    '''

    return parser.parse_args()

def fix_nested_namespaces(args):
    """Convert a Namespace object to a nested Namespace."""
    group_name_keys = []

    for key in args.__dict__:
        if '.' in key:
            group, name = key.split('.')
            group_name_keys.append((group, name, key))

    for group, name, key in group_name_keys:
        if group not in args:
            args.__dict__[group] = argparse.Namespace()

        args.__dict__[group].__dict__[name] = args.__dict__[key]
        del args.__dict__[key]    

def get_experiment_number(experiments_dir, experiment_name):
    """Parse directory to count the previous copies of an experiment."""
    dir_structure = GCStorage.MONO.list_files(experiments_dir)
    dirnames = [exp_dir.split('/')[-1] for exp_dir in dir_structure[1]]

    ret = 1
    for d in dirnames:
        if d[:d.rfind('_')] == experiment_name:
            ret = max(ret, int(d[d.rfind('_') + 1:]) + 1)
    return ret

def namespace_to_dict(args):
    """Turn a nested Namespace object into a nested dictionary."""
    args_dict = vars(copy.deepcopy(args))

    for arg in args_dict:
        obj = args_dict[arg]
        if isinstance(obj, argparse.Namespace):
            item = namespace_to_dict(obj)
            args_dict[arg] = item
        else:
            if isinstance(obj, pathlib.PosixPath):
                args_dict[arg] = str(obj)

    return args_dict

def parse_args():
    args = define_parser()
    fix_nested_namespaces(args)
    
    username = getpass.getuser()
    us_timezone = timezone('US/Pacific')
    date = datetime.datetime.now(us_timezone).strftime("%Y-%m-%d")
    args.misc_args.save_dir = Path(EXP_STORAGE) / date

    args.misc_args.exp_name = username + '_' + args.misc_args.exp_name
    exp_num = get_experiment_number(args.misc_args.save_dir,
                                    args.misc_args.exp_name)
    args.misc_args.exp_name = args.misc_args.exp_name + '_' + str(exp_num)
    args.misc_args.save_dir = args.misc_args.save_dir / args.misc_args.exp_name
    args.misc_args.log_file = args.misc_args.save_dir / 'run_log.txt'

    args_path = args.misc_args.save_dir / 'args.json'
    arg_dict = namespace_to_dict(args)

    with GCOpen(args_path, 'w') as f:
        json.dump(arg_dict, f, indent=4, sort_keys=True)
        f.write('\n')

    # Now let's add stuff that can't be written into JSON
    args.misc_args.logger = Logger(args.misc_args.log_level,
                                   args.misc_args.log_file,
                                   args.misc_args.save_dir)

    args.misc_args.logger.log_data(arg_dict, 'args.json')
    args.misc_args.logger.log(str(arg_dict))

    arg_text = pp.pformat(arg_dict, indent=4)
    cl_text = ' '.join(sys.argv)
    args.misc_args.logger.log_text({'setup:command_line': cl_text,
                                    'setup:arguments': arg_text},
                                    0, False)

    return args


def parse_from_string(string=''):
    sys.argv = sys.argv[:1]
    sys.argv.extend(string.split())
    return parse_args()