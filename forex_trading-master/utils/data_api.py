import os
import pickle

import datetime
from pathlib import Path
import pandas as pd
from dateutil import parser

from .gc_storage import GCStorage
from .fast_csv_reader import read_csv
from constants import *


def str_to_datetime(function):
    def wrapper(*args, **kwargs):
        
        all_kwargs = {k: kwargs[k] for k in kwargs}
        if type(kwargs['start_time']) == str:
            start_time = parser.parse(kwargs['start_time'], dayfirst=False)
            all_kwargs['start_time'] = start_time
        
        if 'end_time' in kwargs and type(kwargs['end_time']) == str:
            start_time = parser.parse(kwargs['end_time'], dayfirst=False)
            all_kwargs['end_time'] = start_time

        result = function(*args, **all_kwargs)
        return result
    return wrapper


def get_fpath(pair, lp_idx, start_time):
    whole_day = start_time.replace(minute=0, hour=0,
                                          second=0, microsecond=0)
    nice_date = whole_day.strftime('%Y%m%d')

    # fname = f'{nice_date}-{pair}-{lp_idx}.csv'
    fname = f'{nice_date}-{pair}-{lp_idx}.pickle'
    return Path(pair) / nice_date / fname, whole_day


def binary_search(orig, value):
    
    if value <= orig[0]:
        return 0
    elif value > orig[-1]:
        print('Exceed end')
        return -1
    
    left = 0
    right = len(orig)
    
    while True:
        mid = (left + right) // 2
        if orig[mid - 1] < value and orig[mid] >= value:
            return mid
        elif orig[mid - 1] <= value and orig[mid] >= value:
            return mid - 1
        elif orig[mid] > value:
            left = left
            right = mid
        else:
            left = mid
            right = right


class RawDataAPI:

    MONO = None
    
    @staticmethod
    def get_DataAPI(*args, **kwargs):
        if RawDataAPI.MONO is not None:
            return RawDataAPI.MONO
        else:
            RawDataAPI.MONO = RawDataAPI(*args, **kwargs)
            print('Unique instance for RawDataAPI has been created')
            return RawDataAPI.MONO
    
    def __init__(self):
        self.data_path = Path(TEMP_FOLDER) / DATA_STORAGE
        self.storage = GCStorage(PROJECT_NAME, GC_BUCKET, CREDENTIAL_PATH)

        # TODO (JBY), implement a better cache
        self.local_storage = []

    def describe_cache(self):
        for item in self.local_storage:
            print(f'{item["pair"]}[{item["lp_idx"]}]: {item["start_time"]} ({len(item["data"])})')
       
    @str_to_datetime
    def get(self, pair, lp_idx, start_time, length, end_time=None, verbose=False):

        assert length is None or end_time is None, "Exactly one of length or end_time must be None"
        
        if verbose:
            print(f'Length of cache is {len(self.local_storage)}')
        
        fpath, whole_day = get_fpath(pair, lp_idx, start_time)

        df = None
        time_list = None
        for item in self.local_storage:
            if item['pair'] == pair and item['lp_idx'] == lp_idx and \
                item['start_time'] == whole_day:
                df = item['data']
                time_list = item['time_list']
                break

        if df is None:
            df = self._cloud_get(fpath, whole_day,
                                 cutoff_time=whole_day + datetime.timedelta(days=1),
                                 verbose=verbose)
            time_list = list(df['time'])
            self.local_storage.append({'pair': pair, 'lp_idx': lp_idx,
                                        'start_time': whole_day,
                                        'data': df,
                                        'time_list': time_list})
        
        df = df[(df['bid price'] > 0.1) & (df['ask price'] > 0.1)]
        time_list = list(df['time'])        # do again to ensure consistency
        # print(f'Length of entire dataframe: {len(time_list)}')
        start_index = None
        # for i, row in df.iterrows():
        #    if row['time'] > start_time:
        #        start_index = i
        #        break
        start_index = binary_search(time_list, start_time)

        assert start_index is not None

        # result_dict = {k: df[k][start_index: start_index + length] for k in df}
        # result_df = pd.DataFrame(result_dict)
        # return result_df

        if length is not None:
            return df.iloc[start_index: start_index + length]
        else:
            end_time = parser.parse(end_time, dayfirst=False) if type(end_time) == str else end_time
            end_index = binary_search(time_list, end_time)
            temp = df.iloc[start_index: end_index]
            return temp


    def _cloud_get(self, fpath, whole_day, cutoff_time, verbose):
        
        target_file = self.data_path / fpath
        cloud_path = Path(DATA_STORAGE) / fpath

        #if not os.path.exists(target_file):
        #    if verbose:
        #        print(f'Downloading from {cloud_path}')
        #    self.storage.download(target_file, cloud_path)
        #else:
        #    if verbose:
        #        print(f'Path already exist: {cloud_path}')

        if verbose:
            print(f'Loading {fpath}')
        # df = pd.read_csv(target_file, parse_dates=['time'])
        # df = pd.DataFrame(read_csv(target_file, date_col='time', cutoff_time=cutoff_time))
        df = pickle.load(open(target_file, 'rb'))
        
        if len(self.local_storage) > 150:
            self.local_storage.pop(0)

        return df



class ProcessedDataAPI:

    MONO = None
    
    @staticmethod
    def get_DataAPI(*args, **kwargs):
        if ProcessedDataAPI.MONO is not None:
            return ProcessedDataAPI.MONO
        else:
            ProcessedDataAPI.MONO = ProcessedDataAPI(*args, **kwargs)
            print('Unique instance for DataAPI has been created')
            return ProcessedDataAPI.MONO
    
    def __init__(self, processed_storage):
        self.processed_storage = processed_storage
        self.data_path = Path(TEMP_FOLDER) / processed_storage
        self.storage = GCStorage(PROJECT_NAME, GC_BUCKET, CREDENTIAL_PATH)

        # TODO (JBY), implement a better cache
        self.local_storage = []

    def describe_cache(self):
        for item in self.local_storage:
            print(f'{item["pair"]}[{item["lp_idx"]}]: {item["start_time"]} ({len(item["data"])})')
       
    @str_to_datetime
    def get(self, pair, lp_idx, start_time, length, end_time=None, verbose=False):

        assert length is None or end_time is None, "Exactly one of length or end_time must be None"
        
        if verbose:
            print(f'Length of cache is {len(self.local_storage)}')
        
        fpath, whole_day = get_fpath(pair, lp_idx, start_time)

        df = None
        time_list = None
        for item in self.local_storage:
            if item['pair'] == pair and item['lp_idx'] == lp_idx and \
                item['start_time'] == whole_day:
                df = item['data']
                time_list = item['time_list']
                break

        if df is None:
            df = self._cloud_get(fpath, whole_day,
                                 cutoff_time=whole_day + datetime.timedelta(days=1),
                                 verbose=verbose)
            time_list = list(df.index)
            self.local_storage.append({'pair': pair, 'lp_idx': lp_idx,
                                       'start_time': whole_day,
                                       'data': df,
                                       'time_list': time_list})
        
        # df = df[(df['bid price'] > 0.1) & (df['ask price'] > 0.1)]
        # time_list = list(df.index)        # do again to ensure consistency
        # print(f'Length of entire dataframe: {len(time_list)}')
        start_index = None
        # for i, row in df.iterrows():
        #    if row['time'] > start_time:
        #        start_index = i
        #        break
        start_index = binary_search(time_list, start_time)

        assert start_index is not None

        # result_dict = {k: df[k][start_index: start_index + length] for k in df}
        # result_df = pd.DataFrame(result_dict)
        # return result_df

        df['Date'] = df.index        

        if length is not None:
            return df.iloc[start_index: start_index + length]
        else:
            end_time = parser.parse(end_time, dayfirst=False) if type(end_time) == str else end_time
            end_index = binary_search(time_list, end_time)
            temp = df.iloc[start_index: end_index]
            return temp


    def _cloud_get(self, fpath, whole_day, cutoff_time, verbose):
        
        target_file = self.data_path / fpath
        cloud_path = Path(self.processed_storage) / fpath

        if not os.path.exists(target_file):
            if verbose:
                print(f'Downloading from {cloud_path}')
            self.storage.download(target_file, cloud_path)
        else:
            if verbose:
                print(f'Path already exist: {cloud_path}')

        if verbose:
            print(f'Loading {fpath}')

        df = pickle.load(open(target_file, 'rb'))
        
        if len(self.local_storage) > 150:
            self.local_storage.pop(0)

        return df


class CandleDataAPI:
    '''Assumes that data actually exists'''

    MONO = None
    
    @staticmethod
    def get_DataAPI(*args, **kwargs):
        if CandleDataAPI.MONO is not None:
            return CandleDataAPI.MONO
        else:
            CandleDataAPI.MONO = CandleDataAPI(*args, **kwargs)
            print('Unique instance for CandleDataAPI has been created')
            return CandleDataAPI.MONO
    
    def __init__(self, candle_interval):
        self.processed_storage = f'candle_{candle_interval}_data'
        self.candle_interval = datetime.timedelta(seconds=candle_interval)
        self.data_path = Path(TEMP_FOLDER) / self.processed_storage
        self.storage = GCStorage(PROJECT_NAME, GC_BUCKET, CREDENTIAL_PATH)

        self.local_storage = []

    def describe_cache(self):
        for item in self.local_storage:
            print(f'{item["pair"]}[{item["lp_idx"]}]: {item["start_time"]} ({len(item["data"])})')

    @str_to_datetime
    def get(self, pair, lp_idx, start_time, length, verbose):
        
        if verbose:
            print(f'Length of cache is {len(self.local_storage)}')
        
        fpath, whole_day = get_fpath(pair, lp_idx, start_time)

        data = None
        for item in self.local_storage:
            if item['pair'] == pair and item['lp_idx'] == lp_idx and \
                item['start_time'] == whole_day:
                data = item['data']
                break

        if data is None:
            target_file = self.data_path / fpath
            data = pickle.load(open(target_file, 'rb'))

            self.local_storage.append({'pair': pair, 'lp_idx': lp_idx,
                                       'start_time': whole_day,
                                       'data': data})
        
        start_index = int((start_time - whole_day) / self.candle_interval)

        if start_index + length > len(data['candle_starts']):
            raise IndexError(f'Requested data exceeds day length')

        bids_df = {k: data['candle_bids'][k][start_index: start_index + length] \
                                    for k in data['candle_bids']}
        bids_df = pd.DataFrame(bids_df)
        asks_df = {k: data['candle_asks'][k][start_index: start_index + length] \
                                    for k in data['candle_asks']}
        asks_df = pd.DataFrame(asks_df)
        result = {'candle_starts': data['candle_starts'][start_index: start_index + length],
                  'candle_bids': bids_df, 'candle_asks': asks_df}
        return result
