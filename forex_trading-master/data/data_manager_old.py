from dateutil import parser
import datetime
import random
import numpy as np

from torch.utils.data import DataLoader, Dataset

from constants import *
from .preprocessing import *
from utils import *


random.seed(448)


# Define column names
OHLC = ['Open', 'High', 'Low', 'Close']


def date_to_frac_minute(row):
    t = row.name
    return t.hour * 60 + t.minute / 1440


class FXDataset(Dataset):

    def __init__(self, days, logger, data_args):
        self.days = days
        self.logger = logger
        self.data_args = data_args

        # self.api = utils.DataAPI()
        
        self.api = ProcessedDataAPI.get_DataAPI(f'processed_{data_args.candle_interval}s_data')

        hrs = 0
        self.index_day_map = {}
        for d in self.days:
            day_index_start = hrs
            day_hour_start = DAY_HOUR_LOOKUP[d][0]
            for i in range(DAY_HOUR_LOOKUP[d][1] - DAY_HOUR_LOOKUP[d][0]):
                self.index_day_map[hrs] = (d, day_index_start, day_hour_start)
                hrs += 1
        self.hours = hrs

    def __len__(self):
        # Chunk data every DATA_INTERVAL seconds
        return self.hours * 3600 // self.data_args.candle_interval

    def _retry(self, message):
        self.logger.log_stdout(message, level=LOG_DEBUG)
        return self.__getitem__(random.randint(0, len(self) - 1))

    def __getitem__(self, index):

        hour_index = index // (3600 // self.data_args.candle_interval)
        day, day_index_start, day_hour_start = self.index_day_map[hour_index]

        hour_offset_in_day = hour_index - day_index_start
        candle_offset_in_hour = index % (3600 // self.data_args.candle_interval)
        start_timedelta = datetime.timedelta(seconds=candle_offset_in_hour * self.data_args.candle_interval,
                                             hours=hour_offset_in_day)

        day_datetime = parser.parse(day, dayfirst=False)
        day_datetime = day_datetime.replace(hour=day_hour_start)
        slot_start = day_datetime + start_timedelta
        slot_end = day_datetime + start_timedelta + \
                datetime.timedelta(seconds=self.data_args.candle_interval * self.data_args.num_candles)

        candle_interval = datetime.timedelta(seconds=self.data_args.candle_interval)

        final_lookahead = datetime.timedelta(
            seconds=self.data_args.candle_interval * self.data_args.num_iterval_ahead)
        lookahead_end = slot_end + final_lookahead

        # If end is beyond current day then randomly select another data point
        if (slot_end.day != slot_start.day or slot_end.hour >= DAY_HOUR_LOOKUP[day][1]) \
            or \
           (lookahead_end.day != slot_start.day or lookahead_end.hour >= DAY_HOUR_LOOKUP[day][1]):
            return self._retry('Day end index exceed limit')

        self.logger.log_stdout(f'Index[{index}] ({slot_start}, {slot_end})', level=LOG_DEBUG)

        
        # self.data_args.currency_pair.upper(),
        all_result = []
        all_price_means = []
        all_price_stds = []
        all_volume_means = []
        all_volume_stds = []
        for pair in TRADING_PAIRS:
            pair_result = []
            for i in range(NUM_LP):
                candles = self.api.get(pair,
                                       str(i + 1),
                                       start_time=slot_start,
                                       length=None,
                                       end_time=slot_end,
                                       verbose=self.logger.log_level <= LOG_DEBUG)
                candles = candles.to_dict('records')

                if len(candles) < 2:
                    return self._retry(f'LP {i + 1} has insufficient number of candles {len(candles)}')

                #print(slot_start) 
                #print(slot_end)
                #print(candle_interval)
                #print(candles)
                candle_df = pad_candle(candles, candle_interval,
                                       slot_start, slot_end)

                # print(candle_df.columns)
                candle_df = candle_df.rename(
                    {k: f'{pair}_{k}_{i + 1}' for k in candle_df.columns},
                    axis='columns')
                candle_df = candle_df.rename(
                    {'Volume': f'{pair}_{k}_{i + 1}' for k in candle_df.columns},
                    axis='columns')
                pair_result.append(candle_df)

            cur_df = pair_result[0]
            for i in range(1, NUM_LP):
                cur_df = cur_df.join(pair_result[i], on='Date')

            price_columns = [f'{pair}_{k}_{i + 1}' for i in range(NUM_LP) for k in OHLC]
            volume_columns = [f'{pair}_Volume_{i + 1}' for i in range(NUM_LP)]

            all_prices = cur_df[price_columns].to_numpy()
            price_mean = np.mean(all_prices)
            price_std = np.std(all_prices)

            all_volumes = cur_df[volume_columns].to_numpy()
            volume_mean = np.mean(all_volumes)
            volume_std = np.std(all_volumes)

            if volume_std == 0 or price_std == 0:
                return self._retry(f'Index[{index}] ({slot_start}, {slot_end}) has 0 volume or price std')
        
            all_price_means.append(price_mean)
            all_price_stds.append(price_std)
            all_volume_means.append(volume_mean)
            all_volume_stds.append(volume_std)

            all_result.append(cur_df)

        joint_df = all_result[0]
        for i in range(1, len(TRADING_PAIRS)):
            joint_df = joint_df.join(all_result[i], on='Date')

        joint_df['Frac_Minute'] = joint_df.apply(
                lambda row: date_to_frac_minute(row), axis=1)

        processed_df = data_normalize(joint_df, TRADING_PAIRS, range(NUM_LP),
                                      all_price_means, all_price_stds,
                                      all_volume_means, all_volume_stds)

        pair_idx = TRADING_PAIRS.index(self.data_args.currency_pair.upper())
        #try:
        pair_mean = all_price_means[pair_idx]
        pair_std = all_price_stds[pair_idx]
        target = generate_target(self.api, self.data_args.currency_pair.upper(), range(NUM_LP),
                                    slot_end, lookahead_end,
                                    pair_mean, pair_std)
        #except:
        #    return self._retry(f'Index[{index}] ({slot_start}, {slot_end}) future outlook is Nan')

        return processed_df.to_numpy(), target


def get_dataloaders(args):

    if args.misc_args.fast_debug:
        train_days, valid_days = convert_train_val_days(TRAIN_DAYS,
                                                        VALID_DAYS)
    else:
        train_days, valid_days = TRAIN_DAYS, VALID_DAYS

    train_dataset = FXDataset(train_days, args.misc_args.logger, args.data_args)
    valid_dataset = FXDataset(valid_days, args.misc_args.logger, args.data_args)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.train_args.batch_size,
                                  shuffle=True,
                                  num_workers=args.data_args.num_workers)
    
    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=args.train_args.batch_size,
                                  shuffle=False,
                                  num_workers=args.data_args.num_workers)
    
    return train_dataloader, valid_dataloader
