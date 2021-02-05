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


class SingleFXDatasetBase(Dataset):

    def __init__(self, days, logger, data_args):
        self.days = days
        self.logger = logger
        self.data_args = data_args

        self.api = CandleDataAPI.get_DataAPI(data_args.candle_interval)
        self.candle_interval = datetime.timedelta(seconds=data_args.candle_interval)

        index = 0
        self.day_index_range = []
        for d in self.days:
            day_index_start = index
            day_hour_start = DAY_HOUR_LOOKUP[d][0]
            day_hour_end = DAY_HOUR_LOOKUP[d][1]

            num_hours = day_hour_end - day_hour_start
            num_candles = int(num_hours * 3600 / data_args.candle_interval)
            index_range = (day_index_start, num_candles, d, day_hour_start)
            self.day_index_range.append(index_range)
            index += num_candles
        self.total_len = index

    def __len__(self):
        return self.total_len

    def _retry(self, message):
        self.logger.log_stdout(message, level=LOG_DEBUG)
        return self.getitem(random.randint(0, len(self) - 1))

    def getitem(self, index):

        day = None
        offset = None
        day_hour_start = None
        for day_range in self.day_index_range:
            if index >= day_range[0] and index < day_range[0] + day_range[1]:
                day = day_range[2]
                offset = index - day_range[0]
                day_hour_start = day_range[3]
                break
        
        day_datetime = parser.parse(day, dayfirst=False)
        start_timedelta = datetime.timedelta(hours=day_hour_start,
                                             seconds=offset * self.data_args.candle_interval,
                                             )
        slot_start = day_datetime + start_timedelta
        slot_end = slot_start + datetime.timedelta(seconds=self.data_args.num_candles * self.data_args.candle_interval)
        # self.data_args.num_candles

        if slot_start.day != slot_end.day:
            return self._retry(f'Index[{index}] ({slot_start}) is too close to the end')
        
        all_features = []
        all_targets = []
        pair = self.data_args.currency_pair
        for i in TRADING_LP:
            try:
                feature = self.api.get(pair=pair,
                                       lp_idx=i,
                                       start_time=slot_start,
                                       length=self.data_args.num_candles,
                                       verbose=self.logger.log_level <= LOG_DEBUG)
                target = self.api.get(pair=pair,
                                      lp_idx=i,
                                      start_time=slot_end,
                                      length=1,
                                      verbose=self.logger.log_level <= LOG_DEBUG)
            except IndexError:
                return self._retry(f'Index[{index}] ({slot_start}) is too close to the end')

            # ol skips keyword volume
            # Should be ['Close', 'High', 'Low', 'Open', 'Volume']
            # Note that this way we actually have no volume information
            sorted_keys = sorted(feature['candle_bids'].keys())
            for k in sorted_keys:
                if 'ol' not in k:
                    all_features.append(feature['candle_bids'][k])
                else:
                    all_features.append(feature['candle_bids'][k] / 1000000)
            for k in sorted_keys:
                if 'ol' not in k:
                    all_features.append(feature['candle_asks'][k])
                else:
                    all_features.append(feature['candle_asks'][k] / 1000000)
            
            assert len(target['candle_bids']['Close']) == 1

            target_data = (target['candle_bids']['Close'][0], target['candle_bids']['High'][0], target['candle_bids']['Low'][0],
                           target['candle_asks']['Close'][0], target['candle_asks']['High'][0], target['candle_asks']['Low'][0])
            all_targets.append(target_data)
            # all_targets.append(target['candle_asks']['Close'][0])

            assert len(all_features) == 2 * len(TRADING_LP) * 5 # close high low open volume

        # actual_target = np.mean(all_targets)

        return np.array(all_features), all_targets


class SingleFXDatasetRegression(SingleFXDatasetBase):

    def __init__(self, days, logger, data_args):
        super(SingleFXDatasetRegression, self).__init__(days, logger, data_args)

    def __getitem__(self, index):
        all_features, all_targets =  super().getitem(index)
        
        # Use mean of close as '1'
        cur_sum = 0
        for i in range(0, len(all_features), 5):
            cur_sum += all_features[i][-1]
        avg_last = cur_sum / len(range(0, len(all_features), 5))
        
        # For regressuion let's take the mean of close of bid and asks (num_lp * 2)
        cur_sum = 0
        for lp in range(len(all_targets)):
            cur_sum += all_targets[lp][0] + all_targets[lp][3]
        target = cur_sum / 2 / len(all_targets)
        
        all_features = all_features / avg_last
        target = target / avg_last
        return all_features, target

class SingleFXDatasetBinaryClassification(SingleFXDatasetBase):

    def __init__(self, days, logger, data_args):
        super(SingleFXDatasetBinaryClassification, self).__init__(days, logger, data_args)

    def __getitem__(self, index):
        all_features, all_targets = super().getitem(index)

        # Let's make binary classification be based on whether close of mean of bid and ask
        # if higher/lower than mean of close of bid and ask of last
        # tie = randomly high/low to avoid class bias

        # print(all_targets)
        cur_sum = 0
        for i in range(len(all_targets)):
            cur_sum += all_targets[i][0] + all_targets[i][3]
        future = cur_sum / 2 / len(all_targets)

        cur_sum = 0
        for i in range(0, len(all_features), 5):
            cur_sum += all_features[i][-1]
        last = cur_sum / len(range(0, len(all_features), 5))

        if future > last:
            target = 1
        elif last < future:
            target = 0
        else:
            target = random.random() > 0.5

        return all_features, target


def get_dataloaders(args):

    if args.misc_args.fast_debug:
        train_days, valid_days = convert_train_val_days(TRAIN_DAYS,
                                                        VALID_DAYS)
    else:
        train_days, valid_days = TRAIN_DAYS, VALID_DAYS

    # Cheat to get the dataset class
    dataset_class = globals()[ALL_SETTINGS[args.misc_args.exp_setting]['dataset']]

    # We should force dataset initialization to use the same parameters
    train_dataset = dataset_class(train_days, args.misc_args.logger, args.data_args)
    valid_dataset = dataset_class(valid_days, args.misc_args.logger, args.data_args)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.train_args.batch_size,
                                  shuffle=True,
                                  num_workers=args.data_args.num_workers)
    
    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=args.train_args.batch_size,
                                  shuffle=False,
                                  num_workers=args.data_args.num_workers)
    
    return train_dataloader, valid_dataloader
