import numpy as np

OHLC = ['Open', 'High', 'Low', 'Close']


def data_normalize(df, pairs, lps,
                   price_means, price_stds, volume_means, volume_stds):
    '''Normalize all currency pairs.
    All LPs and OHLC are used together to compute mean/std
    '''

    for i, pair in enumerate(pairs):
        for lp in lps:
            price_columns = [f'{pair}_{k}_{lp + 1}' for k in OHLC]
            volume_columns = [f'{pair}_Volume_{lp + 1}']

            df[price_columns] = (df[price_columns] - price_means[i]) / price_stds[i]
            df[volume_columns] = (df[volume_columns] - price_means[i]) / price_stds[i]

    return df


# Example of how to preprocess future candles for classification or ordinal regression
def generate_target(api, pair, lps,
                    slot_end, lookahead_end,
                    pair_mean, pair_std):
    '''Prediction of future mid-price'''

    if np.isnan(pair_std):
        raise ValueError(f'Encountered 0 std or nan when generating target')

    mean_mean = 0
    for i in lps:
        day_df = api.get(pair, str(i + 1),
                         start_time=slot_end,
                         length=None,
                         end_time=lookahead_end,
                         verbose=False)
        
        close_name = None
        for c in day_df.columns:
            if 'Close' in c:
                close_name = c
                break

        # mean = np.mean((day_df['bid price']  + day_df['ask price']) / 2)
        mean = np.mean(day_df[close_name])
        mean_mean += mean
    mean_mean = mean_mean / len(lps)
    mean_mean = (mean_mean - pair_mean) / pair_std
    
    if np.isnan(mean_mean):
        raise ValueError(f'Encountered 0 std or nan when generating target')

    target = int(mean_mean)
    target = int(round((min(max(-3, target), 3)) + 3))    # threshold to -3 ~ 3

    return target

