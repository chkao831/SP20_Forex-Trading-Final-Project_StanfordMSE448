import re
import math
import datetime

import pandas as pd


units = [('hour', 3600000000),
         ('minute', 60000000),
         ('second', 1000000),
         ('microsecond', 1)]
full_day = 86400000000


def microsecond_to_hms(micro):

    cur = micro
    to_replace = {}
    for u, l in units:
        amount = cur // l
        left_over = cur % l
        to_replace[u] = amount
        cur = left_over

    return to_replace


def hms_to_microsecond(hms):

    micro = 0
    for u, l in units:
        micro += getattr(hms, u) * l

    return micro


def get_start_time(time, interval):
    
    micro = 0
    if 'ms' in interval:
        value = int(interval[:-2])
        micro = value * 1000
    elif 'm' in interval:
        value = int(interval[:-1])
        micro = value * 1000000 * 60
    elif 's' in interval:
        value = int(interval[:-1])
        micro = value * 1000000

    delta = datetime.timedelta(microseconds=micro)

    init_micro = hms_to_microsecond(time)
    rounded_micro = int(math.ceil(init_micro / micro)) * micro
    rounded_hms = microsecond_to_hms(rounded_micro)
    rounded = time.replace(**rounded_hms)

    
    return rounded, delta


'''
def get_start_time(earliest, interval):

    if 'ms' in interval:
        value = int(interval[:-2])
        cur = earliest.microsecond / 1000 / value
        rounded = int(math.ceil(cur) * 1000) * value
        rounded = earliest.replace(microsecond=rounded)
        delta = datetime.timedelta(milliseconds=value)
    elif 'm' in interval:
        value = int(interval[:-1])
        cur = earliest.minute / value
        rounded = int(math.ceil(cur)) * value
        rounded = earliest.replace(minute=rounded, second=0, microsecond=0)
        delta = datetime.timedelta(minutes=value)
    elif 's' in interval:
        value = int(interval[:-1])
        cur = earliest.second / value
        rounded = int(math.ceil(cur)) * value
        rounded = earliest.replace(second=rounded, microsecond=0)
        delta = datetime.timedelta(seconds=value)
    return rounded, delta
'''


def generate_candles(df, interval, bma, get_df=True):

    bma = bma.lower()
    if bma == 'bid price':
        prices = df['bid price']
        all_vol = df['bid volume']
    elif bma == 'ask price':
        prices = df['ask price']
        all_vol = df['ask volume']
    elif bma == 'mid price':
        prices = (df['bid price'] + df['ask price']) / 2
        all_vol = (df['bid volume'] + df['ask volume']) / 2
    else:
        raise ValueError

    all_time = list(df['time'])
    all_vol = list(all_vol)
    prices = list(prices)

    rounded_start_time, delta = get_start_time(all_time[0], interval)
    
    start_i = None
    for i in range(len(all_time)):
        if all_time[i] > rounded_start_time:
            start_i = i
            break
    
    # print(f'Actual start time is {rounded_start_time}')

    if start_i is None:
        raise ValueError(f'Unable to locate desired start time: {rounded_start_time}')

    candles = []
    open_price = prices[0]
    high_price = prices[0]
    low_price = prices[0]
    close_price = prices[0]
    volume = 0
    next_target = rounded_start_time + delta
    i = start_i

    while i < len(all_time):
        ct = all_time[i]
        cp = prices[i]
        cv = all_vol[i]

        if cp < 0.1:
            cp = prices[i - 1]
            # continue

        # print(ct, next_target)

        if ct < next_target:
            # print('\t', ct, next_target)
            high_price = max(high_price, cp)
            low_price = min(low_price, cp)
            volume += cv
            close_price = cp
        else:
            candles.append({'High': high_price, 'Low': low_price,
                            'Open': open_price, 'Close': close_price,
                            'Volume': volume, 'Date': next_target - delta})
            open_price = cp
            high_price = cp
            low_price = cp     # None of the currencies are this high anyways
            close_price = cp
            volume = 0
            next_target = next_target + delta

        if ct < next_target:
            i += 1
            # pbar.update()

    if get_df:
        df = pd.DataFrame(candles)
        df = df.set_index('Date')
        return df
    else:
        return candles


def pad_candle(candles, interval, start_time, end_time):

    target_number = (end_time - start_time) / interval

    assert len(candles) <= target_number, f'Too many candles {len(candles)} to begin with for ({start_time} {end_time}). {candles}'
    
    first_time = candles[0]['Date']
    last_time = candles[-1]['Date']

    # Prepend to candles
    valid = candles[0]

    insert = {k: [] for k in candles[0]}
    
    did_prepend, did_append = False, False
    should_prepend, should_append = first_time > start_time, last_time + interval < end_time
    
    cur_time = start_time
    while cur_time < first_time:
        did_prepend = True
        insert['Date'].append(cur_time)
        for k in valid:
            if k == 'Volume':
                insert[k].append(0)
            elif k != 'Date':
                insert[k].append(valid['Open'])
        cur_time += interval

    for i in range(len(insert['Date'])):
        candles.insert(i, {k: insert[k][i] for k in insert})

    # Append to candles
    valid = candles[-1]
    insert = {k: [] for k in candles[-1]}

    cur_time = last_time + interval
    prepend_end = cur_time
    while cur_time < end_time:
        did_append = True
        insert['Date'].append(cur_time)
        for k in valid:
            if k == 'Volume':
                insert[k].append(0)
            elif k != 'Date':
                insert[k].append(valid['Close'])
        cur_time += interval

    for i in range(len(insert['Date'])):
        candles.append({k: insert[k][i] for k in insert})
    
    # Return reformatted candles
    df = pd.DataFrame(candles)
    df = df.set_index('Date')

    assert len(df) == target_number, f'For ({(interval, start_time, end_time, first_time)}), should have {target_number} candles, got {len(df)}\n{candles} instead. Did {(should_prepend, should_append, did_prepend, did_append, first_time, prepend_end)},\n{df}'
    
    return df


if __name__ == '__main__':

    start = datetime.datetime(2020, 2, 26, 4, 59, 59, )

    print(f'Orig Time: {start}')
    
    def check(interval):
        rounded = round_time_up(start, interval)
        print(f'Next Time ({interval.ljust(6)}): {rounded}')

    check('10ms')
    check('100ms')
    check('500ms')
    check('1s')
    check('5s')
    check('10s')
    check('1m')
    check('5m')
    check('10m')
    

    