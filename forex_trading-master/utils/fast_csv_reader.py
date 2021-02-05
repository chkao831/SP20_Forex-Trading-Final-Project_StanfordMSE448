import csv
from collections import defaultdict
from dateutil import parser


def read_csv(fpath, date_col, cutoff_time):

    result = defaultdict(list)
    date_okay = True
    with open(fpath, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            row_data = {}
            for k in row:
                if k == date_col:
                    try:
                        this_date = parser.parse(row[k], dayfirst=False)
                    except:
                        print(f'Unable to parse {row[k]}')
                        break
                    if this_date > cutoff_time:
                        date_okay = False
                        break
                    # result[k].append(this_date)
                    row_data[k] = this_date
                elif k == 'bid volume' or k == 'ask volume' or \
                     k == 'bid price' or k == 'ask price':
                    # result[k].append(float(row[k]))
                    row_data[k] = float(row[k])
                else:
                    # result[k].append(row[k])
                    row_data[k] = row[k]

            if not date_okay:
                break

            if len(row_data) == len(row):
                for k in row_data:
                    result[k].append(row_data[k])
    
    arr = [len(result[r]) for r in result]
    if len(arr) == 0:
        raise ValueError(f'Path {fpath} contains invalid file')
    min_dict_len = min(arr)
    result = {r: result[r][:min_dict_len] for r in result}

    return result