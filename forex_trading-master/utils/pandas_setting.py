import pandas as pd


def set_display_options(column=None, rows=None, col_width=-1):
    pd.set_option('display.max_columns', column)        # or 1000
    pd.set_option('display.max_rows', rows)             # or 1000
    pd.set_option('display.max_colwidth', col_width)    # or 199