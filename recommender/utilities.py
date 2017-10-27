"""Utility functions for the project."""
import time
import pandas as pd

from datetime import datetime


def timer(start=None, to_var=False):
    """Print or Return time spent in a function.

    Args:
        start(optional, float): the timestamp at a moment

    Returns:
        float: the timestamp at a moment if not time was given

    """
    if not start:
        print(datetime.now().ctime())
        return time.time()
    stop = time.time()
    m, s = divmod(stop - start, 60)
    h, m = divmod(m, 60)
    if to_var:
        return '{}:{}:{}'.format(int(h), int(m), round(s))
    print('total time {}:{}:{}'.format(int(h), int(m), round(s)))


def csv_to_dictionary(train_csv, group_by):
    """Convert the ratings csv file to dictionary.

    Read a csv file with columns ['user', 'item', 'rating'] and based on the
    group_by parameter the function uses the user or item as keys.

    Args:
        train_csv(str): The train file name to read
        group_by(str): The grouping parameter 'user' or 'item'

    Returns:
        dict: The train data in a dictionary form with keys either
        user ids or item ids

    """
    start = timer()

    df = pd.read_csv(train_csv, sep=',', usecols=['user', 'item', 'rating'])

    other = 'item'
    if group_by == other:
        other = 'user'

    unique_keys = df[group_by].unique()
    train = dict.fromkeys(unique_keys)
    for i in unique_keys:
        train[i] = {}

    for u_m, m_u, r in zip(df[group_by].values, df[other].values,
                           df.rating.values):
        train[u_m][m_u] = r
    timer(start)
    return train
