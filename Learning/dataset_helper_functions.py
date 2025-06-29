import random
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

free_throw_play_ids = list(range(12,22))



def freeze_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def time_to_seconds(time_str):
    if ':' in time_str:
        minutes, seconds = map(int, time_str.split(':'))
        return minutes * 60 + seconds
    else:
        return int(float(time_str))

def seconds_to_time(seconds):
    minutes = seconds // 60
    remaining_seconds = seconds % 60
    return f"{minutes}:{remaining_seconds:02d}"


def get_dataset(path, verbose=False, rm_ft_ds= False, add_game_idx=False):
    dataset = pd.read_csv(path)
    if add_game_idx:
        new_game = (
                (dataset["home_team"] != dataset["home_team"].shift()) |
                (dataset["away_team"] != dataset["away_team"].shift()) |
                (dataset["date"] != dataset["date"].shift())
        )
        dataset["game_id"] = new_game.cumsum()
    if rm_ft_ds:
        print(f"num rows before removing fts: {len(dataset.index)}")
        dataset = dataset[~dataset["play"].isin(free_throw_play_ids)]
        print(f"num rows after removing fts: {len(dataset.index)}")
    aug_count = 0
    if verbose:
        print(f"{aug_count}. num cols: {len(dataset.columns)}")
        aug_count+=1
    #quarter to categorical
    dataset = pd.get_dummies(dataset, columns=['quarter'])
    if verbose:
        print(f"{aug_count}. num cols: {len(dataset.columns)}")
        aug_count+=1

    #time to ordinal
    dataset['time_left_qtr'] = dataset['time_left_qtr'].apply(time_to_seconds)
    if verbose:
        print(f"{aug_count}. num cols: {len(dataset.columns)}")
        aug_count+=1

    #dates to ordinal
    dataset['date'] = pd.to_datetime(dataset['date'], format='%B %d, %Y')
    first_date = dataset['date'].min()
    dataset['days_since_first_game'] = (dataset['date'] - first_date).dt.days
    if verbose:
        print(f"{aug_count}. num cols: {len(dataset.columns)}")
        aug_count+=1
    dataset = dataset.drop(columns=['date'])
    if verbose:
        print(f"{aug_count}. num cols: {len(dataset.columns)}")
        aug_count+=1

    #convert plays to categorical
    dataset = pd.get_dummies(dataset, columns=['play'])
    if verbose:
        print(f"{aug_count}. num cols: {len(dataset.columns)}")
        aug_count+=1

    #teams to categorical
    dataset = pd.get_dummies(dataset, columns=['home_team', 'away_team', 'current_team'])
    if verbose:
        print(f"{aug_count}. num cols: {len(dataset.columns)}")
        aug_count+=1

    #player names to categorical
    dataset = pd.get_dummies(dataset, columns=['name', 'assister', 'stolen_by'])
    if verbose:
        print(f"{aug_count}. num cols: {len(dataset.columns)}")
        aug_count+=1

    return dataset



