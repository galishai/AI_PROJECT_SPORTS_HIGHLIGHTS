import random
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from parsing.id_info.espn_plays import *
import json

data_path = "../../full season data/plays_with_onehot_v2_withoutOT.csv"

free_throw_play_ids = list(range(12,22))

map_quarter_to_int = {'1st': 1,
                  '2nd': 2,
                  '3rd': 3,
                  '4th': 4}

def group_plays():
    play_to_group = {}
    for p in makes_three:
        play_to_group[p] = 'makes_three'
    for p in misses_three:
        play_to_group[p] = 'misses_three'
    for p in makes_two:
        play_to_group[p] = 'makes_two'
    for p in misses_two:
        play_to_group[p] = 'misses_two'
    for p in makes_acrobatic_layups_and_dunks:
        play_to_group[p] = 'makes_acrobatic'
    for p in block:
        play_to_group[p] = 'block'
    for p in rebound:
        play_to_group[p] = 'rebound'
    for p in turnover:
        play_to_group[p] = 'turnover'
    for p in foul:
        play_to_group[p] = 'foul'
    for p in flagrant:
        play_to_group[p] = 'flagrant'
    for p in dramatic_fouls:
        play_to_group[p] = 'dramatic_foul'
    for p in one_ft:
        play_to_group[p] = 'one_ft'
    for p in two_ft:
        play_to_group[p] = 'two_ft'
    for p in three_ft:
        play_to_group[p] = 'three_ft'
    for p in technical_ft:
        play_to_group[p] = 'technical_ft'

    play_to_group_int = {p.value: grp for p, grp in play_to_group.items()}

    return  play_to_group_int


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

def get_star_power(dataset):
    df_h = dataset[dataset['is_highlight'] == 1]

    all_players = pd.concat([
        df_h['name'],
        df_h['assister'],
        df_h['stolen_by']
    ])

    all_players = all_players[all_players != 'Blank']

    star_power = all_players.value_counts()
    star_power.index.name = 'name'
    star_power = star_power.reset_index()
    total_highlights = (dataset['is_highlight'] == 1).sum()
    star_power['count'] = star_power['count'] / total_highlights

    mean, std = star_power['count'].mean(), star_power['count'].std()
    star_power['count'] = (star_power['count'] - mean) / std

    return star_power


def calculate_oncourt_star(row, team_rosters, oncourt_cols, mapping_dict):
    team = row['current_team']
    star_power = 0
    for i, col in enumerate(oncourt_cols):
        if row[col] == 1:  # Player is on court
            if team_rosters[team][i] in mapping_dict.keys():
                star_power += mapping_dict[team_rosters[team][i]]  #
    return star_power

def get_dataset(ds, verbose=False, rm_ft_ds= False, add_game_idx=False, play_context_window=0,team_context_window=0, compact_players=False, compact_oncourt = False, compact_current_team = False, drop_home_away_teams = False, group_all_plays = False, enum_quarters=False):
    dataset = ds.copy()
    total_new_columns = []
    if add_game_idx:
        new_game = (
                (dataset["home_team"] != dataset["home_team"].shift()) |
                (dataset["away_team"] != dataset["away_team"].shift()) |
                (dataset["date"] != dataset["date"].shift())
        )
        dataset["game_id"] = new_game.cumsum()

   #new_columns = []
    for i in range(team_context_window):
        # begin team context

        shifted_after = (
            dataset
            .groupby(['game_id', 'quarter'])['current_team']
            .shift(-(i + 1))
        )

        shifted_before = (
            dataset
            .groupby(['game_id', 'quarter'])['current_team']
            .shift((i + 1))
        )
        original = dataset['current_team']  # ← plain Series
        dataset[f'team_{str(i + 1)}_after'] = original == shifted_after
        dataset[f'team_{str(i + 1)}_after'] = dataset[f'team_{str(i + 1)}_after'].fillna(0).astype(int)
        total_new_columns.append(f'team_{str(i + 1)}_after')

        dataset[f'team_{str(i + 1)}_before'] = original == shifted_before
        dataset[f'team_{str(i + 1)}_before'] = dataset[f'team_{str(i + 1)}_before'].fillna(0).astype(int)
        total_new_columns.append(f'team_{str(i + 1)}_after')

        # end team context
    #if len(new_columns) > 0:
        ##dataset = pd.get_dummies(dataset, columns=new_columns)
    if rm_ft_ds:
        print(f"num rows before removing fts: {len(dataset.index)}")
        dataset = dataset[~dataset["play"].isin(free_throw_play_ids)]
        print(f"num rows after removing fts: {len(dataset.index)}")

    #time to ordinal
    dataset['time_left_qtr'] = dataset['time_left_qtr'].apply(time_to_seconds)


    #dates to ordinal
    '''dataset['date'] = pd.to_datetime(dataset['date'], format='%B %d, %Y')
    first_date = dataset['date'].min()
    dataset['days_since_first_game'] = (dataset['date'] - first_date).dt.days
    if verbose:
        print(f"{aug_count}. num cols: {len(dataset.columns)}")
        aug_count+=1'''
    dataset = dataset.drop(columns=['date'])




    '''#remove unimportant players:
    player_counts = dataset['name'].value_counts()
    common = player_counts[player_counts >= 500].index
    dataset['name'] = dataset['name'].where(dataset['name'].isin(common) | dataset['name'] == 'Blank', 'Other')
    dataset['assister'] = dataset['assister'].where(dataset['assister'].isin(common) | dataset['assister'] == 'Blank', 'Other')
    dataset['stolen_by'] = dataset['stolen_by'].where(dataset['stolen_by'].isin(common) | dataset['stolen_by'] == 'Blank', 'Other')'''
    star_power = get_star_power(dataset)
    mapping = star_power.set_index('name')['count']
    if compact_players:

        dataset['name_star_power'] = dataset['name'].map(mapping).fillna(0).astype(float)

        dataset['assister_star_power'] = dataset['assister'].map(mapping).fillna(
            0).astype(float)

        dataset['stolen_by_star_power'] = dataset['stolen_by'].map(mapping).fillna(
            0).astype(float)

        dataset = dataset.drop(columns=['name', 'assister', 'stolen_by'])
    else:
        total_new_columns += ['name', 'assister', 'stolen_by']
    if compact_oncourt:
        with open('/Users/galishai/PycharmProjects/AI_PROJECT_SPORTS_HIGHLIGHTS/full season data/temp_rosters.json') as json_data:
            team_rosters_full = json.load(json_data)

        oncourt_cols = [c for c in dataset if c.startswith('Oncourt_Player')]

        mapping_dict = mapping.to_dict()

        team_star_power = {team: [0] * 33 for team in team_rosters_full.keys()}

        for team in team_rosters_full.keys():
            #print(team)
            for i, player in enumerate(team_rosters_full[team]):
                if player in mapping_dict.keys():
                    team_star_power[team][i] = mapping_dict[player]
                    #print(f'{player}, power: {mapping_dict[player]}')

        #team_star_power = {team: [0] * 33 for team in team_rosters_full.keys()}

        dataset['oncourt_star_power'] = dataset.apply(calculate_oncourt_star,
                                                          args=(team_rosters_full, oncourt_cols, mapping_dict), axis=1)
        dataset = dataset.drop(columns=oncourt_cols)
    if compact_current_team:
        dataset['current_team_is_home'] = (dataset['home_team'] == dataset['current_team'])
        dataset = dataset.drop(columns=['current_team'])
    else:
        #dataset = pd.get_dummies(dataset, columns=['current_team'])
        total_new_columns += ['current_team']
    if drop_home_away_teams:
        dataset = dataset.drop(columns=['home_team', 'away_team'])
    else:
        total_new_columns += ['home_team', 'away_team']
        #dataset = pd.get_dummies(dataset, columns=['home_team', 'away_team'])
    if group_all_plays:
        plays_to_group = group_plays()

        dataset['play_type'] = (dataset['play'].map(plays_to_group).fillna('other'))
        total_new_columns.append('play_type')
        #new_cols1 = []
        for i in range(play_context_window):
            dataset[f'play_type_{str(i + 1)}_after'] = dataset.groupby(['game_id', 'quarter'])[
                'play'].shift(-(i + 1))
            dataset[f'play_type_{str(i + 1)}_after'] = dataset[f'play_type_{str(i + 1)}_after'].fillna(-1).astype(int)
            dataset[f'play_type_{str(i + 1)}_after'] = dataset[f'play_type_{str(i + 1)}_after'].map(plays_to_group).fillna('other')
            total_new_columns.append(f'play_type_{str(i + 1)}_after')


            dataset[f'play_type_{str(i + 1)}_before'] = dataset.groupby(['game_id', 'quarter'])['play'].shift((i + 1))
            dataset[f'play_type_{str(i + 1)}_before'] = dataset[f'play_type_{str(i + 1)}_before'].fillna(-1).astype(int)

            dataset[f'play_type_{str(i + 1)}_before'] = dataset[f'play_type_{str(i + 1)}_before'].map(plays_to_group).fillna('other')

            total_new_columns.append(f'play_type_{str(i + 1)}_before')

        dataset = dataset.drop(columns=['play'])
        #if len(new_cols1) > 0:
            #dataset = pd.get_dummies(dataset, columns = new_cols1)
    else:
        #new_cols2 = []
        for i in range(play_context_window):
            dataset[f'play_{str(i + 1)}_after'] = dataset.groupby(['game_id', 'quarter'])['play'].shift(-(i + 1))
            dataset[f'play_{str(i + 1)}_after'] = dataset[f'play_{str(i + 1)}_after'].fillna(-1).astype(int)
            total_new_columns.append(f'play_{str(i + 1)}_after')
            dataset[f'play_{str(i + 1)}_before'] = dataset.groupby(['game_id', 'quarter'])['play'].shift((i + 1))
            dataset[f'play_{str(i + 1)}_before'] = dataset[f'play_{str(i + 1)}_before'].fillna(-1).astype(int)
            total_new_columns.append(f'play_{str(i + 1)}_before')
        total_new_columns.append('play')
        #dataset = pd.get_dummies(dataset, columns=['play'] + new_cols2)
        # quarter to categorical
    if enum_quarters:
        dataset['quarter'] = dataset['quarter'].map(map_quarter_to_int)
    else:
        total_new_columns.append('quarter')
        #dataset = pd.get_dummies(dataset, columns=['quarter'])
    if len(total_new_columns) > 0:
        dataset = pd.get_dummies(dataset, columns=total_new_columns)


    return dataset