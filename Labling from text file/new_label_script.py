import csv
import re
import os
from re import split

from tqdm import tqdm
import pandas as pd

EARLY_STOPPING_INDEX = -1

HIGHLIGHT_CONFIDENCE_THRESH = 2

GAME_CONFIDENCE_THRESH = 7

PRINT_HIGHLIGHTS = 0

SHOW_ERRORS = 0

class PlayIterator:

    def __init__(self, data):
        self.data = data
        # group by 'date', 'home_team', and 'away_team' without sorting
        self.grouped = data.groupby(['date', 'home_team', 'away_team'], sort=False)
        self.games = iter(self.grouped)  # iter for grouped games
        self.current_game = None
        self.current_plays = None
        self.current_plays_iterator = None

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_plays_iterator is None:
            self.current_game, plays = next(self.games)
            self.current_plays = plays
            self.current_plays_iterator = iter(plays.iterrows())

        try:
            _, play = next(self.current_plays_iterator)
            return self.current_game, play
        except StopIteration:
            self.current_plays_iterator = None
            return self.__next__()

    def games_only(self):
        """
        only iterate over games
        """
        for game, plays in self.grouped:
            yield game

    def quarters_for_game(self, game):
        """
        generator to iterate over quarters
        """
        if game in self.grouped.groups:
            game_data = self.grouped.get_group(game)
            grouped_quarters = game_data.groupby(['quarter'], sort=False)
            for quarter, plays in grouped_quarters:
                yield quarter, plays
        else:
            raise ValueError(f"Game {game} not found in the dataset.")

    def plays_for_quarter(self, game, quarter):
        """
        generator to iterate over plays in quarter
        """
        if game in self.grouped.groups:
            game_data = self.grouped.get_group(game)
            quarter_data = game_data[game_data['quarter'] == quarter]
            for _, play in quarter_data.iterrows():
                yield play
        else:
            raise ValueError(f"Game {game} not found in the dataset.")

    def plays_for_game(self, game):
        """
        generator to iterate over plays in game
        """
        if game in self.grouped.groups:
            plays = self.grouped.get_group(game)
            for _, play in plays.iterrows():
                yield play
        else:
            raise ValueError(f"Game {game} not found in the dataset.")

def alphanum_key(s):
    return [int(text) if text.isdigit() else float(text) if re.match(r'^[0-9]+\.[0-9]+$', text) else text.lower() for text in re.split('(\d+\.\d+|\d+)', s)]

def list_files_in_directory(directory):
    try:
        # Get the list of files and directories
        entries = os.listdir(directory)

        # Filter out only files
        files = [entry for entry in entries if os.path.isfile(os.path.join(directory, entry)) and not
        entry.startswith('.')]
        files.sort(key=alphanum_key)
        return files
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def reformat_time(time):
    temp_time = time
    temp_time = temp_time.replace(' ', '')

    if temp_time.startswith("."):
        temp_time = temp_time[1:]
    if temp_time.endswith("."):
        temp_time = temp_time[:-1]
    if temp_time.startswith(':'):
        temp_time = temp_time[1:]
        if temp_time.startswith(':'):
            temp_time = temp_time[1:]
        temp_time = temp_time.split('.')[0]
    if temp_time.endswith(":"):
        temp_time = temp_time[:-1]
    if ':' not in temp_time:
        if len(temp_time.split('.')[0]) > 2 and temp_time.split('.')[0][0] == '1':
            temp_time = temp_time[1:]
        temp_time = '0:' + temp_time.split('.')[0]
    return temp_time

def quarter_to_csv_format(quarter):
    if quarter == '1' or quarter.startswith('FIRS'):
        return '1st'
    if quarter == '2' or quarter.startswith('SEC'):
        return '2nd'
    if quarter == '3' or quarter.startswith('THI'):
        return '3rd'
    if quarter == '4' or quarter.startswith('FOU'):
        return '4th'
    if quarter == 'OT':
        return 'OT'
    if quarter == '2OT':
        return '2OT'
    print("cant_id")
    assert 1 == 0


def time_to_int(time):
    try:
        minutes, seconds = map(int, time.split(":"))
        total_seconds = minutes * 60 + seconds
        return total_seconds
    except:
        print(time)
        assert 1==0

def check_quarter_input(quarter):
    qurters = ['1', '2', '3', '4', 'OT', '2OT', '0T', '20T', 'FIRST', 'SECOND', 'THIRD', 'FOURTH', 'OVERTIME']
    quarter = quarter.replace(' ','')
    if quarter.startswith('FIR'):
        return True, '1'
    if quarter.startswith('SEC'):
        return True, '2'
    if quarter.startswith('THI'):
        return True, '3'
    if quarter.startswith('FOU'):
        return True, '4'
    if quarter.startswith('OVER'):
        return True, 'OT'
    if quarter in qurters:
        return True, quarter
    if not quarter.isdigit() and quarter not in qurters:
        #print(f"NON DIGIT: {quarter}")
        return False, 0
    if quarter.isdigit() and int(quarter) not in qurters:
        if quarter[0] in qurters:
            return True, quarter[0]
    return False, 0

def check_time_input(time):
    times = time.split(':')
    if len(times) != 2:
        return False
    if (not times[0].isdigit() or not times[1].isdigit()) or ( int(times[0]) > 12 or int(times[1]) >= 60):
        return False
    return True

def make_game_intervals(iterator):
    game_intervals = []
    for game_count, game in enumerate(iterator.games_only()):
        curr_game_intervals = []
        for quarter_count, quarter in enumerate(iterator.quarters_for_game(game)):
            interval_start = '12:00' #time_to_int('12:00')
            quarter_txt = quarter[0][0]
            for play_count, play in enumerate(iterator.plays_for_quarter(game, quarter_txt)):
                play_time = play['time_left_qtr']
                reformatted_time = reformat_time(play_time)
                interval_end = reformatted_time #time_to_int(reformatted_time)
                curr_interval = (interval_start, interval_end, quarter_txt)
                curr_game_intervals.append(curr_interval)
                interval_start = interval_end
        game_intervals.append(curr_game_intervals)
        if game_count == EARLY_STOPPING_INDEX:
            return game_intervals
    return game_intervals

def read_text(file_path):
    with open(file_path, mode='r', encoding='utf-8') as file:
        data = file.read().splitlines()
    cleaned_data = []
    for line in data:
        line = line.replace("(", "").replace(")", "").replace("\\n", "").replace("'", "")
        parts = line.split(",")
        if len(parts) < 2:
            continue
        time = parts[0].strip()
        #time = reformat_time(time)
        quarter = parts[1].strip()
        cleaned_data.append([time, quarter])
    return cleaned_data

def print_highlights(game_interval_list, interval_hist):
    assert len(game_interval_list) == len(interval_hist)
    print("HIGHLIGHTS:")
    for ivl, confidence_count in zip(game_interval_list, interval_hist):
        if confidence_count >= HIGHLIGHT_CONFIDENCE_THRESH:
            print(f"interval: {ivl}, confidence: {confidence_count}")

def print_histogram(game_interval_list, interval_hist):
    assert len(game_interval_list) == len(interval_hist)
    print("HISTOGRAM RESULTS:")
    for ivl, confidence_count in zip(game_interval_list, interval_hist):
        print(f"interval: {ivl}, confidence: {confidence_count}")




if __name__ == "__main__":

    ## IMPORTANT: For safety run on a copy of the original text file folder
    txt_files_path = '/Users/galishai/Desktop/ocr_results/gamestxt_without_corruption'
    csv_file_path = '/Users/galishai/PycharmProjects/AI_PROJECT_SPORTS_HIGHLIGHTS/Labling from text file/new_output.csv'
    txt_files_name_list = list_files_in_directory(txt_files_path)
    txt_files_path_list = [txt_files_path + '/' + file_name for file_name in txt_files_name_list]
    csv_data = pd.read_csv(csv_file_path)
    iterator = PlayIterator(csv_data)
    game_intervals = make_game_intervals(iterator)
    game_index = 0
    games_above_thresh = 0
    failed_games = []
    for game_interval_list, txt_file_path in zip(game_intervals, txt_files_path_list):
        print(f"game index: {game_index}, txt_path: {txt_file_path}")
        interval_hist = [0]* len(game_interval_list) #histogram of times found in each interval
        text_data = read_text(txt_file_path)
        highlights_above_thresh = 0
        for txt_time, txt_quarter in text_data:
            prev_time = -1
            tmp_time = reformat_time(txt_time)
            time_check = check_time_input(tmp_time)
            qtr_check, qtr = check_quarter_input(txt_quarter)
            if not time_check or not qtr_check:
                 if SHOW_ERRORS:
                    print(f"time: {tmp_time}, time_check: {time_check}, qtr: {txt_quarter}, qtr_check: {qtr_check}")
                 continue
            int_txt_time = time_to_int(tmp_time)
            qtr_csv_format = quarter_to_csv_format(qtr)
            for i, interval in enumerate(game_interval_list):
                tmp_start_time, tmp_end_time, interval_qtr = interval
                sanity_start = reformat_time(tmp_start_time)
                sanity_end = reformat_time(tmp_end_time)
                int_start_time = time_to_int(sanity_start)
                int_end_time = time_to_int(sanity_end)
                if txt_time == '.47.3' and interval_qtr == '2nd':
                    a = 1
                if int_start_time != int_end_time:
                    int_start_time -= 1
                if int_end_time <= int_txt_time <= int_start_time and qtr_csv_format == interval_qtr:
                    if int_start_time == int_end_time:
                        interval_hist[i] += HIGHLIGHT_CONFIDENCE_THRESH
                    else:
                        if int_txt_time == prev_time:
                            continue
                        else:
                            interval_hist[i] += 1
                            prev_time = int_txt_time

        for ivl, confidence_count in zip(game_interval_list, interval_hist):
            if confidence_count >= HIGHLIGHT_CONFIDENCE_THRESH:
                highlights_above_thresh += 1
        if highlights_above_thresh >= GAME_CONFIDENCE_THRESH:
            games_above_thresh += 1
        else:
            failed_games.append("game index: " + str(game_index) + ", txt_path: " + txt_file_path + "highlights found: " + str(highlights_above_thresh))
            print_histogram(game_interval_list, interval_hist)
        if PRINT_HIGHLIGHTS:
            print_highlights(game_interval_list, interval_hist)
        if game_index==EARLY_STOPPING_INDEX:
            break
        game_index += 1
    print(f"Total Games Processed: {len(txt_files_name_list)}")
    print(f"Games above thresh={GAME_CONFIDENCE_THRESH}: {games_above_thresh}")
    print(f"Games below thresh: {len(txt_files_name_list) - games_above_thresh}")
    print("Failed games:")
    print(failed_games)

