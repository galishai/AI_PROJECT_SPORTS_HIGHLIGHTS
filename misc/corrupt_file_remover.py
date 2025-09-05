import csv
import re
import os
from tqdm import tqdm
import pandas as pd


class NBAPlayIterator:
    """
    Iterator to go through games and their plays in an NBA dataset.
    Groups plays by unique games identified by 'date', 'home_team', and 'away_team',
    preserving the order of rows as in the CSV file.
    """
    def __init__(self, data):
        self.data = data
        # Group by 'date', 'home_team', and 'away_team' without sorting
        self.grouped = data.groupby(['date', 'home_team', 'away_team'], sort=False)
        self.games = iter(self.grouped)  # Iterator for grouped games
        self.current_game = None
        self.current_plays = None
        self.current_plays_iterator = None

    def __iter__(self):
        return self

    def __next__(self):
        # If no plays or exhausted, fetch the next game
        if self.current_plays_iterator is None:
            self.current_game, plays = next(self.games)  # Get the next game and its plays
            self.current_plays = plays
            self.current_plays_iterator = iter(plays.iterrows())  # Create a fresh iterator for plays

        try:
            # Return the next play in the current game
            _, play = next(self.current_plays_iterator)
            return self.current_game, play
        except StopIteration:
            # Exhausted current game's plays; reset for the next game
            self.current_plays_iterator = None
            return self.__next__()

    def games_only(self):
        """
        Generator to iterate over games without processing individual plays.
        """
        for game, plays in self.grouped:
            yield game

    def plays_for_game(self, game):
        """
        Generator to iterate over the plays of a specific game.
        :param game: A tuple (date, home_team, away_team) identifying the game.
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


def check_quarter_input(quarter):
    qurters = [1, 2, 3, 4, 'FIRST', 'SECOND', 'THIRD', 'FOURTH']
    if quarter.startswith('FIRST'):
        return True, 1
    if quarter.startswith('SECOND'):
        return True, 2
    if quarter.startswith('THIRD'):
        return True, 3
    if quarter.startswith('FOURTH'):
        return True, 4
    if not quarter.isdigit() and quarter not in qurters:
        return False, 0
    if quarter.isdigit() and int(quarter) not in qurters:
        return False, 0
    return True, 0

def check_time_input(time):
    times = time.split(':')
    if len(times) != 2:
        return False
    if not times[0].isdigit() or not times[1].isdigit() or int(times[0]) >= 12 or int(times[1]) >= 59:
        return False
    return True

file_size_limit = 1500

if __name__ == "__main__":

    ## IMPORTANT: For safety run on a copy of the original text file folder
    txt_files_path = '/Users/galishai/Desktop/ocr_results/gamestxt_without_corruption'
    csv_file_path = '/old_outputs/output_full_season_test.csv'
    output_csv_path = '/Labling from text file/new_output_filtered.csv'
    txt_files_name_list = list_files_in_directory(txt_files_path)
    txt_files_path_list = [txt_files_path + '/' + file_name for file_name in txt_files_name_list]
    csv_data = pd.read_csv(csv_file_path)
    iterator = NBAPlayIterator(csv_data)
    with open(output_csv_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(csv_data.columns)
        for game, txt_file_path in tqdm(zip(iterator.games_only(), txt_files_path_list), total=len(txt_files_path_list)):
            if os.path.isfile(txt_file_path):
                file_size = os.path.getsize(txt_file_path)
                if file_size < file_size_limit:
                    os.remove(txt_file_path)
                    print(f"Deleted {txt_file_path} (size {file_size} bytes)")
                    continue
                for play in iterator.plays_for_game(game):
                    writer.writerow(play)
    new_csv_data = pd.read_csv(output_csv_path)
    old_iterator = NBAPlayIterator(csv_data)
    new_iterator = NBAPlayIterator(new_csv_data)
    count_before = 0
    count_after = 0
    for game in old_iterator.games_only():
        count_before += 1
    for game in new_iterator.games_only():
        count_after += 1
    print("before: " + str(count_before))
    print("after: " + str(count_after))
    print("deleted: " + str(count_before - count_after))

    after_txt_files_name_list = list_files_in_directory(txt_files_path)
    print("text files after: " + str(len(after_txt_files_name_list)))
