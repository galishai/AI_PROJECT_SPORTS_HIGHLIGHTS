import csv
import re
from tqdm import tqdm
import os

ADD_LABEL_COLUMN = False

def read_csv(file_path):
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        data = [row for row in reader]
    return data

def write_to_csv(file, data):
    if not data:
        print("No data to write.")
        return

    with open(file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = data[0].keys()
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

def write_list_to_file(file_path, list):
    with open(file_path, mode='w', encoding='utf-8') as file:
        for element in list:
            file.write(f"{element}\n")

def reformat_time(time):
    temp_time = time
    if temp_time.startswith("."):
        temp_time = temp_time[1:]
    if temp_time.startswith(':'):
        temp_time = temp_time[1:].split('.')[0]
    if ':' not in temp_time:
        temp_time = '0:' + temp_time.split('.')[0]
    return temp_time

def check_time_input(time):
    times = time.split(':')
    if len(times) != 2:
        return False
    if not times[0].isdigit() or not times[1].isdigit() or int(times[0]) >= 12 or int(times[1]) >= 59:
        return False
    return True

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

def create_intervals(text_data):
    intervals = []
    start_time = None
    end_time = None
    start_quarter = None

    for time, quarter in text_data:
        time = reformat_time(time)
        try:
            if not check_quarter_input(quarter)[0]:
                continue
            if not check_time_input(time):
                continue
            minutes, seconds = map(int, time.split(":"))
            total_seconds = minutes * 60 + seconds
        except ValueError:
            continue

        if start_time is None:
            start_time = total_seconds
            end_time = total_seconds
            start_quarter = quarter
        else:
            if quarter == start_quarter and end_time - total_seconds <= 2 and end_time >= total_seconds:
                end_time = total_seconds
            else:
                intervals.append(((start_time, end_time), start_quarter))
                start_time = total_seconds
                end_time = total_seconds
                start_quarter = quarter

    if start_time is not None and end_time is not None:
        intervals.append(((start_time, end_time), start_quarter))

    return intervals

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
        time = reformat_time(time)
        quarter = parts[1].strip()
        flag, q = check_quarter_input(quarter)
        if not flag and q == 0:
            continue
        elif flag and q == 1:
            quarter = '1'
        elif flag and q == 2:
            quarter = '2'
        elif flag and q == 3:
            quarter = '3'
        elif flag and q == 4:
            quarter = '4'
        if not check_time_input(time):
            continue
        if re.match(r'^\d+:\d+$', time) and re.match(r'^\d+$', quarter):
            cleaned_data.append([time, quarter])
    return cleaned_data

def add_column(csv_data, column_name, default_value):
    for row in csv_data:
        row[column_name] = default_value
    return csv_data

def is_in_interval(time, quarter, intervals):
    if '.' in time:
        temp_time = time.split('.')[0]
        time = f"0:{temp_time}"
    if not re.match(r'^\d+:\d+$', time):
        return False
    minutes, seconds = map(int, time.split(":"))
    total_seconds = minutes * 60 + seconds
    for (start, end), q in intervals:
        if start >= total_seconds and total_seconds >= end and quarter == q:
            return True
    return False

def tag_moves_for_game(iterator, intervals):
    previous_row = None
    count = 0
    update_rows = []
    try:
        first_row = iterator.peek()
    except StopIteration:
        return

    if is_in_interval(first_row['time_left_qtr'], first_row['quarter'][0], intervals):
        first_row['Label'] = 1
        count += 1
    else:
        first_row['Label'] = 0
    update_rows.append(first_row)

    for row in iterator:
        if previous_row is None or (previous_row['home_team'] == row['home_team'] and previous_row['away_team'] == row['away_team'] and previous_row['date'] == row['date']):
            if is_in_interval(row['time_left_qtr'], row['quarter'][0], intervals):
                row['Label'] = 1
                count += 1
            else:
                row['Label'] = 0
        else:
            break
        update_rows.append(row)

        previous_row = row
    return update_rows, count

def extract_game_number(filename):
    match = re.search(r'frames_Game_Recap_(\d+)_', filename)
    return int(match.group(1)) if match else float('inf')

def write_csv(file_path, data):
    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(data)

class GamesIterator:
    def __init__(self, file):
        self.file = file
        self.reader = None
        self.previous_row = None
        self.file_open = open(file, newline='', encoding='utf-8')
        self.reader = csv.DictReader(self.file_open)
        self.current_game_rows = []

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_game_rows:
            return self.current_game_rows.pop(0)

        for row in self.reader:
            if self.previous_row is None or (
                    self.previous_row['home_team'] == row['home_team'] and self.previous_row['away_team'] == row[
                'away_team'] and self.previous_row['date'] == row['date']):
                self.current_game_rows.append(row)
            else:
                self.previous_row = row
                return self.current_game_rows.pop(0)

        if self.current_game_rows:
            return self
        raise StopIteration

    def peek(self):
        return self.current_game_rows[0] if self.current_game_rows else None

csv_file_path = 'C:/Users/Sahar/Desktop/Computer Science/236502 Artificial inetlligence project/git_files/Labling from text file/output_full_season_v4.csv'
text_folder_path = 'C:/Users/Sahar/Desktop/Computer Science/236502 Artificial inetlligence project/git_files/full season data/ocr_text_files'

csv_data = read_csv(csv_file_path)
text_files = sorted([os.path.join(text_folder_path, f) for f in os.listdir(text_folder_path) if f.endswith('.txt')], key=extract_game_number)
iterator = GamesIterator(csv_file_path)
games_counter = []

if ADD_LABEL_COLUMN:
    add_column(csv_data, 'Label', 0)
    write_to_csv(csv_file_path, csv_data)

for i, game in enumerate(tqdm(iterator, desc="Processing games")):
    text_file_path = text_files[i % len(text_files)]
    text_data = read_text(text_file_path)
    intervals_list = create_intervals(text_data)
    game_rows, count = tag_moves_for_game(iterator, intervals_list)
    if game_rows:
        games_counter.append((game_rows[0]['home_team'] + ' vs ' + game_rows[0]['away_team'] + ' ' + game_rows[0]['date'],count))
        #print(f"finished labeling: {game_rows[0]['home_team']} vs {game_rows[0]['away_team']} {game_rows[0]['date']}")
        """         #remove comment if you want to write to original CSV file
        for updated_row in game_rows:
            for original_row in csv_data:
                if (original_row['home_team'] == updated_row['home_team'] and
                    original_row['away_team'] == updated_row['away_team'] and
                    original_row['date'] == updated_row['date'] and
                    original_row['time_left_qtr'] == updated_row['time_left_qtr'] and
                    original_row['quarter'] == updated_row['quarter']):
                    original_row['Label'] = updated_row['Label']
write_to_csv(csv_file_path, csv_data)
"""
threshold = 7
count_below_threshold = sum(1 for game in games_counter if game[1] <= threshold)
print(f"Number of games below threshold:{count_below_threshold}")
write_list_to_file('C:/Users/Sahar/Desktop/Computer Science/236502 Artificial inetlligence project/git_files/Labling from text file/label_counter.txt', games_counter)









"""
for i, game in enumerate(iterator):
    if isinstance(game, GamesIterator):
        if game.current_game_rows[0]['quarter'] == 'FIRST':
            print("Debug now")
    text_file_path = text_files[i % len(text_files)]
    text_data = read_text(text_file_path)
    print(i)
    print(create_intervals(text_data))


write_list_to_file('C:/Users/Sahar/Desktop/Computer Science/236502 Artificial inetlligence project/git_files/Labling from text file/label_counter.txt', games_counter)
 
    for updated_row in game_rows:
        for original_row in csv_data:
            if (original_row['home_team'] == updated_row['home_team'] and
                original_row['away_team'] == updated_row['away_team'] and
                original_row['date'] == updated_row['date'] and
                original_row['time_left_qtr'] == updated_row['time_left_qtr'] and
                original_row['quarter'] == updated_row['quarter']):
                original_row['Label'] = updated_row['Label']
    write_to_csv(output_csv_file_path, game_rows)
    if i > 5:
       break
"""

#print(f"Tagged CSV data has been written to {output_csv_file_path}.")
