import csv
import re
from tqdm import tqdm

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

def check_time_input(time):
    times = time.split(":")
    if not times[0].isdigit() or not times[1].isdigit() or int(times[0]) >= 12 or int(times[1]) >= 59:
        return False
    return True

def check_quarter_input(quarter):
    qurters = [1, 2, 3, 4]
    if not quarter.isdigit() or int(quarter) not in qurters:
        return False
    return True

def read_text(file_path):
    with open(file_path, mode='r', encoding='utf-8') as file:
        data = file.read().splitlines()
    cleaned_data = []
    for line in data:
        line = line.replace("(", "").replace(")", "").replace("\\n", "").replace("'", "")
        parts = line.split(",")
        time = parts[0].strip()
        quarter = parts[1].strip()
        if not check_time_input(time):
            continue
        if not check_quarter_input(quarter):
            continue
        if re.match(r'^\d+:\d+$', time) and re.match(r'^\d+$', quarter):
            cleaned_data.append([time, quarter])
    return cleaned_data

def add_column(csv_data, column_name, default_value):
    for row in csv_data:
        row[column_name] = default_value
    return csv_data

def tag_moves_for_game(iterator, text_data):
    previous_row = None
    count = 0
    update_rows = []
    try:
        first_row = iterator.peek()
    except StopIteration:
        return

    if (first_row['time_left_qtr'], first_row['quarter'][0]) in text_data:
        first_row['Label'] = 1
        count += 1
    else:
        first_row['Label'] = 0
    update_rows.append(first_row)

    for row in iterator:
        if previous_row is None or (previous_row['home_team'] == row['home_team'] and previous_row['away_team'] == row['away_team'] and previous_row['date'] == row['date']):
            if [row['time_left_qtr'], row['quarter'][0]] in text_data:
                row['Label'] = 1
                count += 1
            else:
                row['Label'] = 0
        else:
            break
        update_rows.append(row)

        previous_row = row
    return update_rows, count


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

csv_file_path = 'C:/Users/Sahar/Desktop/Computer Science/236502 Artificial inetlligence project/git_files/Labling from text file/output_full_season_v3.csv'
text_file_path = 'C:/Users/Sahar/Desktop/Computer Science/236502 Artificial inetlligence project/git_files/full season data/ocr_text_files/frames_Game_Recap_1_Nuggets_vs_Lakers.mp4.txt'
output_csv_file_path = csv_file_path

csv_data = read_csv(csv_file_path)
text_data = read_text(text_file_path)
iterator = GamesIterator(csv_file_path)
add_column(csv_data, 'Label', 0)
games_counter = []
#write_to_csv(csv_file_path, csv_data)
for i, game in enumerate(tqdm(iterator, desc="Processing games")):
    game_rows, count = tag_moves_for_game(iterator, text_data)
    if game_rows:
        games_counter.append((game_rows[0]['home_team'] + ' vs ' + game_rows[0]['away_team'] + ' ' + game_rows[0]['date'],count))
        #print(f"finished labeling: {game_rows[0]['home_team']} vs {game_rows[0]['away_team']} {game_rows[0]['date']}")
threshold = 10
count_below_threshold = sum(1 for game in games_counter if game[1] < threshold)
print(f"Number of games below threshold:{count_below_threshold}")

write_list_to_file('C:/Users/Sahar/Desktop/Computer Science/236502 Artificial inetlligence project/git_files/Labling from text file/label_counter.txt', games_counter)
"""    
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
