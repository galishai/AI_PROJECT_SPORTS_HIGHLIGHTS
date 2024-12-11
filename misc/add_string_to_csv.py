# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import csv
from tkinter import filedialog
import os

def add_string_to_csv_rows(input_path, dest_path, string_to_add):
    with open(input_path, 'r', newline='', encoding='utf-8') as infile, \
            open(dest_path, 'w', newline='', encoding='utf-8') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        for i, row in enumerate(reader):
            if i == 0:
                row.append('is_highlight')
                writer.writerow(row)
            else:
                row.append(string_to_add)
                writer.writerow(row)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    input_path = filedialog.askopenfilename()
    add_string_to_csv_rows(input_path, os.path.dirname(input_path) + '/output_file.csv', '0')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
