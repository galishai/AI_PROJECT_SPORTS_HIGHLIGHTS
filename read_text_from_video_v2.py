# This is a sample Python script.
import shutil
from os import times_result
from tkinter import filedialog

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import cv2
import os
import pytesseract
import json
import easyocr
from itertools import product
import numpy as np

HAVE_FRAMES = 1

DELETE_FRAMES_ON_DONE = 0

count = 0

time_dict = {}


def convert_to_binary_frame(frame, threshold=128):
    frame = cv2.resize(frame, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
    frame = 255 - frame
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((1, 1), np.uint8)
    gray_frame = cv2.dilate(gray_frame, kernel, iterations=1)
    gray_frame = cv2.erode(gray_frame, kernel, iterations=1)
    binary_frame = cv2.threshold(cv2.medianBlur(gray_frame, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return binary_frame


def video_to_frame(video_path, dest_path):
    frames_saved_count = 0
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        exit()

    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    frame_per_second = int(video.get(cv2.CAP_PROP_FPS))
    sanity = 0
    counter = 0
    while video.isOpened():
        is_return, frame = video.read()
        if not is_return:
            break
        if counter % frame_per_second == 0:
            if sanity % 100 == 0:
                print("frames saved: " + str(sanity))
            sanity += 1
            frame_filename = os.path.join(dest_path, f'frame_{counter // frame_per_second:04d}.jpg')
            binary_frame = convert_to_binary_frame(frame, dest_path)
            cv2.imwrite(frame_filename, binary_frame)
            frames_saved_count += 1
        counter += 1

    video.release()
    return frames_saved_count
    # print("finish extract frames, number of frames: {counter // frame_per_second}")


def crop_text_from_image(image_path, dest_path):  # Not work as expected, im try yo use OCR directly
    binary_frame = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    contours, _ = cv2.findContours(binary_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    cropped_frame = binary_frame[y:y + h, x:x + w]
    frame_filename = os.path.join(dest_path, f'cropped_frame.jpg')
    cv2.imwrite(frame_filename, cropped_frame)


def extract_text_from_images(time_frame, qtr_frame):
    global count
    text_time = pytesseract.image_to_string(time_frame, config=' --psm 7')
    text_qtr = pytesseract.image_to_string(qtr_frame, config=' --psm 6')
    if count % 100 == 0:
        print("times used extract: " + str(count))
    count += 1
    return text_time, text_qtr


def cropped_image_by_selection_area_first(image_path):
    binary_frame = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    roi = cv2.selectROI(binary_frame)
    cropped_frame = binary_frame[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
    return roi, cropped_frame


def crop_both(image_path, roi_time, roi_qtr):
    binary_frame = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    cropped_frame_time = binary_frame[int(roi_time[1]):int(roi_time[1] + roi_time[3]),
                         int(roi_time[0]):int(roi_time[0] + roi_time[2])]
    cropped_frame_qtr = binary_frame[int(roi_qtr[1]):int(roi_qtr[1] + roi_qtr[3]),
                        int(roi_qtr[0]):int(roi_qtr[0] + roi_qtr[2])]
    return cropped_frame_time, cropped_frame_qtr


def get_all_file_names_in_directory(dest_folder="."):
    file_names = []
    for file_name in os.listdir(dest_folder):
        if os.path.isfile(os.path.join(dest_folder, file_name)) and (file_name.endswith('.jpg')):
            file_names.append(file_name)
    return file_names


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("Choose video folder:")
    video_path = filedialog.askopenfilename() #"/Users/galishai/Dropbox/AI Project Gal/Playoffs/Finals/17.6.24 Finals G5 Celtics Mavericks.mkv"  # filedialog.askopenfilename()
    dest_path = filedialog.askopenfilename() + '/frames' #"/Users/galishai/Desktop/AI Project/AI_Project/AI_PROJECT_SPORTS_HIGHLIGHTS/frames"  # os.path.dirname(video_path) + '/frames'
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    if not HAVE_FRAMES:
        num_frames_saved = video_to_frame(video_path, dest_path)
    roi_time = [None] * 4
    roi_quarter = [None] * 4

    frames_cropped_dict = {}  # values - (time, quarter)

    roi_sample_path = filedialog.askopenfilename() #"/Users/galishai/Desktop/AI Project/AI_Project/AI_PROJECT_SPORTS_HIGHLIGHTS/frames/frame_0667.jpg"  # filedialog.askopenfilename()
    frame_names = get_all_file_names_in_directory(dest_path)
    roi_time, first_time = cropped_image_by_selection_area_first(roi_sample_path)
    roi_qtr, first_qtr = cropped_image_by_selection_area_first(roi_sample_path)
    sample_frame_num = roi_sample_path.removesuffix('.jpg')[-4:]
    frames_cropped_dict[sample_frame_num] = (first_time, first_qtr)
    cv2.destroyAllWindows()
    for i, file in enumerate(frame_names):
        if i % 100 == 0:
            print("files cropped: " + str(i))
        file_time, file_qtr = crop_both(dest_path + '/' + file, roi_time, roi_qtr)
        file_num = file.removesuffix('.jpg')[-4:]
        frames_cropped_dict[file_num] = (file_time, file_qtr)
    cv2.destroyAllWindows()

    #futures = []
    #reader = easyocr.Reader(['en'])
    flag = 0
    for four_digit_num in product('0123456789', repeat=4):

        num = ''.join(four_digit_num)
        #if num == '0617':
        #    flag = 1
        #if flag == 0:
        #    continue
        #if num == '0887':
            #break
        if num not in frames_cropped_dict:
            break
        time_frame, qtr_frame = frames_cropped_dict[num]
        text_time, text_qtr = extract_text_from_images(time_frame, qtr_frame)
        text_time = text_time.replace("\n", "")
        if (not any(char.isdigit() for char in text_time)) or text_time == '':  # or (frame_text_time, frame_text_quarter) in time_dict.values():
            continue
        qtr_digits_only = 'Q' + ''.join([char for char in text_qtr if char.isdigit()])
        time_dict[num] = (text_time, qtr_digits_only)

    text_file_path = dest_path.split('/frames')[0]
    with open(text_file_path + '/plot_test.txt', 'w', encoding='utf-8') as file:
        file.write(video_path.split('/')[-1] + '/')
        json.dump(time_dict, file, ensure_ascii=False, indent=4)

    if DELETE_FRAMES_ON_DONE:
        if os.path.exists(dest_path):
            shutil.rmtree(dest_path)
