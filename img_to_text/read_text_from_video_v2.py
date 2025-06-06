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
#import easyocr
from itertools import product
import numpy as np
from PIL import Image
#from torch.utils.tensorboard.summary import video

HAVE_FRAMES = 0

DELETE_FRAMES_ON_DONE = 0

STARTING_GAME_NUM = 1

count = 0

def invert_bw_images_fast(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    count_inv = 0

    # Process each file in the input folder
    for filename in os.listdir(input_folder):
        if count_inv % 100 == 0:
            print("files inverted: " + str(count_inv))
        count_inv += 1
        if filename.lower().endswith('.jpg'):  # Process only .jpg files
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Open the image and convert to grayscale (if needed)
            with Image.open(input_path) as img:
                img = img.convert("L")  # Ensure grayscale

                # Convert the image to a NumPy array
                img_array = np.array(img)

                # Invert the image (255 - pixel values)
                inverted_array = 255 - img_array

                # Convert the inverted array back to an image
                inverted_img = Image.fromarray(inverted_array)

                # Save the inverted image
                inverted_img.save(output_path)

            #print(f"Inverted: {filename} -> {output_path}")


def convert_to_binary_frame(frame, threshold=128):
    frame = cv2.resize(frame, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
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
    text_time = pytesseract.image_to_string(time_frame, config=' --psm 6')
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
    dest_dir = '/Users/galishai/Desktop/AI Project/AI_Project/AI_PROJECT_SPORTS_HIGHLIGHTS/img_to_text_plotfiles' #input("enter destination path (must be empty folder)\n") #'/Users/galishai/Desktop/AI Project/AI_Project/AI_PROJECT_SPORTS_HIGHLIGHTS/test_img_to_txt' #input("enter destination path\n") # filedialog.askopenfilename()
    video_paths_txt = '/Users/galishai/Desktop/AI Project/AI_Project/AI_PROJECT_SPORTS_HIGHLIGHTS/img_to_text/video_paths.txt' #input("enter video paths text file\n") #'/Users/galishai/Desktop/AI Project/AI_Project/AI_PROJECT_SPORTS_HIGHLIGHTS/video_paths.txt' #input("enter path of txt file with video paths\n") # filedialog.askopenfilename()
    with open(video_paths_txt, 'r') as file: #in video_folder_dirs every line is of the form: video_dir
        video_paths = []
        for line in file:
            video_paths.append(line.strip())
    video_names = []
    frames_dest_paths = []
    if not os.path.exists(dest_dir + '/frames'):
        os.makedirs(dest_dir + '/frames')
    game_num = STARTING_GAME_NUM
    for video_path in video_paths:
        video_name = video_path.split('/')[-1]
        video_names.append(video_name)
        frames_dest_path = dest_dir + '/frames/game_' + str(game_num) + '_' + video_name + '/frames'
        frames_dest_paths.append(frames_dest_path)
        if not os.path.exists(frames_dest_path):
            os.makedirs(frames_dest_path)
        game_num += 1
    game_num = STARTING_GAME_NUM
    if not HAVE_FRAMES:
        for video_name, video_path, frames_dest_path in zip(video_names, video_paths, frames_dest_paths):
            video_to_frame(video_path, frames_dest_path)
            num_frames = len(os.listdir(frames_dest_path))
            print("finished extracting frames from " + video_name + '. number of frames: ' + str(num_frames))
            game_num += 1
    roi_sample_paths_txt = input("enter samples paths txt file\n") #'/Users/galishai/Desktop/AI Project/AI_Project/AI_PROJECT_SPORTS_HIGHLIGHTS/sample_paths.txt' #input("enter samples paths txt file\n")
    with open(roi_sample_paths_txt, 'r') as file1:
        sample_paths = []
        for line, frames_dest_path in zip(file1,frames_dest_paths):
            cleaned_line = line.strip()
            if cleaned_line[-1] == '1':
                invert_bw_images_fast(frames_dest_path, frames_dest_path)
                sample_paths.append(cleaned_line[:-2])
            else:
                sample_paths.append(cleaned_line)
    assert len(video_paths) == len(sample_paths), "videos num: " + str(len(video_paths)) + " samples num: " + str(len(sample_paths))
    # for all videos
    roi_times = []
    roi_quarters = []
    first_times = []
    first_quarters = []
    #CHOOSING CROP DIMENSIONS
    for sample_path in sample_paths:
        roi_time, first_time = cropped_image_by_selection_area_first(sample_path.strip())
        roi_qtr, first_qtr = cropped_image_by_selection_area_first(sample_path.strip())
        cv2.destroyAllWindows()
        roi_times.append(roi_time)
        roi_quarters.append(roi_qtr)
        first_times.append(first_time)
        first_quarters.append(first_qtr)
    #CROPPING + IMG TO TEXT
    game_num = STARTING_GAME_NUM
    for video_name, sample_path, frames_dest_path, roi_time, roi_qtr, first_time, first_qtr \
            in zip (video_names, sample_paths, frames_dest_paths, roi_times, roi_quarters, first_times, first_quarters):

        frames_cropped_dict = {}  # values - (time, quarter)
        time_dict = {}

        sample_frame_num = sample_path.removesuffix('.jpg')[-4:]
        frames_cropped_dict[sample_frame_num] = (first_time, first_qtr)

        frame_names = get_all_file_names_in_directory(frames_dest_path)
        for i, img_name in enumerate(frame_names):
            if i % 100 == 0:
                print("files cropped: " + str(i))
            file_time, file_qtr = crop_both(frames_dest_path + '/' + img_name, roi_time, roi_qtr)
            file_num = img_name.removesuffix('.jpg')[-4:]
            frames_cropped_dict[file_num] = (file_time, file_qtr)
        print("finished cropping frames from " + video_name)
        # futures = []
        # reader = easyocr.Reader(['en'])
        flag = 0
        for four_digit_num in product('0123456789', repeat=4):

            num = ''.join(four_digit_num)
            # if num == '0617':
            #    flag = 1
            # if flag == 0:
            #    continue
            # if num == '0887':
            # break
            if num not in frames_cropped_dict:
                break
            time_frame, qtr_frame = frames_cropped_dict[num]
            text_time, text_qtr = extract_text_from_images(time_frame, qtr_frame)
            text_time = text_time.replace("\n", "")
            if (not any(char.isdigit() for char in
                        text_time)) or text_time == '':  # or (frame_text_time, frame_text_quarter) in time_dict.values():
                continue
            qtr_digits_only = 'Q' + ''.join([char for char in text_qtr if char.isdigit()])
            time_dict[num] = (text_time, qtr_digits_only)
        print("finished extracting text from " + video_name)



        if not os.path.exists(dest_dir + '/plots'):
            os.makedirs(dest_dir + '/plots')
        text_file_path = dest_dir + '/plots/' + 'game_' + str(game_num) + '_' + video_name.replace('.','') + '.txt'
        with open(text_file_path, 'w', encoding='utf-8') as file:
            file.write(video_name + '/')
            json.dump(time_dict, file, ensure_ascii=False, indent=4)

        print("done with " + video_name)
        game_num += 1
        count = 0

    if DELETE_FRAMES_ON_DONE:

        if os.path.exists(dest_dir+'/frames'):
            shutil.rmtree(dest_dir+'/frames')