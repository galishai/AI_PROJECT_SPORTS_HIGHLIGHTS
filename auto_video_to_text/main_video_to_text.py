import shutil
from tkinter import filedialog
import tkinter as tk
import cv2
import os
import pytesseract
import numpy as np
from PIL import Image
import re
import csv

SHOW_QUARTER = False
SHOW_TIME = False
HAVE_FRAMES = False
READ_TEXT = True
pattern = r'^\d{1,2}:\d{2}$'
scaleFactor = 1.7
inv_list = {'TNT':(1, 1),'CHA': (0, 0),'IND': (0, 0),'ESPN': (0, 1),'ORL': (0, 0),'BKN': (0, 0),'MIA': (0, 0),'TOR': (0, 0),'CHI': (0, 0),'MEM': (0, 0),'UTA': (1, 1),'LAC': (0, 0),'ATL': (0, 0),
            'CLE': (0, 0),'SAS': (0, 0),'DAL': (0, 0),'POR': (1, 1),'DET': (0, 0),'WAS': (1, 1),'NBA TV': (0, 0),'MIN': (0, 0),'OKC': (0, 0),'HOU': (1, 1),'MIL': (0, 0),'PHI': (0, 0),'SAC': (0, 0),
            'DEN': (1, 1),'BOS': (0, 0),'NYK': (1, 1),'GSW': (0, 0),'NOP': (0, 0),'PHX': (0, 0),'LAL': (0, 0),'ABC/ESPN': (1, 1),'ABC': (1, 1)}


def read_list_from_file(file_path):
    items = []
    with open(file_path, 'r', newline='') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            items.append([item.strip() for item in row])
    return items

def save_list_to_file(list, filename):
    with open(filename, 'w') as file:
        for item in list:
            file.write(f"{item}\n")

def load_dict(filename):
    dictionary = {}
    with open(filename, 'r') as file:
        for line in file:
            key, value = line.strip().split(': ', 1)
            dictionary[key] = value
    return dictionary

def unique_first_elements_dict(list_of_lists):
    first_elements = {sublist: None for sublist in list_of_lists}
    return first_elements

def unique_first_elements_list(list_of_lists):
    seen = set()
    unique_elements = []
    for sublist in list_of_lists:
        element = sublist[0]
        if element not in seen:
            seen.add(element)
            unique_elements.append(element)
    return unique_elements

def manual_crop_with_roi(image_path,scale_factor=0.5):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image from path: {image_path}")

    resized_image = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    roi = cv2.selectROI("Select ROI", resized_image, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    x, y, w, h = roi
    original_roi = (int(x / scale_factor), int(y / scale_factor), int(w / scale_factor), int(h / scale_factor))

    return original_roi

def crop_image(image, crop_box):
    x1, y1, x2, y2 = crop_box
    cropped_img = image[y1:y2, x1:x2]
    return cropped_img

def select_frame(frames_path):
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(initialdir=frames_path, title="Select Frame", filetypes=(("JPEG files", "*.jpg"), ("all files", "*.*")))
    root.destroy()
    return file_path
"""
def read_text_from_box(image, temp1, temp2):
    cropped_img_np = np.array(image)
    gray = cv2.cvtColor(cropped_img_np, cv2.COLOR_BGR2GRAY)
    sobel_x = np.array([[1, -1, 1],
                        [-1, 1, -1],
                        [1, -1, 1]], dtype=np.float32)
    gray = cv2.filter2D(gray, -1, sobel_x)
    kernel = np.ones((1, 1), np.uint8)
    gray = cv2.dilate(gray, kernel, iterations=1)
    gray = cv2.erode(gray, kernel, iterations=1)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    blurred = cv2.GaussianBlur(thresh, (5, 5), 0)
    img_text = pytesseract.image_to_string(blurred, lang='eng',
                                           config=' -c tessedit_char_whitelist=0123456789: --psm 7 --user-patterns C:/Users/Sahar/Desktop/Computer Science/236502 Artificial inetlligence project/TEMP/time_patterns.txt --tessdata-dir C:/Users/Sahar/Desktop/Computer Science/236502 Artificial inetlligence project/TEMP/tessdata_best-main --oem 1')

    return img_text
"""

def read_text_from_box(image_path,crop_box, is_time=True, inv=False):
    image = cv2.imread(image_path)
    img_np = np.array(image)
    inputImage = cv2.resize(img_np, None, fx=scaleFactor, fy=scaleFactor, interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((1, 1), np.uint8)
    gray = cv2.dilate(gray, kernel, iterations=1)
    gray = cv2.erode(gray, kernel, iterations=1)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    blurred = cv2.GaussianBlur(thresh, (5, 5), 0)
    if inv:
        blurred = 255 - blurred

    blurred = cv2.resize(blurred, None, fx=1/scaleFactor, fy=1/scaleFactor, interpolation=cv2.INTER_LINEAR)
    cropped_img_time = crop_image(blurred, crop_box[0])
    cropped_img_quarter = crop_image(blurred, crop_box[1])

    if SHOW_TIME:
        if not isinstance(cropped_img_time, np.ndarray):  # section for debug the cropping quarter image
            temp = np.array(cropped_img_time)
            cropped_img_time = temp
        cv2.imshow('Cropped Image Time', cropped_img_time)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if SHOW_QUARTER:
        if not isinstance(cropped_img_quarter, np.ndarray):  # section for debug the cropping quarter image
            temp = np.array(cropped_img_quarter)
            cropped_img_quarter = temp
        cv2.imshow('Cropped Image Quarter', cropped_img_quarter)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if not is_time:
        #plt.imshow(blurred)
        img_text = pytesseract.image_to_string(cropped_img_quarter, lang='eng', config=' -c tessedit_char_whitelist=1234 --psm 11 --user-patterns C:/Users/Sahar/Desktop/Computer Science/236502 Artificial inetlligence project/TEMP/qtr_patterns.txt --tessdata-dir C:/Users/Sahar/Desktop/Computer Science/236502 Artificial inetlligence project/TEMP/tessdata_best-main --oem 1')
    else:
        img_text = pytesseract.image_to_string(cropped_img_time, lang='eng', config=' -c tessedit_char_whitelist=0123456789:. --psm 6 --user-patterns C:/Users/Sahar/Desktop/Computer Science/236502 Artificial inetlligence project/TEMP/time_patterns.txt --tessdata-dir C:/Users/Sahar/Desktop/Computer Science/236502 Artificial inetlligence project/TEMP/tessdata_best-main --oem 1')
    return img_text


def expand_image_right(image, pixels=50):
    if not isinstance(image, np.ndarray):  # section for debug the cropping image
        image = np.array(image)
    height, width, channels = image.shape
    expanded_image = np.zeros((height, width + pixels, channels), dtype=image.dtype)
    expanded_image[:, :width] = image
    return expanded_image


def process_frames(target_dir, recap_list, broadcast_dict):
    print('extracting text file')
    for i, game_dir in enumerate(os.listdir(target_dir)):
        frames_path = game_path = os.path.join(target_dir, game_dir)
        #frames_path = os.path.join(game_path, 'frames')
        if not os.path.isdir(frames_path):
            continue
        broadcast_type = recap_list[i][0]  # Get the broadcast type for this directory
        time_list = []
        for file in os.listdir(frames_path):
            if file.endswith('.jpg'):
                file_path = os.path.join(frames_path, file)
                if broadcast_dict[broadcast_type] is None:
                    temp_file_to_crop = select_frame(frames_path)
                    img_np = np.array(temp_file_to_crop)
                    inputImage = cv2.resize(img_np, None, fx=scaleFactor, fy=scaleFactor, interpolation=cv2.INTER_LINEAR)
                    x, y, w, h = manual_crop_with_roi(inputImage, 0.2)
                    x2, y2, w2, h2 = manual_crop_with_roi(inputImage, 0.2)
                    broadcast_dict[broadcast_type] = [(x, y, x + w, y + h), (x2, y2, x2 + w2, y2 + h2)]
                crop_box = broadcast_dict[broadcast_type]
                #cropped_img_time = crop_image(file_path, crop_box[0])
                #cropped_img_quarter = crop_image(file_path, crop_box[1])
                if not READ_TEXT:
                    continue
                time = read_text_from_box(file_path, crop_box, True, inv_list[broadcast_type][0])

                quarter = read_text_from_box(file_path, crop_box, False, inv_list[broadcast_type][1])
                if re.match(pattern, time) and time != '':
                    time_list.append((time, quarter))
        if not READ_TEXT:
            continue
        text_folder_path = os.path.join(os.path.dirname(target_dir), 'output')
        os.makedirs(text_folder_path, exist_ok=True)
        text_file_path = os.path.join(text_folder_path, f"{game_dir}.txt")
        text_file_path = text_file_path
        save_list_to_file(time_list, text_file_path)

def save_dict_to_file(dictionary, filename):
    with open(filename, 'w') as file:
        for key, value in dictionary.items():
            file.write(f"{key}: {value}\n")

def capture_broadcast_coor(target_dir, broadcast_dict, broadcast_list):

    for i, file in enumerate(os.listdir(target_dir)):
        temp_file = os.path.join(target_dir, file)
        x, y, w, h = manual_crop_with_roi(temp_file, 0.7)
        x2, y2, w2, h2 = manual_crop_with_roi(temp_file, 0.7)

        broadcast_dict[broadcast_list[i]] = ((x, y, x + w, y + h), (x2, y2, x2 + w2, y2 + h2))
    print('stop')
    file_name = os.path.join(target_dir, f'broadcast_coor.txt')
    save_dict_to_file(broadcast_dict, file_name)

def parse_tuple_string(s):
    s = s.strip("()")
    elements = s.split("), (")
    tuple_elements = []
    for element in elements:
        tuple_elements.append(tuple(map(int, element.split(','))))
    return tuple(tuple_elements)

def apply_parse_to_dict(dictionary):
    parsed_dict = {}
    for key, value in dictionary.items():
        parsed_dict[key] = parse_tuple_string(value)
    return parsed_dict

def transform_dict_values(dict):
    transformed_dict = {}                       #This is a help function you don't need to use it or the change the path below
    for key, value in dict.items():
        (x, y, w, z), (s, t, e, r) = value
        transformed_value = ((x, y, x + w, y + z), (s, t, s + e, t + r))
        transformed_dict[key] = transformed_value
    save_dict_to_file(transformed_dict, 'C:/Users/Sahar/Desktop/Computer Science/236502 Artificial inetlligence project/NBA Recap/broadcast_coor_dict_2.txt')


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
            #binary_frame = resize_frame(frame, dest_path)
            cv2.imwrite(frame_filename,frame)     #binary_frame)
            frames_saved_count += 1
        counter += 1

    video.release()
    print("finished extracting frames from " + video_path.split('/')[-1] + '. number of frames: ' + str(sanity))
    return frames_saved_count

def resize_frame(frame, threshold=128):
    frame = cv2.resize(frame, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    return frame

def ocr_for_video_frames(csv_path, broadcast_dict_path, target_folder):
    recap_list = read_list_from_file(csv_path)
    broadcast_dict = load_dict(broadcast_dict_path)
    broadcast_dict = apply_parse_to_dict(broadcast_dict)

    #transform_dict_values(broadcast_dict)
    #broadcast_list = unique_first_elements_list(recap_list)         #only for capture
    recap_dir = target_folder
    process_frames(recap_dir, recap_list, broadcast_dict)
    #broadcast_frames_samples_dir = 'C:/Users/Sahar/Desktop/Computer Science/236502 Artificial inetlligence project/NBA Recap/All broadcast recaps/broadcast-sampels'        #only for capture
    #capture_broadcast_coor(broadcast_frames_samples_dir, broadcast_dict, broadcast_list)        #only for capture


def process_videos_in_batches(target_folder,  csv_path, broadcast_dict_path, batch_size=50):
    video_files = [f for f in os.listdir(target_folder) if f.endswith('.mp4')]

    for i in range(0, len(video_files), batch_size):
        frame_dir = os.path.join(target_folder, 'frames')
        os.makedirs(frame_dir, exist_ok=True)
        batch = video_files[i:i + batch_size]
        for video_file in batch:
            video_path = os.path.join(target_folder, 'frames')
            temp_video_path = os.path.join(video_path, video_file)
            #os.makedirs(temp_video_path, exist_ok=True)
            frames_folder = os.path.join(video_path, f"frames_{video_file}")
            video_path = os.path.join(target_folder,video_file)
            if not HAVE_FRAMES:
                os.makedirs(frames_folder)
                video_to_frame(video_path, frames_folder)
            ocr_for_video_frames(csv_path, broadcast_dict_path, frame_dir)

        shutil.rmtree(frame_dir)

if __name__ == '__main__':
    target_folder = 'C:/Users/Sahar/Desktop/Computer Science/236502 Artificial inetlligence project/NBA Recap/All broadcast recaps'                                 # Enter destination path - folder that contain only the recap videos
    csv_path = 'C:/Users/Sahar/Desktop/Computer Science/236502 Artificial inetlligence project/NBA Recap/‏‏Broadcast-csv.csv'                                   # Enter the path of broadcast_csv in your computer
    broadcast_dict_path = 'C:/Users/Sahar/Desktop/Computer Science/236502 Artificial inetlligence project/NBA Recap/broadcast_coor_with_corrections.txt'            # Enter the broadcast_coor.txt
    process_videos_in_batches(target_folder, csv_path, broadcast_dict_path, batch_size=2)