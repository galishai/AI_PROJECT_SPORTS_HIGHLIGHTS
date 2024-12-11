# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import cv2
import os
import pytesseract
from tkinter import filedialog
import json
import shutil

def convert_to_binary_frame(frame, dest_path, threshold =128):
    #frame = cv2.imread(frame_path)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary_frame = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)
    #frame_filename = os.path.join(dest_path, f'binary_frame.png')
    #cv2.imwrite(frame_filename, binary_frame)
    return binary_frame

def video_to_frame(video_path, dest_path):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        exit()

    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    frame_per_second = int(video.get(cv2.CAP_PROP_FPS))
    counter = 0
    while video.isOpened():
        is_return, frame = video.read()
        if not is_return:
            break
        if counter % frame_per_second == 0:
            frame_filename = os.path.join(dest_path, f'frame_{counter // frame_per_second:04d}.png')
            binary_frame = convert_to_binary_frame(frame, dest_path)
            cv2.imwrite(frame_filename, binary_frame)
        counter += 1

    video.release()
    #print("finish extract frames, number of frames: {counter // frame_per_second}")

def crop_text_from_image(image_path, dest_path):    #Not work as expected, im try yo use OCR directly
    binary_frame = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    contours, _ = cv2.findContours(binary_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    cropped_frame = binary_frame[y:y+h, x:x+w]
    frame_filename = os.path.join(dest_path, f'cropped_frame.png')
    cv2.imwrite(frame_filename, cropped_frame)

def extract_text_from_image(image_path):
    binary_frame = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    text = pytesseract.image_to_string(binary_frame, config='--psm 6')
    return text

def cropped_image_by_selection_area(image_path, roi, isFirst, counter, dest_path):
    binary_frame = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if isFirst:
        roi = cv2.selectROI(binary_frame)
    cropped_frame = binary_frame[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
    cv2.imshow("Cropped binary frame", cropped_frame)
    #cv2.waitKey(0)
    if not os.path.exists(dest_path + '/Cropped frames'):
        os.makedirs(dest_path + '/Cropped frames')
    cv2.imwrite(os.path.dirname(dest_path + '/Cropped frames') + '/Cropped frames/frame ' + str(counter) + '.png', cropped_frame)
    cv2.destroyAllWindows()
    return roi

def get_all_file_names_in_directory(dest_folder="."):
    file_names = []
    for file_name in os.listdir(dest_folder):
        if os.path.isfile(os.path.join(dest_folder, file_name)) and (file_name.endswith('.png')):
            file_names.append(file_name)
    return file_names


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("Choose video folder:")
    video_path = filedialog.askopenfilename()
    dest_path = os.path.dirname(video_path) + '/frames'
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    if not os.path.exists(dest_path + '/Time'):
        os.makedirs(dest_path + '/Time')
    if not os.path.exists(dest_path + '/Quarter'):
        os.makedirs(dest_path + '/Quarter')
    video_to_frame(video_path, dest_path)
    roi_time = [None] * 4
    roi_quarter = [None] * 4
    frame_names = get_all_file_names_in_directory(dest_path)

    roi_sample_path = filedialog.askopenfilename()
    roi_time = cropped_image_by_selection_area(roi_sample_path, None, 1, 0, dest_path + '/Time')
    roi_quarter = cropped_image_by_selection_area(roi_sample_path, None, 1, 0, dest_path + '/Quarter')

    for i, file in enumerate(frame_names):
        cropped_image_by_selection_area(dest_path + '/' + file, roi_time, 0, i, dest_path + '/Time')
        cropped_image_by_selection_area(dest_path + '/' + file, roi_quarter, 0, i, dest_path + '/Quarter')

    cropped_frames_time = get_all_file_names_in_directory(dest_path + '/Time/Cropped frames')
    cropped_frames_quarter = get_all_file_names_in_directory(dest_path + '/Quarter/Cropped frames')
    time_dict = {}
    for file in cropped_frames_time:
        frame_num = file[-5]
        frame_text_time = extract_text_from_image(dest_path + '/Time/Cropped frames/' + file)
        frame_text_time = frame_text_time.replace("\n", "")
        frame_text_quarter = extract_text_from_image(dest_path + '/Quarter/Cropped frames/' + file)
        digits_only = ''.join([char for char in frame_text_quarter if char.isdigit()])
        frame_text_quarter = 'Q' + digits_only
        if any(char.isalpha() for char in frame_text_time): # or (frame_text_time, frame_text_quarter) in time_dict.values():
            continue

        time_dict[frame_num] = (frame_text_time, frame_text_quarter)

    text_file_path = dest_path.split('/frames')[0]
    with open(text_file_path + '/plot.txt', 'w', encoding='utf-8') as file:
        #file.write(video_path.split('/')[-1]+'/')
        json.dump(time_dict, file, ensure_ascii=False, indent=4)

    if os.path.exists(dest_path):
        shutil.rmtree(dest_path)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
