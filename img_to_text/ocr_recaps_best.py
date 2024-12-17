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
import matplotlib.pyplot as plt
from pytesseract import Output
import re

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

def show_boxes(frame):
    #frame = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray_BGR = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    h, w, _ = gray_BGR.shape  # assumes color image
    boxes = pytesseract.image_to_boxes(gray_BGR)  # also include any config options you use

    for b in boxes.splitlines():
        b = b.split(' ')
        im = cv2.rectangle(gray_BGR, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)

    cv2.imshow('filename', gray_BGR)
    cv2.waitKey(0)

def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

images_path_txt = '/Users/galishai/PycharmProjects/AI_PROJECT_SPORTS_HIGHLIGHTS/img_to_text/ocr_test_images.txt'
'''
with open(images_path_txt, 'r') as file1:
    images_path = []
    for line in file1:
        cleaned_line = line.strip()
        images_path.append(cleaned_line)
'''

directory_path = '/Users/galishai/Desktop/img_to_text_plotfiles/frames'

# List the files in the directory
file_names = list_files_in_directory(directory_path)
count = 1

scaleFactor = 1.2
qtr_inv = True
time_inv = False
with open('test_game.txt', mode='w', newline='') as file:
    for i, img_path in enumerate(file_names):
        if i == 0:
            roi_frame = cv2.imread('/Users/galishai/Desktop/img_to_text_plotfiles/frames/frame_0063.jpg')
            inputImage = cv2.resize(roi_frame, None, fx=scaleFactor, fy=scaleFactor, interpolation=cv2.INTER_LINEAR)
            gray = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
            kernel = np.ones((1, 1), np.uint8)
            gray = cv2.dilate(gray, kernel, iterations=1)
            gray = cv2.erode(gray, kernel, iterations=1)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            # open_ = opening(thresh)
            blurred = cv2.GaussianBlur(thresh, (5, 5), 0)
            roi_qtr = cv2.selectROI(blurred)
            roi_time = cv2.selectROI(blurred)
        inputImage = cv2.imread(directory_path + '/' + img_path)
        inputImage = cv2.resize(inputImage, None, fx=scaleFactor, fy=scaleFactor, interpolation=cv2.INTER_LINEAR)
        #h,w, _ = inputImage.shape
        #inputImage = inputImage[h-(int(h/3.5)):h, w//2:]
        # Deep Copy:
        #inputImageCopy = inputImage.copy()
        gray = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((1,1), np.uint8)
        gray = cv2.dilate(gray, kernel, iterations=1)
        gray = cv2.erode(gray, kernel, iterations=1)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        #open_ = opening(thresh)
        blurred = cv2.GaussianBlur(thresh, (5, 5), 0)
        #show_boxes(thresh)
        cropped_qtr = inputImage[int(roi_qtr[1]):int(roi_qtr[1] + roi_qtr[3]), int(roi_qtr[0]):int(roi_qtr[0] + roi_qtr[2])]
        cropped_time = inputImage[int(roi_time[1]):int(roi_time[1] + roi_time[3]), int(roi_time[0]):int(roi_time[0] + roi_time[2])]
        if qtr_inv:
            cropped_qtr = 255 - cropped_qtr
        if time_inv:
            time_inv = 255 - cropped_time
        #img_text = pytesseract.image_to_string(cropped_qtr, lang='eng', config=' -c tessedit_char_whitelist=0123456789:. --psm 11 --user-patterns /Users/galishai/PycharmProjects/AI_PROJECT_SPORTS_HIGHLIGHTS/img_to_text/ocr_config/patterns.txt --tessdata-dir /Users/galishai/PycharmProjects/AI_PROJECT_SPORTS_HIGHLIGHTS/img_to_text/tessdata_best-main --oem 1')
        img_qtr_text = pytesseract.image_to_string(cropped_qtr, lang='eng',
                                               config=' -c tessedit_char_whitelist=1234 --psm 11 --user-patterns /Users/galishai/PycharmProjects/AI_PROJECT_SPORTS_HIGHLIGHTS/img_to_text/ocr_config/qtr_patterns.txt --tessdata-dir /Users/galishai/PycharmProjects/AI_PROJECT_SPORTS_HIGHLIGHTS/img_to_text/tessdata_best-main --oem 1')
        img_time_text = pytesseract.image_to_string(cropped_time, lang='eng',
                                               config=' -c tessedit_char_whitelist=0123456789:. --psm 7 --user-patterns /Users/galishai/PycharmProjects/AI_PROJECT_SPORTS_HIGHLIGHTS/img_to_text/ocr_config/time_patterns.txt --tessdata-dir /Users/galishai/PycharmProjects/AI_PROJECT_SPORTS_HIGHLIGHTS/img_to_text/tessdata_best-main --oem 1')
        if img_time_text == '' or img_qtr_text == '':
            continue
        data = "(" + str(cropped_time) + ", " + str(cropped_qtr) + ")\n"
        file.write(data)
        print("frame :" + str(i) + " qtr: " + img_qtr_text + " time: " + img_time_text + '\n')

