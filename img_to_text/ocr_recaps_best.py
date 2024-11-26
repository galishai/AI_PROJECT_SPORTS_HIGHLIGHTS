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
with open(images_path_txt, 'r') as file1:
    images_path = []
    for line in file1:
        cleaned_line = line.strip()
        images_path.append(cleaned_line)

for img_path in images_path:
    inputImage = cv2.imread(img_path)
    scaleFactor = 1.2
    inputImage = cv2.resize(inputImage, None, fx=scaleFactor, fy=scaleFactor, interpolation=cv2.INTER_LINEAR)
    h,w, _ = inputImage.shape
    inputImage = inputImage[h-(int(h/3.5)):h, w//2:]
    # Deep Copy:
    #inputImageCopy = inputImage.copy()
    gray = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((1,1), np.uint8)
    gray = cv2.dilate(gray, kernel, iterations=1)
    gray = cv2.erode(gray, kernel, iterations=1)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    #open_ = opening(thresh)
    blurred = cv2.GaussianBlur(thresh, (5, 5), 0)
    show_boxes(thresh)
    img_text = pytesseract.image_to_string(blurred, lang='eng', config=' -c tessedit_char_whitelist=0123456789:. --psm 11 --user-patterns /Users/galishai/PycharmProjects/AI_PROJECT_SPORTS_HIGHLIGHTS/img_to_text/ocr_config/time_patterns.txt --tessdata-dir /Users/galishai/PycharmProjects/AI_PROJECT_SPORTS_HIGHLIGHTS/img_to_text/tessdata_best-main --oem 1')
    print(img_text)