# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
import time
import tkinter as tk
from tkinter import filedialog

from moviepy.editor import VideoFileClip
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By

def get_video_file_names(dest_folder="."):
    file_names = []
    for file_name in os.listdir(dest_folder):
        if os.path.isfile(os.path.join(dest_folder, file_name)) and (file_name.endswith('.mp4') or file_name.endswith('.mkv')):
            file_names.append(file_name)
    return file_names


def separate_audio(video_path, audio_path):
    video_clip = VideoFileClip(video_path)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(audio_path)
    print("separate audio succeed, audio path:")
    print(audio_path)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    file_names = []
    print("Choose video folder:")
    video_folder = target_folder = filedialog.askdirectory()
    print("Choose destination folder:")
    audio_folder = target_folder = filedialog.askdirectory()
    print(video_folder)
    file_names = get_video_file_names(video_folder)
    for file in file_names:
        if file.endswith('.mp4'):
            separate_audio(video_folder + '/' + file, audio_folder + '/' + file.split('.mp4')[0] + '.mp3')
        if file.endswith('.mkv'):
            separate_audio(video_folder + '/' + file, audio_folder + '/' + file.split('.mkv')[0] + '.mp3')


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
