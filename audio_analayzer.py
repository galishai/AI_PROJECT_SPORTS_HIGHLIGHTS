# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
from tkinter import filedialog
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from pydub import AudioSegment
import matplotlib.pyplot as plt
from scipy.io import wavfile


def convert_mp3_to_wav(mp3_path, wav_path):
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format="wav")
    print(f"File converted and saved to {wav_path}")

def intensity_sound_analyze(file_path):
    sample_rate, data = wavfile.read(file_path)
    if len(data.shape) > 1:
        data = data[:, 0]
    intensity = np.abs(data)
    threshold = np.mean(intensity) + 2 * np.std(intensity)
    spikes = np.where(intensity > threshold)[0]

    time = np.arange(len(data)) / sample_rate
    temp_time = np.array(time[spikes])
    temp_intensity = np.array(intensity[spikes])
    plt.figure(figsize=(10, 6))
    plt.plot(temp_time, temp_intensity, label="Intensity")
    plt.scatter(temp_time, temp_intensity, color='red', label="Spikes")
    plt.xlabel("Time (s)")
    plt.ylabel("Intensity")
    plt.title("Sound Intensity and Volume Spikes")
    plt.legend()
    plt.show()

def get_mp3_file_names(dest_folder="."):
    file_names = []
    for file_name in os.listdir(dest_folder):
        if os.path.isfile(os.path.join(dest_folder, file_name)) and file_name.endswith('.mp3'):
            file_names.append(file_name)
    return file_names

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("Choose video folder:")
    mp3_folder = target_folder = filedialog.askdirectory()
    print("Choose destination folder:")
    wav_folder = target_folder = filedialog.askdirectory()
    file_names = get_mp3_file_names(mp3_folder)
    for file in file_names:
        if file.endswith('.mp3'):
            convert_mp3_to_wav(mp3_folder + '/' + file, wav_folder + '/' + file.split('.mp3')[0] + '.wav')
            intensity_sound_analyze(wav_folder + '/' + file.split('.mp3')[0] + '.wav')


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
