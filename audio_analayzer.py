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
from scipy.signal import convolve

SAMPLES_PER_SECONDE = 10
MIN_TIMR_BETWEEN_SPIKE =5

def convert_mp3_to_wav(mp3_path, wav_path):
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format="wav")
    print(f"File converted and saved to {wav_path}")

def average_samples(data, sample_rate, factor=1):
    samples_per_average = int(44100 * factor)
    averaged_data = np.array([np.mean(data[i:i + samples_per_average]) for i in range(0, len(data), samples_per_average)])
    return averaged_data, factor

def intensity_sound_analyze(file_path):
    sample_rate, data = wavfile.read(file_path)
    if len(data.shape) > 1:
        data = data[:, 0]


    averaged_data, sample_rate = average_samples(data, sample_rate, 1 / SAMPLES_PER_SECONDE)
    intensity = np.abs(averaged_data)
    threshold = np.mean(intensity) + 4 * np.std(intensity)
    #min_threshold = np.mean(intensity)
    #spikes = np.where((intensity > threshold) & (intensity > min_threshold))[0]
    spikes = np.where(intensity > threshold)[0]
    print(np.size(spikes))

    filtered_spikes = []
    for i in spikes:
        if len(filtered_spikes) == 0 or i - filtered_spikes[-1] >= MIN_TIMR_BETWEEN_SPIKE * SAMPLES_PER_SECONDE:
            filtered_spikes.append(i)

    print(np.size(filtered_spikes))
    audio_name = file_path.split('/')[-1].split('.')[0]
    time = np.arange(len(averaged_data)) / SAMPLES_PER_SECONDE
    plt.figure(figsize=(10, 6))
    plt.plot(time, intensity, label="Intensity")
    plt.scatter(time[filtered_spikes], intensity[filtered_spikes], color='red', label="Spikes")
    plt.xlabel("Time (s)")
    plt.ylabel("Intensity")
    plt.title("Sound Intensity and Volume Spikes\n" + audio_name + "\n number of spikes: " + str(np.size(filtered_spikes)))
    plt.legend()
    plt.savefig(audio_name + '.png')
    plt.show()

    return filtered_spikes


def get_mp3_file_names(dest_folder="."):
    file_names = []
    for file_name in os.listdir(dest_folder):
        if os.path.isfile(os.path.join(dest_folder, file_name)) and file_name.endswith('.mp3'):
            file_names.append(file_name)
    return file_names


def filter_noise(audio_file, kernel):
    sample_rate, audio_data = wavfile.read(audio_file)
    if len(audio_data.shape) == 2:
        filtered_audio = np.zeros_like(audio_data)
        for channel in range(audio_data.shape[1]):
            filtered_audio[:, channel] = convolve(audio_data[:, channel], kernel, mode='same')
    else:
        filtered_audio = convolve(audio_data, kernel, mode='same')

    #filtered_audio = convolve(audio_data, kernel, mode='same')
    dest_path = os.path.join(os.path.dirname(audio_file), 'filtered_audio.wav')
    wavfile.write(dest_path, sample_rate, filtered_audio.astype(np.int16))
    return dest_path

def gaussian_kernel(size, sigma=3):
    kernel = np.fromfunction(lambda x: (1/(np.sqrt(2*np.pi)*sigma)) * np.exp(-((x-(size-1)/2)**2)/(2*sigma**2)),(size,))
    return kernel / np.sum(kernel)


def file_to_dict(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        content = content[1:-1].strip()
        my_dict = eval(f"{{{content}}}")
    return my_dict

def write_list_to_file(lst, file_name):
    with open(file_name, 'w') as file:
        for item in lst:
            file.write(f"{item}\n")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("Choose audio folder:")
    mp3_folder = target_folder = filedialog.askdirectory()
    wav_folder = target_folder
    file_names = get_mp3_file_names(mp3_folder)
    #kernel = np.array([1, -1, 1, -1, 1])
    kernel = np.array(gaussian_kernel(5))
    for file in file_names:
        if file.endswith('.mp3'):
            convert_mp3_to_wav(mp3_folder + '/' + file, wav_folder + '/' + file.split('.mp3')[0] + '.wav')
            filtered_audio_path = filter_noise(wav_folder + '/' + file.split('.mp3')[0] + '.wav', kernel)
            spikes = intensity_sound_analyze(filtered_audio_path)
            time_dict = {}
            print("Choose dictionary text file:")
            time_dict = file_to_dict(filedialog.askopenfilename())
            time_list = []
            for spike in spikes:
                game_time_spikes = time_dict[str(int(spike / SAMPLES_PER_SECONDE))]
                game_time_spikes_in_sec = int(float(game_time_spikes[0])) + (int(game_time_spikes[1][1])-1) * 60 * 12
                time_list.append(game_time_spikes_in_sec)
            write_list_to_file(time_list, target_folder + '/' + file.split('.mp3')[0] + '_spikes.txt')

            #intensity_sound_analyze(wav_folder + '/' + file.split('.mp3')[0] + '.wav')
    #file_path = filedialog.askopenfilename()
    #filtered_audio_path = filter_noise(file_path, kernel)
    #intensity_sound_analyze(file_path)



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
