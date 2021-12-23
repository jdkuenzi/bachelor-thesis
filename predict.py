#
# Author    : Jean-Daniel Kuenzi
# Date      : 03.05.2021
# Update    : 15.08.2021
# Desc      : File that loads a model and a dataset and then displays the model's predictions on the data
# Version   : 1.0.0
#

import os
import random
import sys
from pathlib import Path
import neuralnetwork as mynn

from scipy.io.wavfile import read
from librosa.display import waveplot

import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

SAMPLING_RATE = 44100  # 44.1[KHz]
EXAMPLE_SIZE = 2048  # ~(0.093[s]) of a record represents an example
NOISE_WEIGHT = 10

def get_sample_from_main_data_directory(path):
    x = []
    res_path = '{}/res/'.format(path)
    if os.path.exists(path):
        folders = os.listdir(path)
        for label, name in enumerate(folders):
            dir_path = Path(path) / name
            audio_sample_paths = [
                os.path.join(dir_path, filepath)
                for filepath in os.listdir(dir_path)
                if filepath.endswith(".wav")
            ]
            x += audio_sample_paths
        print("Found {} files.".format(len(x)))
        if not os.path.exists(res_path):
            os.makedirs(res_path)
    return x

def load_data(data_x, win_length=1024, noise_weight=1, sampling_rate=44100):
    pro_x = []
    track_x = []

    hop_length = win_length // 4
    print(data_x)
    sample_name = data_x.split('\\')[-1]
    sr, output = read(data_x)
    output = output.astype(np.float32)

    noise_data = output[:win_length]
    noise_coeff = amp_sum(noise_data, win_length) * noise_weight


    is_attack = True
    i = 0 
    i_max = len(output) - win_length
    while i <= i_max :
        sp = output[i:i+win_length]
        if amp_sum(sp, win_length) > noise_coeff:
            if is_attack:
                is_attack = False
                i += (win_length - hop_length)
            else:
                pro_x.append(sp)
                track_x.append((i, i+win_length))
        else:
            is_attack = True
        i += hop_length

    pro_x = torch.tensor(pro_x, dtype=torch.float32)  # magnitude of the frequency

    return pro_x, track_x, output, sample_name

def amp_sum(data, len_data):
    return np.sum(np.abs(data))/len_data

def draw_line_with_note(start_x, stop_x, class_name, ax, label, y, color, plot_legend=False):
    if plot_legend:
        ax.plot([start_x, stop_x], [y, y],
                marker="|", color=color, label=label)
    else:
        ax.plot([start_x, stop_x], [y, y], marker="|", color=color)
    
    ax.annotate(class_name,
                xy=((start_x + stop_x) / 2, y),
                xytext=(0, 0),
                textcoords="offset points",
                ha='center', va='bottom', color=color, fontsize=10, fontweight='bold')

def plot_notes(track_x, class_y, class_names, ax, legend, row_y=0, color='black', sampling_rate=44100):
    old_track = track_x[0]
    old_y = class_y[0]
    label = None
    start = track_x[0][0] / sampling_rate
    stop = 0.0
    plot_legend = True

    for y, track in zip(class_y, track_x):
        if old_y != y or old_track[1] < track[0]:
            stop = old_track[1] / sampling_rate
            label = class_names[old_y]
            # print('{} from {:.2f} to {:.2f} [s]'.format(
                # class_name, start, stop))
            draw_line_with_note(start, stop, label, ax,
                                legend, row_y, color, plot_legend)
            plot_legend = False
            start = track[0] / sampling_rate
        old_y = y
        old_track = track
    label = class_names[old_y]
    stop = old_track[1] / sampling_rate
    # print('{} from {:.2f} to {:.2f} [s]'.format(class_name, start, stop))
    draw_line_with_note(start, stop, label, ax, legend, row_y, color, plot_legend)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('error command is : predict.py <model_path> <data_path>')
        exit(0)
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cpu')

    model_path = sys.argv[1]
    data_path = sys.argv[2]

    data = get_sample_from_main_data_directory(data_path)

    model = torch.load(model_path)
    model.to(device)

    class_names = model.class_names
    print(class_names)
    for path in data:
        x, track_x, temporal, sample_name = load_data(
            path, win_length=EXAMPLE_SIZE, noise_weight=NOISE_WEIGHT)

        x = x.to(device)
        prob_y = model(x).detach()
        class_y = prob_y.argmax(1)

        fig = plt.figure(figsize=(20,10))
        ax = fig.subplots()
        waveplot(temporal, sr=SAMPLING_RATE, ax=ax, color='r')
        ax.set_title('Signal {}'.format(path))
        ax.set_ylabel('amplitude')
        ax.set_xlabel('time[s]')
        ax.set_facecolor('darkgray')

        plot_notes(track_x, class_y.numpy(), class_names, ax, 'predicted')

        plt.legend()
        plt.tight_layout()
        plt.savefig('{}/res/{}.png'.format(data_path, sample_name))
        plt.close()
