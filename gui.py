#
# Author    : Jean-Daniel Kuenzi
# Date      : 05.05.2021
# Update    : 15.08.2021
# Desc      : File containing the GUI class which will define the Tkinter interface
# Version   : 1.0.0
#

import pyaudio
import numpy as np
import tkinter as tk
import tkinter.ttk as ttk

import torch
import neuralnetwork as mynn

class gui:
    def __init__(self, model, chunk=1024, sr=44100, channels=1):
        self.model = model
        self.class_names = np.array(model.class_names)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.device = torch.device('cpu') # I dont have CUDA available, I force CPU utilisation here
        self.model.to(self.device)
        self.model.eval()

        self.chunk = chunk
        self.old_x = None
        self.sr = sr
        self.channels = channels

        self.noise_coeff = None
        self.noise_weight = 5
        self.stream = None

        self.root = tk.Tk()  # Makes main window
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.title("Projet Bachelor")

        # Graphics window
        self.create_button()
        self.create_predict_labels()
        self.create_channels_dropdown()

    def get_natural_noise(self):
        data = np.array(np.frombuffer(self.stream.read(self.chunk), dtype=np.float32), dtype=np.float32)
        self.noise_coeff = self.amp_sum(data, self.chunk) * self.noise_weight

    def predict_data(self, in_data, frame_count, time_info, status):
        xi = np.array(np.frombuffer(in_data, dtype=np.float32), dtype=np.float32)

        if self.amp_sum(xi, frame_count) > self.noise_coeff:
            if self.old_x is not None:
                # print(xi.shape)
                # print(self.old_x.shape)
                x = np.concatenate((self.old_x, xi))
                x = torch.from_numpy(x).view(1, -1).to(self.device)
                
                x = self.model(x).detach().squeeze()
                x, y = x.topk(4)
                y = self.class_names[y]

                self.current_predict.set(f"Current predict:\n{y[0]}\n{x[0]:.3f}")
                self.top_3_predicts.set(f"top 3 predicted chords and notes :\n{y[1]}\n{x[1]:.3f}\n\n{y[2]}\n{x[2]:.3f}\n\n{y[3]}\n{x[3]:.3f}")
            self.old_x = xi

        return (in_data, pyaudio.paContinue) if self.stream else (in_data, pyaudio.paComplete)

    def on_channel_change(self, *args):
        self.reset_predict_labels()
        channel_index = self.channels_infos[self.channels_dropdown_var.get()]['index']
        
        if self.stream:
            if self.stream.is_active():
                self.toggle_stream()
            self.stream.close()

        self.stream = self.p.open(input_device_index=channel_index, format=pyaudio.paFloat32, channels=self.channels, rate=self.sr,
            input=True, output=False, frames_per_buffer=self.chunk)
        
        self.get_natural_noise()
        self.stream.close()

        self.stream = self.p.open(input_device_index=channel_index, format=pyaudio.paFloat32, channels=self.channels, rate=self.sr,
            input=True, output=False, frames_per_buffer=self.chunk//2, start=False, stream_callback=self.predict_data)

    def create_predict_labels(self):
        self.top_3_predicts = tk.StringVar()
        tk.Label(self.root, textvariable=self.top_3_predicts, font="Helvetica 10", relief=tk.FLAT).grid(row=1, column=0)
        
        self.current_predict = tk.StringVar()
        tk.Label(self.root, textvariable=self.current_predict, font="Helvetica 14 bold", relief=tk.FLAT).grid(row=1, column=1)

        # self.reset_predict_labels()
        

    def reset_predict_labels(self):
        self.old_x = None
        self.top_3_predicts.set("top 3 predicted chords and notes :\n-\n0.000\n\n-\n0.000\n\n-\n0.000")
        self.current_predict.set("Current predict:\n-\n0.000")

    def create_channels_dropdown(self):
        self.channels_option = []
        self.channels_infos = {}
        self.channels_dropdown_var = tk.StringVar()
        self.p = pyaudio.PyAudio()
        for n in range(self.p.get_device_count()):
            infos = self.p.get_device_info_by_index(n)
            self.channels_option.append(infos['name'])
            self.channels_infos[infos['name']] = infos

        self.channels_dropdown = tk.OptionMenu(self.root, self.channels_dropdown_var, *self.channels_option).grid(row=0, column=0)
        # link function to change dropdown
        self.channels_dropdown_var.trace('w', self.on_channel_change)
        self.channels_dropdown_var.set(self.channels_option[0]) # default value

    def create_button(self):
        # Play-Pause Button
        self.play_pause = tk.Button(
            self.root, text="Play", width=15, command=self.toggle_stream
        )
        self.play_pause.grid(row=0, column=1, padx=(0, 200))

    def toggle_stream(self):
        if self.stream.is_active():
            self.play_pause.configure(text="Play")
            self.stream.stop_stream()
        else:
            self.play_pause.configure(text="Pause")
            self.stream.start_stream()

    def amp_sum(self, data, len_data):
        return np.sum(np.abs(data))/len_data

    def close(self):
        self.stream.close()
        self.p.terminate()
        self.root.destroy()
