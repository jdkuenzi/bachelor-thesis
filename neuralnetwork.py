#
# Author    : Jean-Daniel Kuenzi
# Date      : 03.05.2021
# Update    : 15.08.2021
# Desc      : File containing the different architectures of PyTorch models
# Version   : 1.0.0
#

import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class AverageMeter(object):
    """Computes and stores the average"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def avg(self):
        return self.sum / self.count

class AccuracyMeter(object):
    """Computes and stores the accuracy and return it in percent"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.correct = 0
        self.total = 0

    def update(self, correct, total):
        self.correct += correct
        self.total += total

    def acc(self):
        return 100 * self.correct / self.total

class DAE(nn.Module):
    def __init__(self):
        super(DAE, self).__init__()
        self.epochs = 50
        self.criterion = None
        self.optimizer = None

        self.encoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
        )

    def forward(self, x, train=False):
        if not train:
            x = self.normalize(x)
        else:
            x = self.add_gaussian_white_noise(x, std=0.3)
        x = x.view(x.size(0), -1, 256)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(x.size(0), -1)
        x = self.normalize(x)
        return x

    def add_gaussian_white_noise(self, x, mean=0.0, std=1.0):
        noise = torch.randn_like(x) * std + mean
        return self.normalize(x + noise)

    def replace(self, x, duration, sr):
        i_0 = int(duration * sr)
        i_1 = i_0 * 2
        x[:,:i_0] = x[:,i_0:i_1]
        return x

    def normalize(self, x):
        x -= x.min(-1, keepdim=True)[0]
        x /= x.max(-1, keepdim=True)[0]
        x *= 2.0
        x -= 1.0
        return x

class FFT(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.n_fft = input_size
        self.hamming = torch.hamming_window(self.n_fft, True)

    def normalize(self, x):
        x -= x.min(-1, keepdim=True)[0]
        x /= x.max(-1, keepdim=True)[0]
        return x

    def forward(self, x):
        x *= self.hamming
        x = torch.fft.fft(x, self.n_fft).abs()
        x = x[:,:self.n_fft//2]
        x = self.normalize(x)
        return x

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, batch_size, autoencoder):
        super(MLP, self).__init__()
        self.criterion = None
        self.optimizer = None
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.epochs = 50
        self.class_names = None

        self.autoencoder_fft = nn.Sequential(
            autoencoder,
            FFT(input_size)
        )

        self.classifier = nn.Sequential(
            nn.Linear(input_size//2, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            # nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x, train=False):
        if not train:
            x = self.autoencoder_fft(x)
        x = self.classifier(x)
        return x

class LSTMNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers, batch_size, autoencoder, bidirectional=True, dropout=0.15):
        super(LSTMNN, self).__init__()
        self.criterion = None
        self.optimizer = None
        self.epochs = 50
        self.class_names = None

        # self.init_hidden(n_layers, batch_size, 128, bidirectional)
        
        self.autoencoder_fft = nn.Sequential(
            autoencoder,
            FFT(input_size),
        )

        self.dense = nn.Sequential(
            nn.Linear(input_size//2, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            # nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(128, 128, n_layers, bidirectional=bidirectional, batch_first=True, dropout=(0. if n_layers == 1 else dropout))

        self.classifier = nn.Sequential(
            nn.Linear(384, output_size),
        )

    def normalize(self, x):
        x -= x.min(-1, keepdim=True)[0]
        x /= x.max(-1, keepdim=True)[0]
        x *= 2.0
        x -= 1.0
        return x

    def forward(self, x, train=False):
        if not train:
            x = self.autoencoder_fft(x)
        x = self.dense(x)
        xi = x.reshape(x.size(0), 1, -1)
        xi, _ = self.lstm(xi)
        xi = xi.reshape(xi.size(0), -1)
        xi = torch.sigmoid(xi)
        x = torch.cat((xi, x), -1)
        x = self.classifier(x)
        return x

    # def init_hidden(self, n_layers, batch_size, hidden_dim, bidirectional):
    #     self.hidden = (
    #         torch.zeros(n_layers * 2 if bidirectional else n_layers, batch_size, hidden_dim),
    #         torch.zeros(n_layers * 2 if bidirectional else n_layers, batch_size, hidden_dim)
    #     )

class CNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, batch_size, autoencoder):
        super(CNN, self).__init__()
        self.criterion = None
        self.optimizer = None
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.epochs = 50
        self.class_names = None

        self.autoencoder_fft = nn.Sequential(
            autoencoder,
            FFT(input_size)
        )

        self.dense = nn.Sequential(
            nn.Linear(input_size//2, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            # nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.cnn_classifier = nn.Sequential(
            nn.Conv1d(1, 8, 16),
            nn.ReLU(),
            nn.Conv1d(8, 8, 8),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(208, 128),
            # nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, output_size),
        )

    def forward(self, x, train=False):
        if not train:
            x = self.autoencoder_fft(x)
        x = self.dense(x)
        x = x.reshape(x.size(0), 1, -1)
        x = self.cnn_classifier(x)
        return x