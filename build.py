#
# Author    : Jean-Daniel Kuenzi
# Date      : 03.05.2021
# Update    : 15.08.2021
# Desc      : File that will create the instances of the different models and train them
# Version   : 1.0.0
#

import os
import sys
import copy
import random
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import confusion_matrix
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt

import neuralnetwork as mynn

from pathlib import Path

from scipy.io.wavfile import read

from tqdm import tqdm

SAVE_MODEL_PATH = 'model/'
SAVE_DAE_PATH = SAVE_MODEL_PATH + 'dae.pt'
SAVE_MLP_PATH = SAVE_MODEL_PATH + 'mlp.pt'
SAVE_CNN_PATH = SAVE_MODEL_PATH + 'cnn.pt'
SAVE_LSTM_PATH = SAVE_MODEL_PATH + 'lstmb.pt'
SAVE_GENERATED_DATA_PATH = 'generated/'
SAVE_CONFUSION_MATRIX_PATH = 'confusion_matrix.png'
SAMPLING_RATE = 44100  # 44.1[KHz]
EXAMPLE_SIZE = 2048  # ~(0.093[s]) of a record represents an example
BATCH_SIZE = 4000 # Number of examples in a batch
DATASET_TRAIN_PATH = './dataset/train'
DATASET_VALID_PATH = './dataset/test'
NOISE_WEIGHT = 10

def data_to_temporal(data_x, data_y, win_length=1024, noise_weight=1, sampling_rate=44100, desc=None):
    pro_x = []
    pro_y = []

    hop_length = win_length // 4
    for x, y in tqdm(zip(data_x, data_y), desc=f"Processing {desc}" if desc else None, total=len(data_x)):
        sample_name = x.split('\\')[-1]
        # print("+--- Processing {}... ---+".format(sample_name))
        sr, output = read(x)
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
                    pro_y.append(y)
            else:
                is_attack = True
            i += hop_length
            
    return torch.tensor(pro_x), torch.tensor(pro_y)

def z_score(x):
    return (x - x.mean()) / torch.sqrt(x.var())

def amp_sum(data, len_data):
    return np.sum(np.abs(data))/len_data

def get_sample_from_main_data_directory(path, desc=None):
    x, y, class_names = [], [], []
    label = 0
    if os.path.exists(path):
        folders = os.listdir(path)
        for folder in tqdm(folders, desc=f"Loading {desc}" if desc else None, total=len(folders)):
            dir_path = Path(path) / folder
            audio_sample_paths = [
                os.path.join(dir_path, filepath)
                for filepath in os.listdir(dir_path)
                if filepath.endswith(".wav")
            ]
            if len(audio_sample_paths) > 0:
                x += audio_sample_paths
                y += [label] * len(audio_sample_paths)
                class_names.append(folder)
                label += 1
        print(
            "Found {} files belonging to {} classes.".format(
                len(x), len(class_names))
        )
    return x, y, class_names

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        if (height >= 0.01):
            ax.annotate('{:.2f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

def build_dae():
    dae = mynn.DAE()
    dae.to(device)

    dae.criterion = nn.MSELoss()
    dae.optimizer = torch.optim.Adam(dae.parameters(), lr=0.005)

    return dae

def build_mlp(input_size, output_size, hidden_dim, device, class_names, batch_size, autoencoder):
    model = mynn.MLP(input_size, output_size, hidden_dim, batch_size, autoencoder)
    model.to(device)
    model.class_names = class_names

    model.criterion = nn.CrossEntropyLoss()
    model.optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    return model

def build_cnn(input_size, output_size, hidden_dim, device, class_names, batch_size, autoencoder):
    model = mynn.CNN(input_size, output_size, hidden_dim, batch_size, autoencoder)
    model.to(device)
    model.class_names = class_names

    model.criterion = nn.CrossEntropyLoss()
    model.optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    return model

def build_lstm(input_size, output_size, hidden_dim, n_layers, device, class_names, batch_size, autoencoder):
    model = mynn.LSTMNN(input_size, output_size, hidden_dim, n_layers, batch_size, autoencoder)
    model.to(device)
    model.class_names = class_names

    model.criterion = nn.CrossEntropyLoss()
    model.optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    return model

def fit_autoencoder(model, train_dataset, val_dataset, device, save_model_path):
    train_losses = mynn.AverageMeter()
    val_losses = mynn.AverageMeter()
    save_model = None
    min_val_losses = None
    
    for epoch in range(model.epochs):
        train_losses.reset()
        val_losses.reset()
        model.train()
        for x, y in tqdm(train_dataset, desc=f"training epoch {epoch}", total=len(train_dataset)):
            # zero the parameter gradients
            model.zero_grad()

            # x, mu, logvar = model(x, True)
            x = model(x, True)
            loss = model.criterion(x, y)

            loss.backward()
            train_losses.update(loss.item())

            model.optimizer.step()

        model.eval()
        with torch.no_grad():
            for x, y in tqdm(val_dataset, desc=f"validating epoch {epoch}", total=len(val_dataset)):
                # validation
                x = model(x, True)
                loss = model.criterion(x, y)

                val_losses.update(loss.item())

        print(f"\ntrain_loss : {train_losses.avg():.5f} - val_loss : {val_losses.avg():.5f}\n")
        if min_val_losses is None or val_losses.avg() <= min_val_losses:
            min_val_losses = val_losses.avg()
            save_model = copy.deepcopy(model)

    torch.save(save_model, save_model_path)
    print(f'min_val_losses : {min_val_losses:.5f}')
    return save_model

def fit_model(model, train_dataset, val_dataset, device, save_model_path):
    train_losses = mynn.AverageMeter()
    train_acc = mynn.AccuracyMeter()
    val_losses = mynn.AverageMeter()
    val_acc = mynn.AccuracyMeter()

    save_model = None
    min_val_losses = None
    val_accuracy = None
    best_epoch = None
    for epoch in range(model.epochs):
        train_losses.reset()
        train_acc.reset()
        val_losses.reset()
        val_acc.reset()
        model.train()
        for x, y in tqdm(train_dataset, desc=f"training epoch {epoch}", total=len(train_dataset)):
            # zero the parameter gradients
            model.zero_grad()
            # train : forward + backward + optimize
            x = model(x, True).squeeze()
            loss = model.criterion(x, y)

            x_correct = (x.argmax(1) == y).sum()
            train_acc.update(x_correct, y.shape[0])

            loss.backward()
            train_losses.update(loss.item())

            model.optimizer.step()
        
        model.eval()
        with torch.no_grad():
            for x, y in tqdm(val_dataset, desc=f"validating epoch {epoch}", total=len(val_dataset)):
                # validation
                x = model(x, True).squeeze()
                loss = model.criterion(x, y)
                val_losses.update(loss.item())

                x_correct = (x.argmax(1) == y).sum()
                val_acc.update(x_correct, y.shape[0])
            
        print(
            f"\ntrain_loss : {train_losses.avg():.5f} - train_acc : {train_acc.correct}/{train_acc.total} = {train_acc.acc():.2f}% - val_loss : {val_losses.avg():.5f} - val_acc : {val_acc.correct}/{val_acc.total} = {val_acc.acc():.2f}%\n"
        )

        # if min_val_losses is None or val_losses.avg() < min_val_losses:
        #     min_val_losses = val_losses.avg()
        #     val_accuracy = val_acc.acc()
        #     best_epoch = epoch
        #     save_model = copy.deepcopy(model)

        if val_accuracy is None or val_acc.acc() > val_accuracy:
            min_val_losses = val_losses.avg()
            val_accuracy = val_acc.acc()
            best_epoch = epoch
            save_model = copy.deepcopy(model)

    torch.save(save_model, save_model_path)
    print(f'saved model = epoch : {best_epoch} - val_losses : {min_val_losses:.5f} - val_accuracy : {val_accuracy:.2f}%')
    return save_model

#################################################################
# This code was not written by me and was found on the internet #
# Ref. https://deeplizard.com/learn/video/0LhiS6yu2qQ           #
#################################################################
def plot_confusion_matrix(cm, classes, normalize=False, title='Matrice de confusion', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)
    plt.figure(figsize=(20,20))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), verticalalignment="center", horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    # plt.tight_layout()
    plt.ylabel('Classes correctes')
    plt.xlabel('Classes prédites')
    plt.savefig(SAVE_GENERATED_DATA_PATH + SAVE_CONFUSION_MATRIX_PATH)
    plt.close()
#################################################################

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('error command is : build.py <ARG_1> <ARG_2> <ARG_3>')
        print('ARG_1 : train_dae || load_dae')
        print('ARG_2 : train_mlp || train_lstmb || train_cnn || load_mlp || load_lstmb || load_cnn')
        print('ARG_3 : plot_confusion_matrix')
        print('<ARG_X> are optional but you need at least one')
        print('Example, to train a new CNN with a loaded DAE and plot is confusion matrix, command is :')
        print('build.py load_dae train_cnn plot_confusion_matrix')
        exit(0)
    cmd = sys.argv
    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cpu') # I dont have CUDA available, I force CPU utilisation here
    print("PyTorch using {} device".format(device))

    x, y, class_names = get_sample_from_main_data_directory(DATASET_TRAIN_PATH, desc='train dataset')
    raw_train_x, raw_train_y = data_to_temporal(x, y, win_length=EXAMPLE_SIZE, noise_weight=NOISE_WEIGHT, desc='train dataset')

    x, y, _ = get_sample_from_main_data_directory(DATASET_VALID_PATH, desc='valid dataset')
    raw_valid_x, raw_valid_y = data_to_temporal(x, y, win_length=EXAMPLE_SIZE, noise_weight=NOISE_WEIGHT, desc='valid dataset')

    print('+-------------------------------------+')
    print(f'raw_train_x shape : {raw_train_x.shape}')
    print(f'raw_train_y shape : {raw_train_y.shape}')
    print(f'raw_valid_x shape : {raw_valid_x.shape}')
    print(f'raw_valid_y shape : {raw_valid_y.shape}')
    print('+-------------------------------------+')

    train_flag = True
    save_path = ''
    autoencoder = None
    if "train_dae" in cmd:
        print("building DAE...")
        autoencoder = build_dae()
        save_path = SAVE_DAE_PATH
    else:
        if "load_dae" in cmd:
            print("loading DAE...")
            save_path = SAVE_DAE_PATH

        train_flag = False
        if save_path != '':
            autoencoder = torch.load(save_path)

    if train_flag and autoencoder is not None:
        print(autoencoder)

        train_x = raw_train_x
        train_x = autoencoder.normalize(train_x)

        val_x = raw_valid_x
        val_x = autoencoder.normalize(val_x)

        train_dataset = TensorDataset(train_x, train_x)
        val_dataset = TensorDataset(val_x, val_x)

        train_dataset = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_dataset = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=True)

        print("training autoencoder...")
        autoencoder = fit_autoencoder(autoencoder, train_dataset, val_dataset, device, save_model_path=save_path)


    train_flag = True
    save_path = ''
    model = None
    if "train_mlp" in cmd:
        print("building MLP...")
        model = build_mlp(EXAMPLE_SIZE, len(class_names), EXAMPLE_SIZE, device, class_names, BATCH_SIZE, autoencoder)
        save_path = SAVE_MLP_PATH
    elif "train_lstmb" in cmd:
        print("building LSTMB...")
        model = build_lstm(EXAMPLE_SIZE, len(class_names), EXAMPLE_SIZE, 1, device, class_names, BATCH_SIZE, autoencoder)
        save_path = SAVE_LSTM_PATH
    elif "train_cnn" in cmd:
        print("building CNN...")
        model = build_cnn(EXAMPLE_SIZE, len(class_names), EXAMPLE_SIZE, device, class_names, BATCH_SIZE, autoencoder)
        save_path = SAVE_CNN_PATH
    else:
        if "load_mlp" in cmd:
            print("loading MLP...")
            save_path = SAVE_MLP_PATH
        elif "load_lstmb" in cmd:
            print("loading LSTMB...")
            save_path = SAVE_LSTM_PATH
        elif "load_cnn" in cmd:
            print('loading CNN...')
            save_path = SAVE_CNN_PATH
        
        train_flag = False
        if save_path != '':
            model = torch.load(save_path)
    

    if train_flag and model is not None:
        print(model)
        val_x = model.autoencoder_fft(raw_valid_x).detach()

        train_x = model.autoencoder_fft(raw_train_x).detach()

        val_dataset = TensorDataset(val_x, raw_valid_y)
        train_dataset = TensorDataset(train_x, raw_train_y)

        val_dataset = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=True)
        train_dataset = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        print("+------------------------------+")
        print(train_x.shape)
        print(raw_train_y.shape)
        print(val_x.shape)
        print(raw_valid_y.shape)
        print("+------------------------------+")
        print("training model...")
        model = fit_model(model, train_dataset, val_dataset, device, save_model_path=save_path)

    if "plot_confusion_matrix" in cmd and model is not None:
        print("plotting confusion matrix...")
        model.eval()
        preds = model(raw_valid_x)

        cmt = confusion_matrix(raw_valid_y, preds.argmax(1))

        plot_confusion_matrix(cmt, class_names)
