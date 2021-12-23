#
# Author    : Jean-Daniel Kuenzi
# Date      : 03.05.2021
# Update    : 15.08.2021
# Desc      : File which will instantiate the Tkinter window and load the PyTorch model
# Version   : 1.0.0
#

import gui
import torch
import sys

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('error command is : main.py <model_path> <chunck>')
        exit(0)
    model_path = sys.argv[1]
    chunck = int(sys.argv[2])
    model = torch.load(model_path)
    window = gui.gui(model, chunck)
    window.root.mainloop()  # Starts GUI
