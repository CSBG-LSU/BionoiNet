"""
Helper functions and modules
"""
import torch
import numpy as np
from skimage import io
from torch.utils import data
import matplotlib.pyplot as plt 

def imshow(img):
    """
    display an image tensor
    """
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def list_plot(lst):
    """
    plot a list 
    """
    fig = plt.figure()
    plt.plot(lst,label='training loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title("Training Loss")
    plt.legend()
    plt.draw()
    fig.savefig('./log/loss.png', dpi=fig.dpi)  
