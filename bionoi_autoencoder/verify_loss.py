"""
Given a directory of input images. We want to verify that the loss of
reconstructing is comparable to the loss of model running on the training
data which is in the range of (0.8, 0.9).
"""

import torch
import argparse
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch import nn
from os import listdir
from os.path import isfile, join, isdir
from utils import UnsuperviseDataset, inference
from helper import imshow
from utils import DenseAutoencoder
from utils import ConvAutoencoder
from utils import ConvAutoencoder_dense_out
from utils import ConvAutoencoder_conv1x1
from dataset_statistics import dataSetStatistics

def getArgs():
    parser = argparse.ArgumentParser('python')
    parser.add_argument('-data_dir',
                        default='../bindingsite-img/',
                        required=False,
                        help='directory of training images')
    parser.add_argument('-style',
                        default='conv_1x1',
                        required=False,
                        choices=['conv', 'dense', 'conv_dense_out', 'conv_1x1'],
                        help='style of autoencoder')
    parser.add_argument('-feature_size',
                        default=1024,
						type=int,
                        required=False,
                        help='size of output feature of the autoencoder')
    parser.add_argument('-normalize',
                        default=True,
                        required=False,
                        help='whether to normalize dataset')
    parser.add_argument('-num_data',
                        type=int,
                        default=50000,
                        required=False,
                        help='the batch size, normally 2^n.')
    return parser.parse_args()


if __name__ == "__main__":
    args = getArgs()
    data_dir = args.data_dir
    style = args.style
    feature_size = args.feature_size
    normalize = args.normalize
    num_data = args.num_data
    batch_size = 1

    if style == 'conv':
        model_file = '../trained_model/bionoi_autoencoder_conv.pt'
    elif style == 'dense':
        model_file = '../trained_model/bionoi_autoencoder_dense.pt'
    elif style == 'conv_dense_out':
        model_file = './log_ConvAutoencoder_dense_out_adam/bionoi_autoencoder_conv.pt'
    elif style == 'conv_1x1':
        model_file = './log/conv1x1norm.pt'

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Current device: '+str(device))

    # instantiate and load model
    if style == 'conv':
        model = ConvAutoencoder()
    elif style == 'dense':
        model = DenseAutoencoder(input_size, feature_size)
    elif style == 'conv_dense_out':
        model = ConvAutoencoder_dense_out(feature_size)
    elif style == 'conv_1x1':
        model = ConvAutoencoder_conv1x1()

    # if there are multiple GPUs, split the batch to different GPUs
    if torch.cuda.device_count() > 1:
        print("Using "+str(torch.cuda.device_count())+" GPUs...")
        model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_file))

    statistics = dataSetStatistics(data_dir, 128, num_data)
    data_mean = statistics[0].tolist()
    data_std = statistics[1].tolist()

    if normalize == True:
        print('normalizing data:')
        print('mean:', data_mean)
        print('std:', data_std)
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((data_mean[0], data_mean[1], data_mean[2]),
                                                             (data_std[0], data_std[1], data_std[2]))])
    else:
        transform = transforms.Compose([transforms.ToTensor()])

    # put images into dataset
    img_list = []
    for item in listdir(data_dir):
        if isfile(join(data_dir, item)):
            img_list.append(item)
        elif isdir(join(data_dir, item)):
            update_data_dir = join(data_dir, item)
            for f in listdir(update_data_dir):
                if isfile(join(update_data_dir, f)):
                    img_list.append(item + '/' + f)
    dataset = UnsuperviseDataset(data_dir, img_list, transform=transform)
    # create dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=32)

    # calulate the input size (flattened)
    image, file_name = dataset[0]
    print('name of input:', file_name)
    image_shape = image.shape
    print('shape of input:', image_shape)
    input_size = image_shape[0]*image_shape[1]*image_shape[2]
    print('flattened input size:',input_size)

    # measure the loss between the 2 images
    criterion = nn.MSELoss()
    running_loss = 0.0
    model = model.to(device)
    for images, _ in dataloader:
        images = images.to(device) # send to GPU if available
        images_out = model(images)# forward
        loss = criterion(images,images_out)
        if batch_size == 1:
            print('individual loss:', loss.item())
        running_loss += loss.item() * images.size(0) # accumulate loss for this epoch
    epoch_loss = running_loss / len(dataloader.dataset) # averaging the loss over entire epoch
    print('Loss:{:4f}'.format(epoch_loss))
