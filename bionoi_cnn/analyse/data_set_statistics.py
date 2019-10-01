'''
calculate the mean and std of the images in a given ImageFolder
'''
# import libs
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.utils.data
import copy
import argparse

def get_args():
    parser = argparse.ArgumentParser('python')

    parser.add_argument('-op',
                        default='heme_vs_nucleotide',
                        required=False,
                        choices = ['control_vs_heme', 'control_vs_nucleotide', 'heme_vs_nucleotide'],
                        help='Operation mode, control_vs_heme, or control_vs_nucleotide')

    parser.add_argument('-data_dir_control_vs_heme',
                        default='../../control_vs_heme/',
                        required=False,
                        help='directory to load data for training, validating and testing.')

    parser.add_argument('-data_dir_control_vs_nucleotide',
                        default='../../control_vs_nucleotide/',
                        required=False,
                        help='directory to load data for training, validating and testing.')

    parser.add_argument('-data_dir_heme_vs_nucleotide',
                        default='../../heme_vs_nucleotide/',
                        required=False,
                        help='directory to load data for training, validating and testing.')

    parser.add_argument('-batch_size',
                        type=int,
                        default=256,
                        required=False,
                        help='the batch size, normally 2^n.')

    return parser.parse_args()

def dataSetStatistics(data_dir, batch_size):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.ImageFolder(data_dir,
                                               transform=transform,
                                               target_transform=None)

    m = dataset.__len__()
    print('length of entire dataset:', m)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=16)

    # calculate mean and std for training data
    mean = 0.
    std = 0.
    # m = 0 # number of samples
    for data, data_label in dataloader:
        # print(data)
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1) # reshape
        mean = mean + data.mean(2).sum(0)
        std = std + data.std(2).sum(0)
        # m = m + batch_samples

    mean = mean / m
    std = std / m
    print('mean:',mean)
    print('std:',std)

    return mean, std

if __name__ == "__main__":
    args = get_args()
    op = args.op
    if op == 'control_vs_heme':
        data_dir = args.data_dir_control_vs_heme
    elif op == 'control_vs_nucleotide':
        data_dir = args.data_dir_control_vs_nucleotide
    elif op == 'heme_vs_nucleotide':
        data_dir = args.data_dir_heme_vs_nucleotide
    batch_size = args.batch_size

    # use training set statistics to approximate the statistics of entire dataset
    data_dir = data_dir + 'train/'
    dataSetStatistics(data_dir, batch_size)
