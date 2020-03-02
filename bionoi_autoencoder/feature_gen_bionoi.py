"""
Given the index of an image, take this image and feed it to a trained autoencoder,
then plot the original image and reconstructed image.
This code is used to generate features for bionoi data (10-fold cross-validation),
then use the data to train a random forest as baseline.
"""
import torch
import argparse
import pickle
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch import nn
import os
from os import listdir
from os.path import isfile, join
from utils import UnsuperviseDataset, ConvAutoencoder_conv1x1
from helper import imshow

def getArgs():
    parser = argparse.ArgumentParser('python')
    parser.add_argument('-op',
                        default='control_vs_heme',
                        required=False,
                        choices = ['control_vs_heme', 'control_vs_nucleotide', 'heme_vs_nucleotide'],
                        help='control_vs_heme, control_vs_nucleotide, heme_vs_nucleotide')

    parser.add_argument('-src_dir_control_vs_heme_cv',
                        default='../../bionoi_prj/control_vs_heme_cv/',
                        required=False,
                        help='directory to load data for cross validation.')

    parser.add_argument('-src_dir_control_vs_nucleotide_cv',
                        default='../../bionoi_prj/control_vs_nucleotide_cv/',
                        required=False,
                        help='directory to load data for cross validation.')

    parser.add_argument('-src_dir_heme_vs_nucleotide_cv',
                        default='../../bionoi_prj/heme_vs_nucleotide_cv/',
                        required=False,
                        help='directory to load data for cross validation.')

    parser.add_argument('-feature_dir_control_vs_heme_cv',
                        default='../ft_vec_control_vs_heme_cv/',
                        required=False,
                        help='directory of generated features')

    parser.add_argument('-feature_dir_control_vs_nucleotide_cv',
                        default='../ft_vec_control_vs_nucleotide_cv/',
                        required=False,
                        help='directory of generated features')                                              

    parser.add_argument('-feature_dir_heme_vs_nucleotide_cv',
                        default='../ft_vec_heme_vs_nucleotide_cv/',
                        required=False,
                        help='directory of generated features')

    parser.add_argument('-normalize',
                        default=True,
                        required=False,
                        help='whether to normalize dataset')                       
    return parser.parse_args()

def feature_vec_gen(device, model, dataset, feature_dir):
    """
    Generate feature vectors for a single folder
    """
    m = len(dataset)
    for i in range(m):
        image, file_name = dataset[i]
        file_name = file_name.split('.')
        file_name = file_name[0]
        image = image.to(device) # send to gpu
        feature_vec = model.module.encode_vec(image.unsqueeze(0)) # add one dimension
        feature_vec = np.squeeze(feature_vec)
        #print(feature_vec)
        dict = {'X':feature_vec}
        pickle_file = open(feature_dir + file_name + '.pickle','wb')
        pickle.dump(dict, pickle_file)

def feature_vec_gen_bionoi(device, model, src_dir, feature_dir, classes, normalize):
    """
    Generate feature vectors for bionoi images for 10-fold cross-validation
    """
    # data configuration
    data_mean = [0.6150, 0.4381, 0.6450]
    data_std = [0.6150, 0.4381, 0.6450]   
    if normalize == True:
        print('normalizing data:')
        print('mean:', data_mean)
        print('std:', data_std)
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((data_mean[0], data_mean[1], data_mean[2]),
                                                             (data_std[0], data_std[1], data_std[2]))])
    else:
        transform = transforms.Compose([transforms.ToTensor()])
    
    # generating features for each folder
    for i in range(10): 
        k = i + 1
        for task in ('train/', 'val/'):
            for type in classes:
                src_dir_sub = src_dir + 'cv' + str(k) + '/' + task + type + '/'
                feature_dir_sub =  feature_dir + 'cv' + str(k) + '/' + task + type + '/'
                if not os.path.exists(feature_dir_sub):
                    os.makedirs(feature_dir_sub)
                print('generating features from:', src_dir_sub)
                print('generated features stored at:', feature_dir_sub)
                img_list = [f for f in listdir(src_dir_sub)] # put images into dataset
                dataset = UnsuperviseDataset(src_dir_sub, img_list, transform=transform) 
                feature_vec_gen(device, model, dataset, feature_dir_sub)
    #-------------------------------end of generating features---------------------------------------

if __name__ == "__main__":
    args = getArgs()
    op = args.op
    normalize = args.normalize
    if op == 'control_vs_heme':
        classes = ('control', 'heme')
        src_dir = args.src_dir_control_vs_heme_cv
        feature_dir = args.feature_dir_control_vs_heme_cv
    elif op == 'control_vs_nucleotide':
        classes = ('control', 'nucleotide')
        src_dir = args.src_dir_control_vs_nucleotide_cv
        feature_dir = args.feature_dir_control_vs_nucleotide_cv
    elif op == 'heme_vs_nucleotide':
        classes = ('heme', 'nucleotide')
        src_dir = args.src_dir_heme_vs_nucleotide_cv
        feature_dir = args.feature_dir_heme_vs_nucleotide_cv

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Current device: '+str(device))   

    # model configuration
    model = ConvAutoencoder_conv1x1()
    model_file = './log_ConvAutoencoder_conv1x1_512_tanh_out/bionoi_autoencoder.pt' 
    # if there are multiple GPUs, split a batch to different GPUs
    if torch.cuda.device_count() > 1:
        print("Using "+str(torch.cuda.device_count())+" GPUs...")
        model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_file))
    #print(model)
    #--------------------model configuration ends--------------------------

    # generate features for images in data_dir
    model = model.to(device)
    model.eval() # don't cache the intermediate values
    print('Generating feature vectors...')
    feature_vec_gen_bionoi(device, model, src_dir, feature_dir, classes, normalize)