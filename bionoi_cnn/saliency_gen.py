"""
Generate a saliency map for an image to find the "implicit attention" that
the neural networks are paying to. This method is introduced by the paper
"Deep Inside Convolutional Networks: Visualising Image Classification
Models and Saliency Maps".
After the 6 saliency maps of one .mol2 is generated, it is added to the
pickle file corresponding to this .mol2 which contains 6 dictionaries:
{'atoms':atoms, 'vor':vor}. After this, the 6 dictionaries are:
{'atoms':atoms, 'vor':vor, 'saliencyMap':saliencyMap, 'predResult':predResult}
"""

import torch
import torchvision
import os
import pickle
from torchvision import datasets
import argparse
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from utils import make_model
#from helper import imshow
from saliency_visual import ImageFolderWithPaths
from saliency_visual import saliencyMap, saliencyNorm

def get_args():
    parser = argparse.ArgumentParser('python')

    parser.add_argument('-op',
                        default='heme_vs_nucleotide',
                        required=False,
                        choices = ['control_vs_heme', 'control_vs_nucleotide', 'heme_vs_nucleotide'],
                        help='control_vs_heme, control_vs_nucleotide, heme_vs_nucleotide')

    parser.add_argument('-model_dir_control_vs_heme',
                        default='./model/control_vs_heme_resnet18.pt',
                        required=False,
                        help='file to save the model for control_vs_heme.')
    
    parser.add_argument('-model_dir_control_vs_nucleotide',
                        default='./model/control_vs_nucleotide_resnet18.pt',
                        required=False,
                        help='file to save the model for control_vs_nucleotide.')
    
    parser.add_argument('-model_dir_heme_vs_nucleotide',
                        default='./model/heme_vs_nucleotide_resnet18.pt',
                        required=False,
                        help='file to save the model for control_vs_nucleotide.')    
    
    parser.add_argument('-img_root',
                        default='../analyse/images/',
                        required=False,
                        help='root folder of original images')
    
    parser.add_argument('-map_pickle_root',
                        default='../analyse/map_pickle/',
                        required=False,
                        help='output folder containing results for combining with files in mols_pkl.')
    return parser.parse_args()

def saliencyPklGen(sourceFolder, targetFolder, model, device, transiform, classes):
    """
    gernate the saliency maps, true class and predicted class, save them in a pickle file
    in the target folder
    """
    if not os.path.exists(targetFolder):
        os.makedirs(targetFolder)
    if not os.path.exists(targetFolder+classes[0]+'/'):
        os.makedirs(targetFolder+classes[0]+'/')
    if not os.path.exists(targetFolder+classes[1]+'/'):
        os.makedirs(targetFolder+classes[1]+'/')

    dataset = ImageFolderWithPaths(sourceFolder,transform=transform,target_transform=None)
    m = len(dataset) # length of dataset
    print('source folder being processed:', sourceFolder)
    print('source folder length:', m)
    print('targetFolder:', targetFolder)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,shuffle=False, num_workers=1)

    for image, label, path in dataset:
        # file names
        trueClass = classes[label]
        fileName = os.path.basename(path)
        pickleName = fileName.split('.')[0]
        pickleName = pickleName + '.pickle'
        #print('true class:', trueClass)
        #print('pickle name to save:', pickleName)
        #print(image.shape)
        #imshow(image)

        # get saliency map and predicted result
        image = image.unsqueeze(0)
        imageSaliency, predClass = saliencyMap(model, image, device, classes)
        imageSaliency = saliencyNorm(imageSaliency)

        # dictionary to save in pickle file
        dict = {'saliencyMap':imageSaliency, 'trueClass':trueClass, 'predClass':predClass}

        # dump the dictionary into the pickle file
        pklFile = open(targetFolder+trueClass+'/'+pickleName,'wb')
        pickle.dump(dict, pklFile)

#-----------------------------------------------------------------------------------------
if __name__ == "__main__":
    tasks = ('train', 'val', 'test')
    args = get_args()
    op = args.op
    print('Generating data for ', op)
    img_root = args.img_root
    if op == 'control_vs_heme':
        model_dir = args.model_dir_control_vs_heme
        classes = ('control','heme')
    elif op == 'control_vs_nucleotide':
        model_dir = args.model_dir_control_vs_nucleotide
        classes = ('control','nucleotide')
    elif op == 'heme_vs_nucleotide':
        model_dir = args.model_dir_heme_vs_nucleotide
        classes = ('heme', 'nucleotide')    
    map_pickle_root = args.map_pickle_root
    batch_size = 1
    print('batch size: '+str(batch_size))

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Current device: '+str(device))

    # dataset statistics
    mean_control_vs_heme = [0.6008, 0.4180, 0.6341]
    std_control_vs_heme = [0.2367, 0.1869, 0.2585]
    mean_control_vs_nucleotide = [0.5836, 0.4339, 0.6608]
    std_control_vs_nucleotide = [0.2413, 0.2006, 0.2537]
    mean_heme_vs_nucleotide = [0.6030, 0.4381, 0.6430]
    std_heme_vs_nucleotide = [0.2378, 0.1899, 0.2494]
    if op == 'control_vs_heme':
        mean = mean_control_vs_heme
        std = std_control_vs_heme
    elif op == 'control_vs_nucleotide':
        mean = mean_control_vs_nucleotide
        std = std_control_vs_nucleotide
    elif op == 'heme_vs_nucleotide':
        mean = mean_heme_vs_nucleotide
        std = std_heme_vs_nucleotide

    # initialize model
    model_name = 'resnet18'
    feature_extracting = False
    model = make_model(model_name, feature_extracting = feature_extracting, use_pretrained = True)
    if torch.cuda.device_count() > 1:
        print("Using "+str(torch.cuda.device_count())+" GPUs...")
        model = nn.DataParallel(model)

    # send model to GPU
    model = model.to(device)

    # Load trained parameters
    if str(device) == 'cpu':
        model = nn.DataParallel(model) # model is saved as nn.DataParallel() 
        model.load_state_dict(torch.load(model_dir, map_location='cpu'))
    else:
        model.load_state_dict(torch.load(model_dir))

    # define a transform with mean and std, another transform without them.
    transform = transforms.Compose(
              [transforms.ToTensor(),
               transforms.Normalize((mean[0], mean[1], mean[2]), (std[0], std[1], std[2]))
              ])

    for task in tasks:
        ImageFolder = img_root + op + '/' + task + '/'
        mapPickleOutFolder = map_pickle_root + op + '/' + task + '/'
        saliencyPklGen(ImageFolder, mapPickleOutFolder, model, device, transform, classes)
