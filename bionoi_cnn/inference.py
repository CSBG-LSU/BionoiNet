import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import os
import numpy as np
import argparse
import collections
import pickle
#from neuralNet import neuralNet
from utils import make_model
from saliency_visual import ImageFolderWithPaths

def get_args():
    """
    The parser function to set the default vaules of the program parameters or read the new value set by user.
    """
    parser = argparse.ArgumentParser('python')

    parser.add_argument('-op',
                        default='control_vs_heme',
                        required=False,
                        choices = ['control_vs_heme', 'control_vs_nucleotide', 'heme_vs_nucleotide'],
                        help='control_vs_heme, control_vs_nucleotide, heme_vs_nucleotide')

    parser.add_argument('-data_dir_control_vs_heme',
                        default='../analyse/images/control_vs_heme/',
                        required=False,
                        help='directory to load data for training, validating and testing.')

    parser.add_argument('-data_dir_control_vs_nucleotide',
                        default='../analyse/images/control_vs_nucleotide/',
                        required=False,
                        help='directory to load data for training, validating and testing.')

    parser.add_argument('-data_dir_heme_vs_nucleotide',
                        default='../analyse/images/heme_vs_nucleotide/',
                        required=False,
                        help='directory to load data for training, validating and testing.')

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

    parser.add_argument('-log_dir',
                        default='./log/',
                        required=False,
                        help='directory to save results.')                        
    return parser.parse_args()

def inference(model, dataloader, device, op, log_dir, classes):
    """
    inference the model on test data
    """
    model.eval() # set model to evaluation mode
    epoch_labels = [] # true labels
    epoch_preds = [] # predictions
    epoch_outproba = [] # output probabilities

    # send model to GPU
    model = model.to(device)

    # Iterate over data.
    num_correct = 0 # number of correct predictions
    correct_preds = [] # list containing files that are predicted correctly
    for image, label, path in dataloader:
        image = image.to(device)
        label = label.to(device)

        # calculate output of neural network
        output = model(image)

        # sigmoid output, so >0.0 means sigmoid(preds>0.5)
        pred = (output > 0.0)
        pred = pred.item()
        label = label.item()

        # add the file to the list if it is predicted correctly.
        #print(pred)
        #print(label)
        if pred == label:
            #print(path)
            file_name = os.path.basename(path[0])
            true_class = classes[label]
            file_dict = {'name':file_name, 'type':true_class}
            correct_preds.append(file_dict)
            num_correct = num_correct + 1

    return num_correct, correct_preds
    
if __name__ == "__main__":
    '''
    Run this file to load and inference the trained model on the test data set
    '''    
    # get input parameters for the program
    args = get_args()
    op = args.op
    if op == 'control_vs_heme':
        data_dir = args.data_dir_control_vs_heme
        model_dir = args.model_dir_control_vs_heme
        classes = ('control', 'heme')
    elif op == 'control_vs_nucleotide':
        data_dir = args.data_dir_control_vs_nucleotide
        model_dir = args.model_dir_control_vs_nucleotide
        classes = ('control', 'nucleotide')
    elif op == 'heme_vs_nucleotide':
        data_dir = args.data_dir_heme_vs_nucleotide
        model_dir = args.model_dir_heme_vs_nucleotide
        classes = ('heme', 'nucleotide')    
    log_dir = args.log_dir

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

    # define a transform with mean and std, another transform without them.
    transform = transforms.Compose(
              [transforms.ToTensor(),
               transforms.Normalize((mean[0], mean[1], mean[2]), (std[0], std[1], std[2]))
              ])

    dataset = ImageFolderWithPaths(data_dir+'test',transform=transform,target_transform=None)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,shuffle=False, num_workers=1)
 
    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Current device: '+str(device))

    # initialize model
    modelName = 'resnet18'
    feature_extracting = False
    net = make_model(modelName, feature_extracting = feature_extracting, use_pretrained = True)
    print(net)

    if torch.cuda.device_count() > 1:
        print("Using "+str(torch.cuda.device_count())+" GPUs...")
        net = nn.DataParallel(net)

    # Load trained parameters
    net.load_state_dict(torch.load(model_dir))

    # run trained model on test dataset
    num_correct, correct_preds = inference(net, dataloader, device, op, log_dir, classes)
    print('number of correct predictions:', num_correct)
    m = len(dataset)
    acc = num_correct / m
    print('testing accuracy:', acc)
    print('correctly predicted files:')
    #print(correct_preds)

    #for p in correct_preds: print(p)

    # compute the results of voting:
    # there are 6 directions for 1 file,
    # as long as there is >=4 corrected predictions,
    # it is considered a correct prediction.
    correct_preds_prefix_0 = []
    correct_preds_prefix_1 = []
    for pred in correct_preds:
        if pred['type'] == classes[0]:
            correct_preds_prefix_0.append((pred['name'].split('-')[0], pred['type']))
        elif pred['type'] == classes[1]:
            correct_preds_prefix_1.append((pred['name'].split('-')[0], pred['type']))
        else:
            print('error: class names don not match!')    

    correct_preds_list_0 = [item for item, count in collections.Counter(correct_preds_prefix_0).items() if count >= 4]
    correct_preds_list_1 = [item for item, count in collections.Counter(correct_preds_prefix_1).items() if count >= 4]
    print('number of correctly predicted {}: {}'.format(classes[0], len(correct_preds_list_0)))
    print('number of correctly predicted {}: {}'.format(classes[1], len(correct_preds_list_1)))

    correct_preds_dict = {0:correct_preds_list_0, 1:correct_preds_list_1}
    pickle_file = open(log_dir + op + '_correct_preds_6_dirs_vote' + '.pickle', 'wb')
    pickle.dump(correct_preds_dict, pickle_file)