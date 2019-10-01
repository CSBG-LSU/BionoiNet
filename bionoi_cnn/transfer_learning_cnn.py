'''
 Bionoi CNN classifier using transfer learning

 A classifier for 'control' vs 'heme' or 'control' vs 'nucleotide'.
 A pre-trained CNN(resnet18) is used as the feature extractor. New fully-connected layers are
 stacked upon the feature extractor, and we fine-tune the entire Neural-Network since the orginal
 CNN is trained on a different data domain(imageNet)
 '''

# import libs
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import matplotlib.pyplot as plt
import time
import copy
import sklearn.metrics as metrics
import argparse
# user defined modules
from helper import imshow
from utils import dataPreprocess
from utils import control_vs_heme_config
from utils import control_vs_nucleotide_config
from utils import heme_vs_nucleotide_config
from utils import train_model
from utils import test_inference

def get_args():
    parser = argparse.ArgumentParser('python')

    parser.add_argument('-op',
                        default='heme_vs_nucleotide',
                        required=False,
                        choices = ['control_vs_heme', 'control_vs_nucleotide', 'heme_vs_nucleotide'],
                        help="'control_vs_heme', 'control_vs_nucleotide', 'heme_vs_nucleotide'")

    parser.add_argument('-seed',
                        default=123,
                        required=False,
                        help='seed for random number generation.')

    parser.add_argument('-data_dir_control_heme',
                        default='../control_vs_heme/',
                        required=False,
                        help='directory to load data for training, validating and testing.')

    parser.add_argument('-data_dir_convtrol_nucleotide',
                        default='../control_vs_nucleotide/',
                        required=False,
                        help='directory to load data for training, validating and testing.')

    parser.add_argument('-data_dir_heme_nucleotide',
                        default='../heme_vs_nucleotide/',
                        required=False,
                        help='directory to load data for training, validating and testing.')                        

    parser.add_argument('-model_dir_control_heme',
                        default='./model/control_vs_heme_resnet18.pt',
                        required=False,
                        help='file to save the model for control_vs_heme.')

    parser.add_argument('-model_dir_control_nucleotide',
                        default='./model/control_vs_nucleotide_resnet18.pt',
                        required=False,
                        help='file to save the model for control_vs_nucleotide.')

    parser.add_argument('-model_dir_heme_nucleotide',
                        default='./model/heme_vs_nucleotide_resnet18.pt',
                        required=False,
                        help='file to save the model for heme_vs_nucleotide.')

    parser.add_argument('-batch_size',
                        type=int,
                        default=256,
                        required=False,
                        help='the batch size, normally 2^n.')

    parser.add_argument('-normalizeDataset',
                        default=True,
                        required=False,
                        help='whether to normalize dataset')

    parser.add_argument('-num_control',
                        default=74784,
                        required=False,
                        help='number of control data points, used to calculate the positive weight for the loss function')

    parser.add_argument('-num_heme',
                        default=22944,
                        required=False,
                        help='number of heme data points, used to calculate the positive weight for the loss function.')

    parser.add_argument('-num_nucleotide',
                        default=59664,
                        required=False,
                        help='number of Nucleotide data points, used to calculate the positive weight for the loss function.')

    return parser.parse_args()

if __name__ == "__main__":
    # get input parameters for the program
    args = get_args()
    op = args.op
    seed = args.seed
    if op == 'control_vs_heme':
        data_dir = args.data_dir_control_heme
        model_dir = args.model_dir_control_heme
    elif op == 'control_vs_nucleotide':
        data_dir = args.data_dir_convtrol_nucleotide
        model_dir = args.model_dir_control_nucleotide
    elif op == 'heme_vs_nucleotide':
        data_dir = args.data_dir_heme_nucleotide
        model_dir = args.model_dir_heme_nucleotide
    print('data directory:', data_dir)
    print('trained model saved as:'+str(model_dir))
    batch_size = args.batch_size
    print('batch size: '+str(batch_size))
    
    normalizeDataset = args.normalizeDataset
    num_control = args.num_control
    num_heme = args.num_heme
    num_nucleotide = args.num_nucleotide
    
    # calculate positive weight for loss function
    if op == 'control_vs_heme':
        print('performing control_vs_heme task.')
        loss_fn_pos_weight = num_control/num_heme
    elif op == 'control_vs_nucleotide':
        print('performing control_vs_nucleotide task.')
        loss_fn_pos_weight = num_control/num_nucleotide
    elif op == 'heme_vs_nucleotide':
        loss_fn_pos_weight = num_heme/num_nucleotide

    
    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Current device: '+str(device))
    
    # put data into 3 DataLoaders
    trainloader, valloader, testloader = dataPreprocess(batch_size, data_dir, seed, op, normalizeDataset)
    
    # create the data loader dictionary
    dataloaders_dict = {"train": trainloader, "val": valloader}
    
    # the classes
    if op == 'control_vs_heme':
        classes = ('control', 'heme')
    elif op == 'control_vs_nucleotide':
        classes = ('control', 'nucleotide')
    elif op == 'heme_vs_nucleotide':
        classes = ('heme', 'nucleotide')
    
    # get some random training images
    #dataiter = iter(trainloader)
    #images, labels = dataiter.next()
    
    # imshow(torchvision.utils.make_grid(images)) # show images
    # print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size))) # print labels
    # print( ('size of images: ', images.size())
    # print( ('size of labels: ',labels.size())
    # print(images)  # print the image to see the value
    
    
    #--------------------------Train the Neural Network-----------------------------
    # import the model and training configurations
    if op == 'control_vs_heme':
        net, loss_fn, optimizerDict, num_epochs = control_vs_heme_config(device)
    elif op == 'control_vs_nucleotide':
        net, loss_fn, optimizerDict, num_epochs = control_vs_nucleotide_config(device)
    elif op == 'heme_vs_nucleotide':
        net, loss_fn, optimizerDict, num_epochs = heme_vs_nucleotide_config(device)
    
    # train the neural network
    trained_model, best_mcc_roc, train_acc_history, val_acc_history, train_loss_history, val_loss_history, train_mcc_history, val_mcc_history = train_model(net,
                                                     device,
                                                     dataloaders_dict,
                                                     loss_fn,
                                                     optimizerDict,
                                                     loss_fn_pos_weight = loss_fn_pos_weight,
                                                     num_epochs = num_epochs
                                                     )
    # The Plottings
    # plot train/validation loss vs # of epochs
    fig1 = plt.figure()
    plt.plot(train_loss_history,label='train')
    plt.plot(val_loss_history,label='validation')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title("Train Loss vs Validation Loss")
    plt.legend()
    plt.draw()
    
    # plot train/validation accuracy vs # of epochs
    fig2 = plt.figure()
    plt.plot(train_acc_history,label='train')
    plt.plot(val_acc_history,label='validation')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title("Train Accuracy vs Validation Accuracy")
    plt.legend()
    plt.draw()
    
    # plot train/validation mcc vs # of epochs
    fig3 = plt.figure()
    plt.plot(train_mcc_history,label='train')
    plt.plot(val_mcc_history,label='validation')
    plt.xlabel('epoch')
    plt.ylabel('MCC')
    plt.title("Train MCC vs Validation MCC")
    plt.legend()
    plt.draw()
    
    # save the trained model with best MCC value
    torch.save(trained_model.state_dict(), model_dir)
    
    # test the results on the test dataset
    test_inference(trained_model, testloader, device)
    
    # show all the figures
    plt.show()
    
    print('training finished, end of program')
    