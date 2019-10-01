"""
Modules and functions for contructing the model
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.utils.data
import copy
from math import sqrt
import time
import sklearn.metrics as metrics
from helper import calc_metrics

'''
Funtions that used to initialize the model. There are two working modes: feature_extracting and fine_tuning.
In feature_extracting, only the last fully connected layer is updated. In fine_tuning, all the layers are
updated.
'''
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
    return model

# Parameters of newly constructed modules have requires_grad=True by default
def make_model(modelName, feature_extracting, use_pretrained):
    if modelName == 'resnet18':
        model = models.resnet18(pretrained = use_pretrained)
        model = set_parameter_requires_grad(model, feature_extracting)
        model.fc = nn.Linear(2048, 1)
        return model
    elif modelName == 'resnet50':
        model = models.resnet50(pretrained = use_pretrained)
        model = set_parameter_requires_grad(model, feature_extracting)
        model.fc = nn.Linear(8192, 1)
        return model
    elif modelName == "vgg11":
        model = models.vgg11(pretrained=use_pretrained)
        model = set_parameter_requires_grad(model, feature_extracting)
        model.classifier[0] = nn.Linear(32768,4096)
        model.classifier[3] = nn.Linear(4096,4096)
        model.classifier[6] = nn.Linear(4096, 1)
        return model
    elif modelName == "vgg11_bn":
        model = models.vgg11_bn(pretrained=use_pretrained)
        model.classifier[0] = nn.Linear(32768,4096)
        model.classifier[3] = nn.Linear(4096,4096)
        model.classifier[6] = nn.Linear(4096, 1)
        return model
    else:
        print("Invalid model name, exiting...")
        exit()
#-------------------------------------end of make_model-------------------------------------------


def dataPreprocess(batch_size, data_dir, seed, opMode, normalizeDataset):
    """
    Function to pre-process the data: load data, normalize data, random split data
    with a fixed seed. Return the size of input data points and DataLoaders for training,
    validation and testing.

    The mean and std are from the statistics from the entire dataset (control and heme when
    it is control_vs_heme)

    Current device:  cuda:0
    length of entire dataset: 15252
    mean: tensor([0.7022, 0.5750, 0.7241])
    std: tensor([0.1666, 0.1282, 0.1813])
    number of control data points:  11676
    number of heme datapoints:  3576
    length of training set:  12202
    length of validation set  1525
    length of testing set  1525
    """

    # seed for random splitting dataset
    torch.manual_seed(seed)

    # dataset statistics
    mean_control_vs_heme = [0.6008, 0.4180, 0.6341]
    std_control_vs_heme = [0.2367, 0.1869, 0.2585]
    mean_control_vs_nucleotide = [0.5836, 0.4339, 0.6608]
    std_control_vs_nucleotide = [0.2413, 0.2006, 0.2537]
    mean_heme_vs_nucleotide = [0.6030, 0.4381, 0.6430]
    std_heme_vs_nucleotide = [0.2378, 0.1899, 0.2494]
    if opMode == 'control_vs_heme':
        mean = mean_control_vs_heme
        std = std_control_vs_heme
    elif opMode == 'control_vs_nucleotide':
        mean = mean_control_vs_nucleotide
        std = std_control_vs_nucleotide
    elif opMode == 'heme_vs_nucleotide':
        mean = mean_heme_vs_nucleotide
        std = std_heme_vs_nucleotide

    # define the transforms we need
    # Load the training data again with normalization if needed.
    if normalizeDataset == True:
        print( 'normalizing data, mean:'+str(mean)+', std:'+str(std))
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((mean[0], mean[1], mean[2]),
                                                            (std[0], std[1], std[2]))
                                                            ])
    else:
        print( 'NOT normalizing data.')
        transform = transforms.Compose([transforms.ToTensor()])

    # load training set
    trainset = torchvision.datasets.ImageFolder(data_dir+'/train',
                                               transform=transform,
                                               target_transform=None)
    # load validation set
    valset = torchvision.datasets.ImageFolder(data_dir+'/val',
                                               transform=transform,
                                               target_transform=None)
    # load testing set
    testset = torchvision.datasets.ImageFolder(data_dir+'/test',
                                               transform=transform,
                                               target_transform=None)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=1)

    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                            shuffle=True, num_workers=1)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=1)

    return trainloader, valloader, testloader

def dataPreprocess_cv(batch_size, data_dir, seed, opMode, normalizeDataset):
    """
    Function to pre-process the data: load data, normalize data, random split data
    with a fixed seed. Return the size of input data points and DataLoaders for training,
    validation and testing.

    The mean and std are from the statistics from the entire dataset (control and heme when
    it is control_vs_heme)

    Current device:  cuda:0
    length of entire dataset: 15252
    mean: tensor([0.7022, 0.5750, 0.7241])
    std: tensor([0.1666, 0.1282, 0.1813])
    number of control data points:  11676
    number of heme datapoints:  3576
    length of training set:  12202
    length of validation set  1525
    length of testing set  1525
    """    
    # seed for random splitting dataset
    torch.manual_seed(seed)

    # dataset statistics
    mean_control_vs_heme = [0.6008, 0.4180, 0.6341]
    std_control_vs_heme = [0.2367, 0.1869, 0.2585]
    mean_control_vs_nucleotide = [0.5836, 0.4339, 0.6608]
    std_control_vs_nucleotide = [0.2413, 0.2006, 0.2537]
    mean_heme_vs_nucleotide = [0.6030, 0.4381, 0.6430]
    std_heme_vs_nucleotide = [0.2378, 0.1899, 0.2494]
    if opMode == 'control_vs_heme':
        mean = mean_control_vs_heme
        std = std_control_vs_heme
    elif opMode == 'control_vs_nucleotide':
        mean = mean_control_vs_nucleotide
        std = std_control_vs_nucleotide
    elif opMode == 'heme_vs_nucleotide':
        mean = mean_heme_vs_nucleotide
        std = std_heme_vs_nucleotide

    # define the transforms we need
    # Load the training data again with normalization if needed.
    if normalizeDataset == True:
        print( 'normalizing data, mean:'+str(mean)+', std:'+str(std))
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((mean[0], mean[1], mean[2]),
                                                            (std[0], std[1], std[2]))
                                                            ])
    else:
        print( 'NOT normalizing data.')
        transform = transforms.Compose([transforms.ToTensor()])

    # load training set
    trainset = torchvision.datasets.ImageFolder(data_dir+'/train',
                                               transform=transform,
                                               target_transform=None)
    # load validation set
    valset = torchvision.datasets.ImageFolder(data_dir+'/val',
                                               transform=transform,
                                               target_transform=None)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=1)

    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                            shuffle=True, num_workers=1)

    return trainloader, valloader

def control_vs_heme_config(device):
    # model name
    modelName = 'resnet18'

    # if true, only train the fully connected layers
    featureExtracting = False
    print('Feature extracting: '+str(featureExtracting))

    # whether to use pretrained model
    usePretrained = True
    print('Using pretrained model: '+str(usePretrained))

    # initialize the model
    net = make_model(modelName, feature_extracting = featureExtracting, use_pretrained = usePretrained)
    print(str(net))
    # if there are multiple GPUs, split the batch to different GPUs
    if torch.cuda.device_count() > 1:
        print("Using "+str(torch.cuda.device_count())+" GPUs...")
        net = nn.DataParallel(net)
    # send model to GPU
    net = net.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = net.parameters()
    print("Params to learn:")
    if featureExtracting:
        params_to_update = []
        for name,param in net.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print(str(name))
    else: # if finetuning
        for name,param in net.named_parameters():
            if param.requires_grad == True:
                print(str(name))

    # loss function.
    # calulating weight for loss function because number of control datapoints is much larger
    # than heme datapoints. See docs for torch.nn.BCEWithLogitsLoss
    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(params_to_update, lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.005, amsgrad=False)
    #optimizer = optim.SGD(params_to_update, lr=0.0005, weight_decay=0.01, momentum=0)
    print('optimizer:')
    print(str(optimizer))
    
    # whether to decay the learning rate
    learningRateDecay = True

    # number of epochs to train the Neural Network
    num_epochs = 4

    # learning rate schduler used to implement learning rate decay
    learningRateScheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1], gamma=0.1)

    # a dictionary contains optimizer, learning rate scheduler
    # and whether to decay learning rate
    optimizerDict = {'optimizer': optimizer,
                    'learningRateScheduler':learningRateScheduler,
                    'learningRateDecay':learningRateDecay}

    return net, loss_fn, optimizerDict, num_epochs

def control_vs_nucleotide_config(device):
    # model name
    modelName = 'resnet18'

    # if true, only train the fully connected layers
    featureExtracting = False
    print('Feature extracting: '+str(featureExtracting))

    # whether to use pretrained model
    usePretrained = True
    print('Using pretrained model: '+str(usePretrained))

    # initialize the model
    net = make_model(modelName, feature_extracting = featureExtracting, use_pretrained = usePretrained)
    print(str(net))
    # if there are multiple GPUs, split the batch to different GPUs
    if torch.cuda.device_count() > 1:
        print("Using "+str(torch.cuda.device_count())+" GPUs...")
        net = nn.DataParallel(net)
    # send model to GPU
    net = net.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = net.parameters()
    print("Params to learn:")
    if featureExtracting:
        params_to_update = []
        for name,param in net.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print(str(name))
    else: # if finetuning
        for name,param in net.named_parameters():
            if param.requires_grad == True:
                print(str(name))

    # loss function.
    # calulating weight for loss function because number of control datapoints is much larger
    # than heme datapoints. See docs for torch.nn.BCEWithLogitsLoss
    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(params_to_update, lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.005, amsgrad=False)
    print('optimizer:')
    print(str(optimizer))
    #optimizer = optim.SGD(params_to_update, lr=0.0005, weight_decay=0.01, momentum=0)

    # whether to decay the learning rate
    learningRateDecay = True

    # number of epochs to train the Neural Network
    num_epochs = 5

    # learning rate schduler used to implement learning rate decay
    learningRateScheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1,4], gamma=0.1)
    #print(learningRateScheduler)

    # a dictionary contains optimizer, learning rate scheduler
    # and whether to decay learning rate
    optimizerDict = {'optimizer': optimizer,
                    'learningRateScheduler':learningRateScheduler,
                    'learningRateDecay':learningRateDecay}

    return net, loss_fn, optimizerDict, num_epochs

def heme_vs_nucleotide_config(device):
    # model name
    modelName = 'resnet18'

    # if true, only train the fully connected layers
    featureExtracting = False
    print('Feature extracting: '+str(featureExtracting))

    # whether to use pretrained model
    usePretrained = True
    print('Using pretrained model: '+str(usePretrained))

    # initialize the model
    net = make_model(modelName, feature_extracting = featureExtracting, use_pretrained = usePretrained)
    print(str(net))
    # if there are multiple GPUs, split the batch to different GPUs
    if torch.cuda.device_count() > 1:
        print("Using "+str(torch.cuda.device_count())+" GPUs...")
        net = nn.DataParallel(net)
    # send model to GPU
    net = net.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = net.parameters()
    print("Params to learn:")
    if featureExtracting:
        params_to_update = []
        for name,param in net.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print(str(name))
    else: # if finetuning
        for name,param in net.named_parameters():
            if param.requires_grad == True:
                print(str(name))

    # loss function.
    # calulating weight for loss function because number of control datapoints is much larger
    # than heme datapoints. See docs for torch.nn.BCEWithLogitsLoss
    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(params_to_update, lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.005, amsgrad=False)
    print('optimizer:')
    print(str(optimizer))
    #optimizer = optim.SGD(params_to_update, lr=0.0005, weight_decay=0.01, momentum=0)

    # whether to decay the learning rate
    learningRateDecay = True

    # number of epochs to train the Neural Network
    num_epochs = 3

    # learning rate schduler used to implement learning rate decay
    learningRateScheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1], gamma=0.1)
    #print(learningRateScheduler)

    # a dictionary contains optimizer, learning rate scheduler
    # and whether to decay learning rate
    optimizerDict = {'optimizer': optimizer,
                    'learningRateScheduler':learningRateScheduler,
                    'learningRateDecay':learningRateDecay}

    return net, loss_fn, optimizerDict, num_epochs

def train_model(model, device, dataloaders, criterion, optimizerDict, loss_fn_pos_weight, num_epochs):
    """
    The train_model function handles the training and validation of a given model.
    As input, it takes 'a PyTorch model', 'a dictionary of dataloaders', 'a loss function', 'an optimizer',
    'a specified number of epochs to train and validate for'. The function trains for the specified number
    of epochs and after each epoch runs a full validation step. It also keeps track of the best performing
    model (in terms of validation accuracy), and at the end of training returns the best performing model.
    After each epoch, the training and validation accuracies are printed.
    """
    since = time.time()
    sigmoid = nn.Sigmoid()
    optimizer = optimizerDict['optimizer']
    learningRateScheduler = optimizerDict['learningRateScheduler']
    learningRateDecay = optimizerDict['learningRateDecay']
    if learningRateDecay:
        print('Using learning rate scheduler.')
    else:
        print('Not using andy learning rate scheduler.')

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    train_mcc_history = []
    val_mcc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_mcc = 0.0

    for epoch in range(num_epochs):
        print(' ')
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 15)

        epoch_labels_train = []
        epoch_labels_val = []
        epoch_labels = {'train':epoch_labels_train, 'val':epoch_labels_val}
        epoch_preds_train = []
        epoch_preds_val = []
        epoch_preds = {'train':epoch_preds_train, 'val':epoch_preds_val}
        epoch_outproba_val = [] # output probabilities in validation phase

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            # accumulated loss for a epoch
            running_loss = 0.0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # Get model outputs and calculate loss
                #labels = Variable(labels)
                labels = labels.view(-1,1)

                # calculate output of neural network
                outputs = model(inputs)

                # output probabilities
                outproba = sigmoid(outputs)

                # add a positive weight(see docs) to loss function
                criterion.pos_weight = torch.tensor([loss_fn_pos_weight], device=device)
                loss = criterion(outputs, labels.float())

                # sigmoid output,  preds>0.0 means sigmoid(preds)>0.5
                preds = (outputs > 0.0)
                #print( 'outpus:',outputs)
                #print( 'preds:',preds)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward() # back propagate
                    optimizer.step() # update parameters
                    optimizer.zero_grad() # zero the parameter gradients

                # accumulate loss for this single batch
                running_loss += loss.item() * inputs.size(0)

                # concatenate the list
                # need to conver from torch tensor to numpy array(n*1 dim),
                # then squeeze into one dimensional vectors
                labels = labels.cpu() # cast to integer then copy tensors to host
                preds = preds.cpu() # copy tensors to host
                outproba = outproba.cpu()
                epoch_labels[phase] = np.concatenate((epoch_labels[phase],np.squeeze(labels.numpy())))
                epoch_preds[phase] = np.concatenate((epoch_preds[phase],np.squeeze(preds.numpy())))

                # only collect validation phase's output probabilities
                if phase == 'val':
                    epoch_outproba_val = np.concatenate((epoch_outproba_val,np.squeeze(outproba.detach().numpy())))

            # loss for the epoch, averaged over all the data points
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            #print( epoch_labels[phase].astype(int))
            #print( epoch_preds[phase].astype(int))
            #print( epoch_labels[phase].astype(int).shape)
            #print( epoch_preds[phase].astype(int).shape)

            # calculate metrics using scikit-learn
            epoch_acc, epoch_precision, epoch_recall, epoch_f1, epoch_mcc = calc_metrics(epoch_labels[phase].astype(int),epoch_preds[phase].astype(int))
            print('{} Accuracy:{:.3f} Precision:{:.3f} Recall:{:.3f} F1:{:.3f} MCC:{:.3f}.'.format(phase, epoch_acc, epoch_precision, epoch_recall, epoch_f1, epoch_mcc))

            # record the training accuracy
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_loss_history.append(epoch_loss)
                train_mcc_history.append(epoch_mcc)

            # record the validation accuracy
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)
                val_mcc_history.append(epoch_mcc)

            # deep copy the model with best mcc
            if phase == 'val' and epoch_mcc > best_mcc:
                best_mcc = epoch_mcc
                best_model_wts = copy.deepcopy(model.state_dict())
                # ROC curve when mcc is best
                fpr, tpr, thresholds = metrics.roc_curve(epoch_labels[phase], epoch_outproba_val)
                # print to list so that can be saved in .json file
                fpr = fpr.tolist()
                tpr = tpr.tolist()
                thresholds = thresholds.tolist()
                best_mcc_roc = {'fpr':fpr, 'tpr':tpr, 'thresholds':thresholds}

        # decay the learning rate at end of each epoch
        if learningRateDecay:
            learningRateScheduler.step()

    time_elapsed = time.time() - since
    print( 'Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print( 'Best val MCC: {:4f}'.format(best_mcc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_mcc_roc, train_acc_history, val_acc_history, train_loss_history, val_loss_history, train_mcc_history, val_mcc_history

def test_inference(model, dataloader, device):
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
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        labels = labels.view(-1,1)

        # calculate output of neural network
        outputs = model(inputs)

        # pass outputs to sigmoid to get probabilities
        sigmoid = nn.Sigmoid() # Sigmoid() class, not function
        outproba = sigmoid(outputs)

        # sigmoid output, so >0.0 means sigmoid(preds>0.5)
        preds = (outputs > 0.0)

        labels = labels.cpu() # cast to integer then copy tensors to host
        preds = preds.cpu() # copy tensors to host
        outproba = outproba.cpu()
        epoch_labels = np.concatenate((epoch_labels,np.squeeze(labels.numpy())))
        epoch_preds = np.concatenate((epoch_preds,np.squeeze(preds.numpy())))
        epoch_outproba = np.concatenate((epoch_outproba,np.squeeze(outproba.detach().numpy())))

    # calculate metrics using scikit-learn
    epoch_acc, epoch_precision, epoch_recall, epoch_f1, epoch_mcc = calc_metrics(epoch_labels,epoch_preds)
    print('Testing Accuracy:{:.3f} Precision:{:.3f} Recall:{:.3f} F1:{:.3f} MCC:{:.3f}.'.format(epoch_acc, epoch_precision, epoch_recall, epoch_f1, epoch_mcc))

    # ROC curve
    fpr, tpr, thresholds = metrics.roc_curve(epoch_labels, epoch_outproba)
    # print to list so that can be saved in .json file
    fpr = fpr.tolist()
    tpr = tpr.tolist()
    fig = plt.figure()
    plt.plot(fpr, tpr)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title('ROC curve')
    # Specificity: When the actual value is negative, how often is the prediction correct?
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Recall)')
    plt.grid(True)

    # AUC score
    print('AUC:'+str(metrics.roc_auc_score(epoch_labels, epoch_outproba)))

