"""
A convolutional neural network (CNN) for the reduced homolgy dataset for binding site classification.
"""
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, models, transforms, utils
from skimage import io, transform
import matplotlib.pyplot as plt
import copy
import time
import sklearn.metrics as metrics
import json

def get_args():
    parser = argparse.ArgumentParser('python')
    parser.add_argument('-op',
                        default='heme_vs_nucleotide',
                        required=False,
                        choices = ['control_vs_heme', 'control_vs_nucleotide', 'heme_vs_nucleotide'],
                        help="'control_vs_heme', 'control_vs_nucleotide', 'heme_vs_nucleotide'")
    parser.add_argument('-root_dir',
                        default='../../data_homology_reduced/images/',
                        required=False,
                        help='directory to load data for 5-fold cross-validation.')                        
    parser.add_argument('-batch_size',
                        type=int,
                        default=256,
                        required=False,
                        help='the batch size, normally 2^n.')
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
                        help='number of num_nucleotide data points, used to calculate the positive weight for the loss function.')
    return parser.parse_args()


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
    elif modelName == 'resnet18_dropout':
        model = models.resnet18(pretrained = use_pretrained)
        model = set_parameter_requires_grad(model, feature_extracting)
        model.fc = nn.Sequential(nn.Linear(2048, 256), 
                                 nn.ReLU(), 
                                 nn.Dropout(0.5),
                                 nn.Linear(256, 1))
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

"""
Model configuration for 3 tasks.
"""
def control_vs_heme_config(device):
    # model name
    modelName = 'resnet18'
    print('Model: ', modelName)

    # if true, only train the fully connected layers
    featureExtracting = False
    print('Feature extracting: '+str(featureExtracting))

    # whether to use pretrained model
    usePretrained = True
    print('Using pretrained model: '+str(usePretrained))

    # initialize the model
    net = make_model(modelName, feature_extracting = featureExtracting, use_pretrained = usePretrained)
    #print(str(net))
    # if there are multiple GPUs, split the batch to different GPUs
    if torch.cuda.device_count() > 1:
        print("Using "+str(torch.cuda.device_count())+" GPUs...")
        net = nn.DataParallel(net)
    # send model to GPU
    net = net.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    # finetuning we will be updating all parameters. However, if we are
    # doing feature extract method, we will only update the parameters
    # that we have just initialized, i.e. the parameters with requires_grad
    # is True.
    params_to_update = net.parameters()
    #print("Params to learn:")
    if featureExtracting:
        params_to_update = []
        for name,param in net.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                #print(str(name))
    #else: # if finetuning
    #    for name,param in net.named_parameters():
    #        if param.requires_grad == True:
    #            print(str(name))

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
    optimizer_dict = {'optimizer': optimizer,
                    'learningRateScheduler':learningRateScheduler,
                    'learningRateDecay':learningRateDecay}

    return net, loss_fn, optimizer_dict, num_epochs

def control_vs_nucleotide_config(device):
    # model name
    modelName = 'resnet18'
    print('Model: ', modelName)
    
    # if true, only train the fully connected layers
    featureExtracting = False
    print('Feature extracting: '+str(featureExtracting))

    # whether to use pretrained model
    usePretrained = True
    print('Using pretrained model: '+str(usePretrained))

    # initialize the model
    net = make_model(modelName, feature_extracting = featureExtracting, use_pretrained = usePretrained)
    #print(str(net))
    # if there are multiple GPUs, split the batch to different GPUs
    if torch.cuda.device_count() > 1:
        print("Using "+str(torch.cuda.device_count())+" GPUs...")
        net = nn.DataParallel(net)
    # send model to GPU
    net = net.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    # finetuning we will be updating all parameters. However, if we are
    # doing feature extract method, we will only update the parameters
    # that we have just initialized, i.e. the parameters with requires_grad
    # is True.
    params_to_update = net.parameters()
    #print("Params to learn:")
    if featureExtracting:
        params_to_update = []
        for name,param in net.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                #print(str(name))
    #else: # if finetuning
    #    for name,param in net.named_parameters():
    #        if param.requires_grad == True:
    #            print(str(name))

    # loss function.
    # calulating weight for loss function because number of control datapoints is much larger
    # than heme datapoints. See docs for torch.nn.BCEWithLogitsLoss
    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(params_to_update, 
                           #lr=0.00001,
                           #lr=0.001, 
                           lr=0.001, 
                           betas=(0.9, 0.999), 
                           eps=1e-08, 
                           weight_decay=0.0001,
                           #weight_decay=0.01, 
                           amsgrad=False)
    #optimizer = optim.SGD(params_to_update, lr=0.0005, weight_decay=0.01, momentum=0)
    print('optimizer:')
    print(str(optimizer))

    # whether to decay the learning rate
    learningRateDecay = False

    # number of epochs to train the Neural Network
    num_epochs = 2

    # learning rate schduler used to implement learning rate decay
    learningRateScheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1,4], gamma=0.1)
    #print(learningRateScheduler)

    # a dictionary contains optimizer, learning rate scheduler
    # and whether to decay learning rate
    optimizer_dict = {'optimizer': optimizer,
                    'learningRateScheduler':learningRateScheduler,
                    'learningRateDecay':learningRateDecay}

    return net, loss_fn, optimizer_dict, num_epochs

def heme_vs_nucleotide_config(device):
    # model name
    modelName = 'resnet18'
    print('Model: ', modelName)

    # if true, only train the fully connected layers
    featureExtracting = False
    print('Feature extracting: '+str(featureExtracting))

    # whether to use pretrained model
    usePretrained = True
    print('Using pretrained model: '+str(usePretrained))

    # initialize the model
    net = make_model(modelName, feature_extracting = featureExtracting, use_pretrained = usePretrained)
    #print(str(net))
    # if there are multiple GPUs, split the batch to different GPUs
    if torch.cuda.device_count() > 1:
        print("Using "+str(torch.cuda.device_count())+" GPUs...")
        net = nn.DataParallel(net)
    # send model to GPU
    net = net.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    # finetuning we will be updating all parameters. However, if we are
    # doing feature extract method, we will only update the parameters
    # that we have just initialized, i.e. the parameters with requires_grad
    # is True.
    params_to_update = net.parameters()
    #print("Params to learn:")
    if featureExtracting:
        params_to_update = []
        for name,param in net.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                #print(str(name))
    #else: # if finetuning
    #    for name,param in net.named_parameters():
    #        if param.requires_grad == True:
    #            print(str(name))

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
    optimizer_dict = {'optimizer': optimizer,
                    'learningRateScheduler':learningRateScheduler,
                    'learningRateDecay':learningRateDecay}

    return net, loss_fn, optimizer_dict, num_epochs
#-------------------------------end of model configuration-------------------------------------

"""
Funcitons and modules to generate dataloaders for training and validation
"""
class BionoiDatasetCV(Dataset):
    """
    Dataset for bionoi, cam be used to load multiple folds for training or single fold for validation and testing
    """
    def __init__(self, op, root_dir, folds, transform=None):
        """
        Args:
            op: operation mode, heme_vs_nucleotide, control_vs_heme or control_vs_nucleotide. 
            root_dir: folder containing all images.
            folds: a list containing folds to generate training data, for example [1,2,3,4], [5].
            transform: transform to be applied to images.
        """
        self.op = op
        self.root_dir = root_dir
        self.folds = folds
        self.transform = transform
        print('--------------------------------------------------------')
        if len(folds) == 1:
            print('generating validation dataset...')
        else:
            print('generating training dataset...')
        print('folds: ', self.folds)
        
        self.pocket_classes = self.op.split('_vs_') # two classes of binding sites
        assert(len(self.pocket_classes)==2)
        self.class_name_to_int = {}
        for i in range(len(self.pocket_classes)):
            self.class_name_to_int[self.pocket_classes[i]] = i
        print('class name to integer map: ', self.class_name_to_int)

        self.folder_dirs = [] # directory of folders containing training images                                 
        self.folder_classes = [] # list of classes of each folder, integer        
        for pocket_class in self.pocket_classes:
            for fold in self.folds:
                self.folder_classes.append(self.class_name_to_int[pocket_class])
                self.folder_dirs.append(self.root_dir + pocket_class + '/fold_' + str(fold))
        print('getting training data from following folders: ', self.folder_dirs)
        print('folder classes: ', self.folder_classes)

        self.list_of_files_list = [] # directory of files for all folds, [[files for fold 1], [files for fold 2], ...]
        for folder_dir in self.folder_dirs:
            self.list_of_files_list.append([os.path.join(folder_dir, name) for name in os.listdir(folder_dir) if os.path.isfile(os.path.join(folder_dir, name))])
        #print(self.list_of_files_list)

        self.folder_dirs_lengths = [] # lengths of the folders
        for files_list in self.list_of_files_list:
            self.folder_dirs_lengths.append(len(files_list))
        print('lengths of the folders: ', self.folder_dirs_lengths)

    def __len__(self):
        return sum(self.folder_dirs_lengths)

    def __getitem__(self, idx):
        """
        Built-in function to retrieve item(s) in the dataset.
        Args:
            idx: index of the image
        """
        #if torch.is_tensor(idx):
        #    idx = idx.tolist()

        # get image directory
        folder_idx, sub_idx = self.__locate_file(idx)
        img_dir = self.list_of_files_list[folder_idx][sub_idx]
        
        # read image
        image = io.imread(img_dir)

        # get label 
        label = self.__get_class_int(folder_idx)

        # apply transform to PIL image if applicable
        if self.transform:
            image = self.transform(image)

        return image, label

    def __locate_file(self, idx):
        """
        Function to locate the directory of file
        Args:
            idx: an integer which is the index of the file.
        """
        low = 0
        up = 0
        for i in range(len(self.folder_dirs_lengths)):
            up += self.folder_dirs_lengths[i]
            if idx >= low and idx <up:
                sub_idx = idx - low
                #print('folder:', i)
                #print('sub_idx:', sub_idx)
                #print('low:', low)
                #print('up', up)
                return i, sub_idx
            low = up  

    def __get_class_int(self, folder_idx):
        """
        Function to get the label of an image as integer
        """
        return self.folder_classes[folder_idx]


def gen_loaders(op, root_dir, training_folds, val_fold, batch_size, shuffle=True, num_workers=1):
    """
    Function to generate dataloaders for cross validation
    Args:
        op: operation mode, heme_vs_nucleotide, control_vs_heme or control_vs_nucleotide. 
        root_dir : folder containing all images.
        training_folds: list of integers indicating the folds, e.g: [1,2,3,4]
        val_fold: integer, which fold is used for validation, the other folds are used for training. e.g: 5
        batch_size: integer, number of samples to send to CNN.
    """
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

    # transform to datasets        
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((mean[0], mean[1], mean[2]), (std[0], std[1], std[2]))])

    training_set = BionoiDatasetCV(op=op, root_dir=root_dir, folds=training_folds, transform=transform)
    val_set = BionoiDatasetCV(op=op, root_dir=root_dir, folds=[val_fold], transform=transform)
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_loader, val_loader
#----------------------------end of dataloader generation-------------------------------------

"""
Training helper function
"""
def calc_metrics(label, out):
    """
    Both label and out should be numpy arrays containing 0s and 1s.
    This is used for training/validating/testing.
    """
    acc = metrics.accuracy_score(label, out)
    precision = metrics.precision_score(label,out)
    recall = metrics.recall_score(label,out)
    f1 = metrics.f1_score(label,out)
    mcc = metrics.matthews_corrcoef(label, out)
    return acc, precision, recall, f1, mcc

def train_model(model, device, dataloaders, criterion, optimizer_dict, loss_fn_pos_weight, num_epochs):
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
    optimizer = optimizer_dict['optimizer']
    learningRateScheduler = optimizer_dict['learningRateScheduler']
    learningRateDecay = optimizer_dict['learningRateDecay']
    if learningRateDecay:
        print('Using learning rate scheduler.')
    else:
        print('Not using andy learning rate scheduler.')

    train_loss_history = []
    train_acc_history = []
    train_precision_history=[]
    train_recall_history=[]
    train_mcc_history = []
    val_loss_history = []
    val_acc_history = []
    val_precision_history=[]
    val_recall_history=[]
    val_mcc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_mcc = 0.0
    best_loss = 100000 # an impossiblely large number
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
                train_precision_history.append(epoch_precision)
                train_recall_history.append(epoch_recall)
                train_mcc_history.append(epoch_mcc)

            # record the validation accuracy
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)
                val_precision_history.append(epoch_precision)
                val_recall_history.append(epoch_recall)
                val_mcc_history.append(epoch_mcc)

            # save the model and metrics when the loss is minimum
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                # ROC curve when mcc is best
                fpr, tpr, thresholds = metrics.roc_curve(epoch_labels[phase], epoch_outproba_val)
                # print to list so that can be saved in .json file
                fpr = fpr.tolist()
                tpr = tpr.tolist()
                thresholds = thresholds.tolist()
                best_val_loss_roc = {'fpr':fpr, 'tpr':tpr, 'thresholds':thresholds}
                best_val_loss_metrics = {'loss': epoch_loss, 'acc': epoch_acc, 'precision': epoch_precision, 'recall': epoch_recall, 'mcc': epoch_mcc}

        # decay the learning rate at end of each epoch
        if learningRateDecay:
            learningRateScheduler.step()

    time_elapsed = time.time() - since
    print( 'Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print( 'Best val MCC: {:4f}'.format(best_mcc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    history = {'train_loss':train_loss_history,
               'train_acc':train_acc_history,
               'train_precision':train_precision_history,
               'train_recall':train_recall_history,
               'train_mcc':train_mcc_history,
               'val_loss':val_loss_history,
               'val_acc':val_acc_history,
               'val_precision':val_precision,
               'val_recall':val_recall_history,
               'val_mcc':val_mcc_history}

    return model, best_val_loss_roc, best_val_loss_metrics, history
#------------------------------------------end of training helper function---------------------------------------


if __name__ == "__main__":
    torch.manual_seed(42)
    args = get_args()
    op = args.op    
    root_dir = args.root_dir
    batch_size = args.batch_size
    print('data directory:', root_dir)
    print('batch size: '+str(batch_size))
    num_control = args.num_control
    num_heme = args.num_heme
    num_nucleotide = args.num_nucleotide

    # calculate positive weight for loss function
    if op == 'control_vs_heme':
        print('performing control_vs_heme cross-validation task.')
        loss_fn_pos_weight = num_control/num_heme
    elif op == 'control_vs_nucleotide':
        print('performing control_vs_nucleotide cross-validation task.')
        loss_fn_pos_weight = num_control/num_nucleotide
    elif op == 'heme_vs_nucleotide':
        print('performing heme_vs_nucleotide cross-validation task.')
        loss_fn_pos_weight = num_heme/num_nucleotide 

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Current device: '+str(device))

    # lists of the results of 5 folds, each element of the list is a list containing the
    # corresponding result of 1 fold
    history_over_folds=[]
    best_val_loss_metrics_over_folds=[]
    best_val_loss_roc_over_folds=[]

    # perfrom 5-fold cross-validation
    for i in range(5):
        print('*********************************************************************')
        print('starting {}th fold cross-validation'.format(i+1))
        folds = [1, 2, 3, 4, 5]
        val_fold = i+1
        folds.remove(val_fold)

        # create dataloaders
        train_loader, val_loader = gen_loaders(op=op, 
                                               root_dir=root_dir, 
                                               training_folds=folds,
                                               val_fold=val_fold, 
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=4)
        dataloaders_dict = {"train": train_loader, "val": val_loader}


        # import the model and training configurations
        if op == 'control_vs_heme':
            net, loss_fn, optimizer_dict, num_epochs = control_vs_heme_config(device)
        elif op == 'control_vs_nucleotide':
            net, loss_fn, optimizer_dict, num_epochs = control_vs_nucleotide_config(device)
        elif op == 'heme_vs_nucleotide':
            net, loss_fn, optimizer_dict, num_epochs = heme_vs_nucleotide_config(device)

        # train the neural network
        trained_model, best_val_loss_roc, best_val_loss_metrics, history = train_model(net,
                                                        device,
                                                        dataloaders_dict,
                                                        loss_fn,
                                                        optimizer_dict,
                                                        loss_fn_pos_weight = loss_fn_pos_weight,
                                                        num_epochs = num_epochs
                                                        )

        # put the histories into the lists
        best_val_loss_roc_over_folds.append(best_val_loss_roc)
        best_val_loss_metrics_over_folds.append(best_val_loss_metrics)
        history_over_folds.append(history)
    #----------------------------------end of training---------------------------------------

    lists_to_save = [best_val_loss_metrics_over_folds, best_val_loss_roc_over_folds, history_over_folds]
    print(lists_to_save)

    #resultFile = './log/'+op+'_cv'+'.json'
    #with open(resultFile, 'w') as fp:
    #    json.dump(lists_to_save, fp)

    # print results of the folds when validation loss reaches minimum.

    print('cross-validation finished, end of program')