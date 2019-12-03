"""
10-fold cross validation on feature v
"""
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import copy
import os
import sklearn.metrics as metrics
from statistics import mean 
def get_args():
    parser = argparse.ArgumentParser('python')

    parser.add_argument('-op',
                        default='control_vs_heme',
                        required=False,
                        choices = ['control_vs_heme', 'control_vs_nucleotide'],
                        help="'control_vs_heme', 'control_vs_nucleotide'")

    parser.add_argument('-seed',
                        default=77777,
                        required=False,
                        help='seed for random number generation.')

    parser.add_argument('-data_dir_control_vs_heme',
                        default='../../bionoi_autoencoder_prj/autoencoder_vec_control_vs_heme/',
                        required=False,
                        help='directory to load data for 10-fold cross-validation.')

    parser.add_argument('-data_dir_control_vs_nucleotide',
                        default='../../bionoi_autoencoder_prj/autoencoder_vec_control_vs_nucleotide/',
                        required=False,
                        help='directory to load data for 10-fold cross-validation.')

    parser.add_argument('-log_dir',
                        default='./log_autoencoder_mlp_pytorch/',
                        required=False,
                        help='directory to load data for 10-fold cross-validation.')

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

# model, MLP for binary classification
class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 1)
        self.relu = nn.LeakyReLU(0.1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)

    def flatten(self, x):
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))        
        x = self.dropout(x)
        x = self.fc5(x)
        return x

# loader for dataset folder
def load_pkl(pkl):
    """
    load and unpack the pickle file.
    """
    f = open(pkl, 'rb')
    dict = pickle.load(f)
    f.close()
    ft = np.array(dict['X'], dtype='f')
    return ft

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
                #print(labels)

                # calculate output of neural network
                outputs = model(inputs)
                #print(outputs)

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
            epoch_acc, epoch_precision, epoch_recall, epoch_f1, epoch_mcc, epoch_confusion_matrix = calc_metrics(epoch_labels[phase].astype(int),epoch_preds[phase].astype(int))
            print('{} Accuracy:{:.3f} Precision:{:.3f} Recall:{:.3f} F1:{:.3f} MCC:{:.3f}.'.format(phase, epoch_acc, epoch_precision, epoch_recall, epoch_f1, epoch_mcc))
            tn, fp, fn, tp = epoch_confusion_matrix.ravel()
            print('(tn, fp, fn, tp): ', (tn, fp, fn, tp))

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
                # metrics of the model when mcc is best
                best_mcc_acc = epoch_acc
                best_mcc_precision = epoch_precision
                best_mcc_recall = epoch_recall
                best_mcc_cm = (tn, fp, fn, tp)# confusion matrix

        # decay the learning rate at end of each epoch
        if learningRateDecay:
            learningRateScheduler.step()

    time_elapsed = time.time() - since
    print( 'Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print( 'Best val MCC: {:4f}'.format(best_mcc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    # metrics of epoch with best mcc
    best_mcc_metrics = {'acc': best_mcc_acc,
                        'precision': best_mcc_precision,
                        'recall':best_mcc_recall,
                        'mcc':best_mcc,
                        'roc':best_mcc_roc,
                        'confusion_matrix':best_mcc_cm}
    return model, best_mcc_metrics, train_acc_history, val_acc_history, train_loss_history, val_loss_history, train_mcc_history, val_mcc_history

def dataPreprocess_cv(batch_size, data_dir, seed):
     # seed for random splitting dataset
    torch.manual_seed(seed)

    #transform = transforms.Compose([transforms.ToTensor()])

    # load training set
    trainset = torchvision.datasets.DatasetFolder(root=data_dir+'/train',
                                                loader=load_pkl,
                                                extensions=['pickle'],
                                                #transform=transform,
                                                target_transform=None)
    # load validation set
    valset = torchvision.datasets.DatasetFolder(data_dir+'/val',
                                              loader=load_pkl,
                                              extensions=['pickle'],
                                              #transform=transform,
                                              target_transform=None)
                                              
    #print(trainset[90000], sep='\n')
    #print(valset[0], sep='\n')

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=64)

    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                            shuffle=True, num_workers=64)

    return trainloader, valloader

# both label and out should be numpy arrays containing 0s and 1s
# this is used for training/validating/testing
def calc_metrics(label, out):
    acc = metrics.accuracy_score(label, out)
    precision = metrics.precision_score(label,out)
    recall = metrics.recall_score(label,out)
    f1 = metrics.f1_score(label,out)
    mcc = metrics.matthews_corrcoef(label, out)
    confusion_matrix = metrics.confusion_matrix(label, out)
    return acc, precision, recall, f1, mcc, confusion_matrix

if __name__ == "__main__":
    args = get_args()
    op = args.op
    seed = args.seed
    if op == 'control_vs_heme':
        data_dir = args.data_dir_control_vs_heme
    elif op == 'control_vs_nucleotide':
        data_dir = args.data_dir_control_vs_nucleotide
    log_dir = args.log_dir
    print('data directory:', data_dir)
    batch_size = args.batch_size
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

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Current device: '+str(device))    

    model = MLP(512)
    print(str(model))
    # if there are multiple GPUs, split the batch to different GPUs
    if torch.cuda.device_count() > 1:
        print("Using "+str(torch.cuda.device_count())+" GPUs...")
        model = nn.DataParallel(model)

    # send model to GPU
    model = model.to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    #optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), 
                           lr=0.0001, 
                           betas=(0.9, 0.999), 
                           eps=1e-08, 
                           weight_decay=0.001, 
                           amsgrad=False)

    # number of epochs to train the Neural Network
    num_epochs = 10

    # whether to decay the learning rate
    learningRateDecay = True

    # learning rate schduler used to implement learning rate decay
    learningRateScheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3], gamma=0.1)

    # a dictionary contains optimizer, learning rate scheduler
    # and whether to decay learning rate
    optimizerDict = {'optimizer': optimizer,
                    'learningRateScheduler':learningRateScheduler,
                    'learningRateDecay':learningRateDecay}

    # train and validate 
    train_acc_history_list=[]
    val_acc_history_list=[]
    train_loss_history_list=[]
    val_loss_history_list=[]
    train_mcc_history_list=[]
    val_mcc_history_list=[]
    best_mcc_metrics_list=[]

    # metrics of each fold when mcc is best at that epoch
    val_acc_records = []
    val_precision_records = []
    val_recall_records = []
    val_mcc_records = []
    fpr_records = []
    tpr_records = []
    thresholds_records = []

    k = 10
    for i in range(k):
        print('*********************************************************************')
        print('starting {}th fold cross-validation'.format(i+1))
        data_dir_cv = data_dir + 'cv' + str(i+1)
        trainloader, valloader = dataPreprocess_cv(batch_size, data_dir_cv, seed)
        dataloaders_dict = {"train": trainloader, "val": valloader}
        trained_model, best_mcc_metrics, train_acc_history, val_acc_history, train_loss_history, val_loss_history, train_mcc_history, val_mcc_history = train_model(model,
                                                        device,
                                                        dataloaders_dict,
                                                        loss_fn,
                                                        optimizerDict,
                                                        loss_fn_pos_weight = loss_fn_pos_weight,
                                                        num_epochs = num_epochs
                                                        )

        val_acc_records.append(best_mcc_metrics['acc'])
        val_precision_records.append(best_mcc_metrics['precision'])
        val_recall_records.append(best_mcc_metrics['recall'])
        val_mcc_records.append(best_mcc_metrics['mcc'])
        roc = best_mcc_metrics['roc']
        fpr_records.append(roc['fpr'])
        tpr_records.append(roc['tpr'])
        thresholds_records.append(roc['thresholds'])
    # ------------------end of k-fold cross-validation-----------------------
    
    records = {'val_acc_records':val_acc_records,
               'val_precision_records':val_precision_records,
               'val_recall_records':val_recall_records,
               'val_mcc_records':val_mcc_records,
               'fpr_records':fpr_records,
               'tpr_records':tpr_records,
               'thresholds_records':thresholds_records}
    
    # save the metrics in a pickle file
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    pkl_file = open(log_dir+'mlp_'+op+'_cv'+'.pickle','wb') # save as pickle with the same file name
    pickle.dump(records, pkl_file) # save the pickle file

    # calclute the average metrics for 10-fold cross validation
    print('average accuracy over folds:', mean(val_acc_records))
    print('average precision over folds:', mean(val_precision_records))
    print('average recall over folds:', mean(val_recall_records))
    print('average mcc over folds:', mean(val_mcc_records))


    