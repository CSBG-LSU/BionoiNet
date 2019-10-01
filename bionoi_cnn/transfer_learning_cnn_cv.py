'''
 Bionois hydroph classifier using transfer learning fro 10-fold cross-validation.
'''
# import libs
import torch
import json
import argparse
from utils import dataPreprocess_cv
from utils import control_vs_heme_config
from utils import control_vs_nucleotide_config
from utils import heme_vs_nucleotide_config
from utils import train_model

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

    parser.add_argument('-data_dir_control_vs_heme',
                        default='../control_vs_heme_cv/',
                        required=False,
                        help='directory to load data for 10-fold cross-validation.')

    parser.add_argument('-data_dir_control_vs_nucleotide',
                        default='../control_vs_nucleotide_cv/',
                        required=False,
                        help='directory to load data for 10-fold cross-validation.')

    parser.add_argument('-data_dir_heme_vs_nucleotide',
                        default='../heme_vs_nucleotide_cv/',
                        required=False,
                        help='directory to load data for 10-fold cross-validation.')

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
                        help='number of num_nucleotide data points, used to calculate the positive weight for the loss function.')

    return parser.parse_args()

if __name__ == "__main__":
    # get input parameters for the program
    args = get_args()
    op = args.op
    seed = args.seed
    if op == 'control_vs_heme':
        data_dir = args.data_dir_control_vs_heme
    elif op == 'control_vs_nucleotide':
        data_dir = args.data_dir_control_vs_nucleotide
    elif op == 'heme_vs_nucleotide':
        data_dir = args.data_dir_heme_vs_nucleotide
    print('data directory:', data_dir)
    batch_size = args.batch_size
    print('batch size: '+str(batch_size))
    normalizeDataset = args.normalizeDataset
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

    # lists of the results of 10 folds, each element of the list is a list containing the
    # corresponding result of 1 fold
    train_acc_history_list=[]
    val_acc_history_list=[]
    train_loss_history_list=[]
    val_loss_history_list=[]
    train_mcc_history_list=[]
    val_mcc_history_list=[]
    best_mcc_roc_list=[]

    # perfrom 10-fold cross-validation
    for i in range(10):
        print('*********************************************************************')
        print('starting {}th fold cross-validation'.format(i+1))

        # import the model and training configurations
        if op == 'control_vs_heme':
            net, loss_fn, optimizerDict, num_epochs = control_vs_heme_config(device)
        elif op == 'control_vs_nucleotide':
            net, loss_fn, optimizerDict, num_epochs = control_vs_nucleotide_config(device)
        elif op == 'heme_vs_nucleotide':
            net, loss_fn, optimizerDict, num_epochs = heme_vs_nucleotide_config(device)
        # put data into 2 DataLoaders
        data_dir_cv = data_dir + 'cv' + str(i+1)
        trainloader, valloader = dataPreprocess_cv(batch_size, data_dir_cv, seed, op, normalizeDataset)

        # create the data loader dictionary
        dataloaders_dict = {"train": trainloader, "val": valloader}

        # train the neural network
        trained_model, best_mcc_roc, train_acc_history, val_acc_history, train_loss_history, val_loss_history, train_mcc_history, val_mcc_history = train_model(net,
                                                        device,
                                                        dataloaders_dict,
                                                        loss_fn,
                                                        optimizerDict,
                                                        loss_fn_pos_weight = loss_fn_pos_weight,
                                                        num_epochs = num_epochs
                                                        )
        # pust the histories into the lists
        train_acc_history_list.append(train_acc_history)
        val_acc_history_list.append(val_acc_history)
        train_loss_history_list.append(train_loss_history)
        val_loss_history_list.append(val_loss_history)
        train_mcc_history_list.append(train_mcc_history)
        val_mcc_history_list.append(val_mcc_history)
        best_mcc_roc_list.append(best_mcc_roc)

    lists_to_save = [train_acc_history_list,
                     val_acc_history_list,
                     train_loss_history_list,
                     val_loss_history_list,
                     train_mcc_history_list,
                     val_mcc_history_list,
                     best_mcc_roc_list]

    resultFile = './log/'+op+'_cv'+'.json'
    with open(resultFile, 'w') as fp:
        json.dump(lists_to_save, fp)

    print('cross-validation finished, end of program')
