'''
Helper functions for the Neural Network
'''
import torch
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import numpy as np
# calculate number of correct/incorrect predictions, true positives,
# true negatives, false positives and false negatives.
def calc_single_batch_metrics(predicts, labels):
    # returns the number of correct and incorrect predictions in a single batch
    # returns the number of true positives, true negetives, false positives and false negetives in a single batch
    # predicts: tensor with int32 data type, size (batch_size, 1)
    # labels: tensor with int32 data type, size (batch_size, 1)

    # number of correct predictions
    correct = (predicts == labels).int()
    num_correct = torch.sum(correct).item()

    # number of true positives
    tp = torch.from_numpy(correct.numpy() * labels.numpy()).int()
    num_tp = torch.sum(tp).item()

    # number of incorrect predictions
    incorrect = (predicts != labels).int() # number of incorrect predictions
    num_incorrect = torch.sum(incorrect).item()

    # number of false positives
    fp = torch.from_numpy(incorrect.numpy() * predicts.numpy()).int()
    num_fp = torch.sum(fp).item()

    # number of true negatives
    num_tn = num_correct - num_tp

    # number of false negatives
    num_fn = num_incorrect - num_fp

    #print('predicts:',predicts)
    #print('labels:',labels)
    #print('correct',correct)
    #print('numcorrect:',num_correct)
    #print('tp:',tp)
    #print('num_tp:',num_tp)
    #print('incorrect:',incorrect)
    #print('num_incorrect:',num_incorrect)
    #print('fp:',fp)
    #print('num_fp:',num_fp)

    return num_correct, num_incorrect, num_tp, num_fp, num_tn, num_fn

# test this function
#a = torch.tensor([[0],[0],[1],[1],[1]]).int()
#b = torch.tensor([[0],[1],[0],[1],[1]]).int()
#print(a.size())
#calc_single_batch_metrics(a,b)

# both label and out should be numpy arrays containing 0s and 1s
# this is used for training/validating/testing
def calc_metrics(label, out):
    acc = metrics.accuracy_score(label, out)
    precision = metrics.precision_score(label,out)
    recall = metrics.recall_score(label,out)
    f1 = metrics.f1_score(label,out)
    mcc = metrics.matthews_corrcoef(label, out)
    return acc, precision, recall, f1, mcc

# Showing images after being normalized.
def imshow(img):
    # img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

