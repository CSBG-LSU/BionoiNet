"""
Given the index of an image, take this image and feed it to a trained autoencoder,
then plot the original image and reconstructed image.
This code is used to visually verify the correctness of the autoencoder
"""
import torch
import argparse
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch import nn
from os import listdir
import random
import os.path
from os.path import isfile, join, isdir
from utils import UnsuperviseDataset, inference
from utils import DenseAutoencoder 
from utils import ConvAutoencoder
from utils import ConvAutoencoder_dense_out
from utils import ConvAutoencoder_conv1x1
from utils import ConvAutoencoder_conv1x1_layertest
from utils import ConvAutoencoder_deeper1
from dataset_statistics import dataSetStatistics

def getArgs():
    parser = argparse.ArgumentParser('python')
    parser.add_argument('-data_dir',
                        default='../bae-data-images/',
                        required=False,
                        help='directory of training images')
    parser.add_argument('-style',
                        default='conv_1x1',
                        required=False,
                        choices=['conv', 'dense', 'conv_dense_out', 'conv_1x1', 'conv_deeper', 'conv_1x1_test'],
                        help='style of autoencoder')
    parser.add_argument('-feature_size',
                        default=1024,
						type=int,
                        required=False,
                        help='size of output feature of the autoencoder')                                                
    parser.add_argument('-normalize',
                        default=True,
                        required=False,
                        help='whether to normalize dataset')
    parser.add_argument('-model',
                        default='./log/conv_1x1.pt',
                        required=False,
                        help='trained model location')
    parser.add_argument('-num_data',
                        type=int,
                        default=50000,
                        required=False,
                        help='the batch size, normally 2^n.')
    parser.add_argument('-gpu_to_cpu',
                        default=False,
                        required=False,
                        help='whether to reconstruct image using model made with gpu on a cpu')
    parser.add_argument('-img_count',
                        type=int,
                        default=10,
                        required=False,
                        help='how many reconstructed images to save and show')
    return parser.parse_args()


if __name__ == "__main__":
    args = getArgs()
    data_dir = args.data_dir
    style = args.style
    feature_size = args.feature_size
    normalize = args.normalize
    model_file = args.model
    num_data = args.num_data
    gpu_to_cpu = args.gpu_to_cpu
    img_count = args.img_count


# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Current device: '+str(device))

# Normalizing data and transforming images to tensors
statistics = dataSetStatistics(data_dir, 128, num_data)
data_mean = statistics[0].tolist()
data_std = statistics[1].tolist()

if normalize == True:
    print('normalizing data:')
    print('mean:', data_mean)
    print('std:', data_std)
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((data_mean[0], data_mean[1], data_mean[2]),
                                                         (data_std[0], data_std[1], data_std[2]))])
else:
    transform = transforms.Compose([transforms.ToTensor()])

# Put images into dataset
img_list = []
for item in listdir(data_dir):
    if isfile(join(data_dir, item)):
        img_list.append(item)
    elif isdir(join(data_dir, item)):
        update_data_dir = join(data_dir, item)
        for f in listdir( update_data_dir):
            if isfile(join(update_data_dir, f)):
                img_list.append(item + '/' + f)
            elif isdir(join(update_data_dir, f)):
                deeper_data_dir = join(update_data_dir, f)
                for y in listdir(deeper_data_dir):
                    if isfile(join(deeper_data_dir, y)):
                        img_list.append(item + '/' + f + '/' + y)

dataset = UnsuperviseDataset(data_dir, img_list, transform=transform)


# Instantiate and load model
if style == 'conv':
    model = ConvAutoencoder()
elif style == 'dense':
    model = DenseAutoencoder(input_size, feature_size)
elif style == 'conv_dense_out':
    model = ConvAutoencoder_dense_out(feature_size)
elif style == 'conv_1x1':
    model = ConvAutoencoder_conv1x1()
elif style == 'conv_1x1_test':
    model = ConvAutoencoder_conv1x1_layertest()
elif style == 'conv_deeper':
    model = ConvAutoencoder_deeper1()


# Loading the trained model
# Converting the trained model if trained to be usable on the cpu
# if gpu is unavailable and the model was trained using gpus

if gpu_to_cpu == True:
    # original saved file with DataParallel
    state_dict = torch.load(model_file, map_location='cpu')
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)

else:
    # if there are multiple GPUs, split the batch to different GPUs
    if torch.cuda.device_count() > 1:
        print("Using "+str(torch.cuda.device_count())+" GPUs...")
        model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_file))



def reconstruction_creation(index, dataset):
    '''create reconstructed image and calculate loss between original and reconstructed image'''

    image, name = dataset[index]
    reconstruct_image = inference(device, image.unsqueeze(0), model)  # create reconstructed image using model
    recon_detach = reconstruct_image.detach()
    recon_cpu = recon_detach.cpu() # send to cpu
    recon_numpy = recon_cpu.numpy()  # convert image to numpy array for easier calculations
    recon_numpy = np.squeeze(recon_numpy, axis=0)

    criterion = nn.MSELoss()
    loss = str(criterion(image.unsqueeze(0).cpu(), reconstruct_image.cpu()).item()) # calculate loss

    return recon_numpy, loss


def original_image_extraction(index, dataset):
    '''extraxt the original image and file name from dataset to plot alongside reconstructed image'''
    image, name = dataset[index]
    og_img = image.numpy()  # create list of original images as numpy arrays
    file_name = name  # create list of file names

    return og_img, file_name


# Reconstructing images and plotting the root mean square error
print('Calculating root mean squared error...')

# only calculating rmse for half of dataset due to the size.
rmse_lst = []
for index in range(int(dataset.__len__()/2.)): # remove '/.2' to calculate rmse for every img in dataset
    recon_img = reconstruction_creation(index, dataset)[0]
    og_image = original_image_extraction(index, dataset)[0]

    N = 1
    for dim in og_image.shape:   # determine N for root mean square error
        N *= dim

    rmse = ((np.sum((og_image - recon_img) ** 2) / N) ** .5)     # calculate rmse
    rmse_lst.append(rmse)


# Plot histogram of root mean squared errors between reconstructed images and original images
print('Plotting root mean squared error...')
plt.hist(np.asarray(rmse_lst), bins=30)
plt.ylabel('Number of Image Pairs')
plt.xlabel('Root Mean Squared')
plt.title('RMSE  —  ' + model_file)
plt.savefig('./images/' + 'rmse.png')
# plt.show()



# Plot images before and after reconstruction
print('Constructing figures before and after reconstruction...')

# create a random list of indices to select images from dataset
# to compare to their respective reconstructed images
random_index_lst = []
for i in range(img_count):
    random_index_lst.append(random.randint(0, dataset.__len__()-1))


original_lst = []
reconstruction_lst = []
file_name_lst = []
loss_lst = []
for index in random_index_lst: # extract and create corresponding original and reconstructed images for visual tests
    recon_img = reconstruction_creation(index, dataset)[0]
    loss = reconstruction_creation(index, dataset)[1]
    og_image = original_image_extraction(index, dataset)[0]
    file_name = original_image_extraction(index, dataset)[1]
    reconstruction_lst.append(recon_img)
    original_lst.append(og_image)
    file_name_lst.append(file_name)
    loss_lst.append(loss)


for index in range(len(file_name_lst)): # plotting the original image next to the reconstructed image with loss value
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14, 7))
    ax1.imshow(np.transpose(original_lst[index], (1, 2, 0)))
    ax1.set_title('Original Normalized Image  —  ' + file_name_lst[index])
    ax2.imshow(np.transpose(reconstruction_lst[index], (1, 2, 0)))
    ax2.set_title('Reconstructed Image  —  ' + file_name_lst[index])
    txt = 'Loss between before and after: ' + loss_lst[index]
    plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
    # show both figures
    plt.savefig('./images/' + 'reconstructed_' + file_name_lst[index].split('.')[0] + '.png')
    plt.imshow(np.transpose(original_lst[index], (1, 2, 0)))
    plt.imshow(np.transpose(reconstruction_lst[index], (1, 2, 0)))
    # plt.show()