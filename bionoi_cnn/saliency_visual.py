'''
Generate a saliency map for an image to find the "implicit attention" that
the neural networks are paying to. This method is introduced by the paper
"Deep Inside Convolutional Networks: Visualising Image Classification
Models and Saliency Maps".
We use this module to see which part of the protein is important to the neural
network to distinguish them.
'''

# import libraries
import torch
import torchvision
import os
from torchvision import datasets
import argparse
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from utils import make_model
from helper import imshow
import scipy.ndimage as ndimage
# get input arguments
def getArgs():
    parser = argparse.ArgumentParser('python')

    parser.add_argument('-opMode',
                        default='control_vs_heme',
                        required=False,
                        help='Operation mode, control_vs_heme,control_vs_nucleotide')

    parser.add_argument('-index',
                        type=int,
                        default=1,
                        required=False,
                        help='index of the target image')

    parser.add_argument('-dataDir_control_vs_heme',
                        default='../control_vs_heme/',
                        required=False,
                        help='directory to load data for training, validating and testing.')

    parser.add_argument('-dataDir_control_vs_nucleotide',
                        default='../control_vs_nucleotide/',
                        required=False,
                        help='directory to load data for training, validating and testing.')

    parser.add_argument('-modelDir_control_vs_heme',
                        default='./model/control_vs_heme_resnet18.pt',
                        required=False,
                        help='file to save the model for control_vs_heme.')

    parser.add_argument('-modelDir_control_vs_nucleotide',
                        default='./model/control_vs_nucleotide_resnet18.pt',
                        required=False,
                        help='file to save the model for control_vs_nucleotide.')                        
    return parser.parse_args()

# define our own dataset class.
class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

# the function returns the saliency map of an image
def saliencyMap(model, image, device, classes):
    # set model to evaluation mode
    model.eval()

    # send image to gpu
    image.to(device)

    # don't calculate gradients for the parameters of the model
    for param in model.parameters():
        param.requires_grad = False

    # need to calculate gradients of the input image
    image.requires_grad = True

    # output of the neural network, as in the paper, don't pass the output
    # to softmax or sigmoid
    outputs = model(image)

    # get the gradients of the input
    outputs.backward()
    saliencyMap = image.grad
    #print('size of gradients:',saliencyMap.shape)

    # convert back to numpy
    saliencyMap = saliencyMap.cpu()
    saliencyMap = np.squeeze(saliencyMap.numpy())
    #print('size of gradients:',saliencyMap.shape)
    # absolute values of the pixels
    saliencyMap = np.absolute(saliencyMap)
    #print('size of gradients:',saliencyMap.shape)
    # max value among the channels
    # dim 0: channel, dim 1, 2: height & width
    saliencyMap = np.amax(saliencyMap, axis=0)
    #print('size of gradients:',saliencyMap.shape)

    # the prediction
    predIdx = outputs.detach().cpu().numpy().squeeze().item()
    predIdx = (predIdx > 0)
    predClass = classes[int(predIdx)]
    return saliencyMap, predClass

# normalize the saliency map to have more contrast
def saliencyNorm(saliencyMap):
    maxPixel = np.amax(saliencyMap)
    minPixel = np.amin(saliencyMap)
    rangePixel = maxPixel - minPixel
    #print('range of pixels:',rangePixel)
    return np.true_divide(saliencyMap, rangePixel)

# returns original image, saliency map and file path
def imgGen(dataset, datasetWithNorm, index, classes, model, device):
    # get an image to display (256*256*3)
    imageDspl, labelDspl, pathDspl  = dataset.__getitem__(index)
    imageDspl = imageDspl.unsqueeze(0)
    imageDspl = imageDspl.squeeze().detach().numpy()
    imageDspl = np.transpose(imageDspl, (1, 2, 0))

    # get an normalized image to send to model
    image, label, path = datasetWithNorm.__getitem__(index)
    image = image.unsqueeze(0)

    # double check
    assert(path == pathDspl)
    assert(labelDspl == label)
    imgClass = classes[label]
    fileName = os.path.basename(path)

    # generate saliency map (256*256)
    imageSaliency, predClass = saliencyMap(model, image, device, classes)
    imageSaliency = saliencyNorm(imageSaliency)
    return imageDspl, imageSaliency, fileName, imgClass, predClass

if __name__ == "__main__":
    args = getArgs()
    opMode = args.opMode
    if opMode == 'control_vs_heme':
        dataDir = args.dataDir_control_vs_heme
        modelDir = args.modelDir_control_vs_heme
        classes = ['Control','Heme']
    elif opMode == 'control_vs_nucleotide':
        dataDir = args.dataDir_control_vs_nucleotide
        modelDir = args.modelDir_control_vs_nucleotide
        classes = ['Control','Nucleotide']
    index = args.index
    batchSize = 1
    print('batch size: '+str(batchSize))

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Current device: '+str(device))

    # dataset statistics
    mean_control_vs_heme = [0.6008, 0.4180, 0.6341]
    std_control_vs_heme = [0.2367, 0.1869, 0.2585]
    mean_control_vs_nucleotide = [0.5836, 0.4339, 0.6608]
    std_control_vs_nucleotide = [0.2413, 0.2006, 0.2537]
    if opMode == 'control_vs_heme':
        mean = mean_control_vs_heme
        std = std_control_vs_heme
    elif opMode == 'control_vs_nucleotide':
        mean = mean_control_vs_nucleotide
        std = std_control_vs_nucleotide

    # initialize model
    modelName = 'resnet18'
    feature_extracting = False
    net = make_model(modelName, feature_extracting = feature_extracting, use_pretrained = True)
    if torch.cuda.device_count() > 1:
        print("Using "+str(torch.cuda.device_count())+" GPUs...")
        net = nn.DataParallel(net)

    # send model to GPU
    net = net.to(device)

    # Load trained parameters
    net.load_state_dict(torch.load(modelDir))

    # define a transform with mean and std, another transform without them.
    transform = transforms.Compose([transforms.ToTensor()])
    transformWithNorm = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((mean[0], mean[1], mean[2]),
                                                        (std[0], std[1], std[2]))
                                   ])

    dataset = ImageFolderWithPaths(dataDir+'test',transform=transform,target_transform=None)
    datasetWithNorm = ImageFolderWithPaths(dataDir+'test',transform=transformWithNorm,target_transform=None)

    imageDspl, imageSaliency, fileName, imgClass, predClass = imgGen(dataset, datasetWithNorm, index, classes, net, device)

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3,ncols=1,figsize=(5, 15))
    ax1.imshow(imageDspl)
    #ax1.set_title('Original picture')
    ax2.imshow(imageSaliency)
    #ax2.set_title('Saliency map')
    imageSaliency = ndimage.gaussian_filter(imageSaliency, sigma=(10), order=0)
    ax3.imshow(imageDspl)
    ax3.imshow(imageSaliency, alpha=0.8)
    #ax3.set_title('Heatmap')
    #fig.suptitle('image type:'+imgClass+', filename: '+ fileName+', predicted class:'+predClass)
    # show both figures
    
    # Hide axis
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')
    
    # remove numbers on axes
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])    
    ax3.set_yticklabels([])
    ax3.set_xticklabels([])

    fig.subplots_adjust(wspace=-0.2, hspace=-0.2)
    fig.tight_layout()

    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)
    ax1.set_frame_on(False)

    fileName = fileName.split('.')[0]
    name = opMode + '-' + imgClass + '-' + fileName + '-' + predClass
    plt.savefig('./images/' + name + '.png',bbox_inches='tight', pad_inches=0, dpi=1024)
    plt.show()
