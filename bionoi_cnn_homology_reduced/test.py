import torchvision
from torchvision import transforms

data_dir = '../../data/bionoi_cnn_data/control_vs_heme_cv/cv1/train/'
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.ImageFolder(data_dir,
                                            transform=transform,
                                            target_transform=None)

print(trainset[1])
