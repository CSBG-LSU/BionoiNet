"""
Functions and classes to build the autoencoder for bionoi
"""

import torch
import time
import copy
import numpy as np
from skimage import io
from torch import nn
from torch.utils import data
import matplotlib.pyplot as plt

class UnsuperviseDataset(data.Dataset):
	"""
	A data set containing images for unsupervised learning. All the images are stored in one single folder,
	and there are no labels
	"""
	def __init__(self, data_dir, filename_list, transform=None):
		self.data_dir = data_dir
		self.filename_list = filename_list
		self.transform = transform

	def __len__(self):
		return len(self.filename_list)

	def __getitem__(self, index):
		# indexing the file
		f = self.filename_list[index]
		# load file
		#print("index:",index)
		#print(self.data_dir+f)
		img = io.imread(self.data_dir+f)
		# apply the transform
		return self.transform(img), f

class DenseAutoencoder(nn.Module):
	"""
	A vanilla-style autoencoder, takes 2d images as input, then the
	flattened vector is sent to the autoencoder, which is composed of
	multiple dense layers.
	"""
	def __init__(self, input_size, feature_size):
		"""
		input_size -- int, flattened input size
		feature_size -- int
		"""
		super(DenseAutoencoder, self).__init__()
		self.enc_fc1 = nn.Linear(input_size,256)
		self.enc_fc2 = nn.Linear(256,feature_size)
		self.dec_fc1 = nn.Linear(feature_size,256)
		self.dec_fc2 = nn.Linear(256,input_size)

		self.relu = nn.LeakyReLU(0.1)
		self.sigmoid = nn.Sigmoid()
		self.dropout = nn.Dropout(0.5)

	def flatten(self, x):
		x = x.view(x.size(0), -1)
		return x

	def encode(self, x):
		x = self.flatten(x)
		x = self.relu(self.enc_fc1(x))
		x = self.relu(self.enc_fc2(x))
		return x

	def decode(self, x):
		x = self.relu(self.dec_fc1(x))
		x = self.sigmoid(self.dec_fc2(x))
		x = self.unflatten(x)
		return x

	def unflatten(self, x):
		x = torch.reshape(x,(x.size(0),3,256,256))
		return x

	def forward(self, x):
		x = self.encode(x)
		x = self.decode(x)
		return x

class ConvAutoencoder(nn.Module):
	"""
	Convolutional style autoencoder.
	Here we implement upsampling by setting stride > 1 in the ConvTranspose2d layers.
	"""
	def __init__(self):
		super(ConvAutoencoder, self).__init__()
		self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
		self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
		self.conv3 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
		self.conv4 = nn.Conv2d(32, 16, 3, stride=1, padding=1)
		self.relu = nn.LeakyReLU(0.1)
		self.pool = nn.MaxPool2d(2, 2)
		self.transconv1 = nn.ConvTranspose2d(16, 16, 4, stride=2, padding=1)
		self.transconv2 = nn.ConvTranspose2d(16, 32, 4, stride=2, padding=1)
		self.transconv3 = nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1)
		self.transconv4 = nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1)
		self.sigmoid = nn.Sigmoid()

	def encode(self, x):
		x = self.conv1(x) # 16*256*256
		x = self.relu(x)  # 16*256*256
		x = self.pool(x)  # 16*128*128
		x = self.conv2(x) # 32*128*128
		x = self.relu(x)  # 32*128*128
		x = self.pool(x)  # 32*64*64
		x = self.conv3(x) # 32*64*64
		x = self.relu(x)  # 32*64*64
		x = self.pool(x)  # 32*32*32
		x = self.conv4(x) # 16*32*32
		x = self.relu(x)  # 16*32*32
		x = self.pool(x)  # 16*16*16
		return x

	def decode(self, x):
		x = self.transconv1(x) # 16*32*32
		x = self.relu(x)       # 16*32*32
		x = self.transconv2(x) # 32*64*64
		x = self.relu(x)	   # 32*64*64
		x = self.transconv3(x) # 32*128*128
		x = self.relu(x)	   # 32*128*128
		x = self.transconv4(x) # 3*256*256
		x = self.sigmoid(x)    # 3*256*256
		#print(x.size())
		return x

	def encode_vec(self, x):
		"""
		extract features, and then faltten the feature map as a vector
		"""
		x = self.encode(x)
		return x.view(x.size(0), -1)

	def forward(self, x):
		x = self.encode(x)
		x = self.decode(x)
		return x

class ConvAutoencoder_conv1x1(nn.Module):
	"""
	Convolutional style autoencoder.
	Here we implement upsampling by setting stride > 1 in the ConvTranspose2d layers.
	1x1 conv layers are used in last layer of encoder and first layer of decoder.
	"""
	def __init__(self):
		super(ConvAutoencoder_conv1x1, self).__init__()
		self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
		self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
		self.conv3 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
		self.conv4 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
		self.conv5 = nn.Conv2d(32, 2, 1, stride=1, padding=0)
		self.relu = nn.LeakyReLU(0.1)
		self.tanh = nn.Tanh()
		self.pool = nn.MaxPool2d(2, 2)
		self.transconv1 = nn.Conv2d(2, 32, 1, stride=1, padding=0)
		self.transconv2 = nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1)
		self.transconv3 = nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1)
		self.transconv4 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)
		self.transconv5 = nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1)
		self.sigmoid = nn.Sigmoid()

	def encode(self, x):
		x = self.conv1(x) # 16*256*256
		x = self.relu(x)  # 16*256*256
		x = self.pool(x)  # 16*128*128
		x = self.conv2(x) # 32*128*128
		x = self.relu(x)  # 32*128*128
		x = self.pool(x)  # 32*64*64
		x = self.conv3(x) # 32*64*64
		x = self.relu(x)  # 32*64*64
		x = self.pool(x)  # 32*32*32
		x = self.conv4(x) # 32*32*32
		x = self.relu(x)  # 32*32*32
		x = self.pool(x)  # 32*16*16
		x = self.conv5(x)  # 32*32*32
		x = self.tanh(x)  # 2*16*16
		#print(x.size())
		return x

	def decode(self, x):
		x = self.transconv1(x) # 32*16*16
		x = self.relu(x)       # 32*16*16 
		x = self.transconv2(x) # 32*32*32
		x = self.relu(x)       # 32*32*32
		x = self.transconv3(x) # 32*64*64
		x = self.relu(x)	   # 32*64*64
		x = self.transconv4(x) # 16*128*128
		x = self.relu(x)	   # 16*128*128
		x = self.transconv5(x)  # 16*128*128
		x = self.sigmoid(x)    # 3*256*256
		#print(x.size())
		return x

	def encode_vec(self, x):
		"""
		extract features, and then faltten the feature map as a vector
		"""
		x = self.encode(x)
		return x.view(x.size(0), -1).cpu().detach().numpy() # convert to numpy

	def forward(self, x):
		x = self.encode(x)
		x = self.decode(x)
		return x

class ConvAutoencoder_dense_out(nn.Module):
	"""
	Convolutional style autoencoder.
	Here we implement upsampling by setting stride > 1 in the ConvTranspose2d layers.
	A dense layer followed by relu is added to the end of encoder and beginning of
	decoder to force the output features to be compact and sparse.
	"""
	def __init__(self, feature_size):
		super(ConvAutoencoder_dense_out, self).__init__()
		self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
		self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
		self.conv3 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
		self.conv4 = nn.Conv2d(32, 16, 3, stride=1, padding=1)
		self.leaky_relu = nn.LeakyReLU(0.1)
		self.pool = nn.MaxPool2d(2, 2)
		self.enc_fc = nn.Linear(4096,feature_size)
		self.dec_fc = nn.Linear(feature_size,4096)		
		self.relu = nn.ReLU()
		self.transconv1 = nn.ConvTranspose2d(16, 16, 4, stride=2, padding=1)
		self.transconv2 = nn.ConvTranspose2d(16, 32, 4, stride=2, padding=1)
		self.transconv3 = nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1)
		self.transconv4 = nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1)
		self.sigmoid = nn.Sigmoid()

	def flatten(self, x):
		x = x.view(x.size(0), -1)
		return x

	def unflatten(self, x):
		x = torch.reshape(x,(x.size(0),16,16,16))
		return x

	def encode(self, x):
		x = self.conv1(x) 		# 16*256*256
		x = self.leaky_relu(x)  # 16*256*256
		x = self.pool(x)  		# 16*128*128
		x = self.conv2(x) 		# 32*128*128
		x = self.leaky_relu(x)  # 32*128*128
		x = self.pool(x)  		# 32*64*64
		x = self.conv3(x) 		# 32*64*64
		x = self.leaky_relu(x)  # 32*64*64
		x = self.pool(x)  		# 32*32*32
		x = self.conv4(x) 		# 16*32*32
		x = self.leaky_relu(x)  # 16*32*32
		x = self.pool(x)  		# 16*16*16
		x = self.flatten(x) 	# 4096
		x = self.enc_fc(x) 		# feature_size
		x = self.relu(x) 		# feature_size
		return x

	def decode(self, x):
		x = self.dec_fc(x) 		# 4096
		x = self.leaky_relu(x)  # 4096
		x = self.unflatten(x)   # 16*16*16
		x = self.transconv1(x)  # 16*32*32
		x = self.leaky_relu(x)	# 16*32*32
		x = self.transconv2(x)  # 32*64*64
		x = self.leaky_relu(x)	# 32*64*64
		x = self.transconv3(x)  # 32*128*128
		x = self.leaky_relu(x)	# 32*128*128
		x = self.transconv4(x) 	# 3*256*256
		x = self.sigmoid(x)    	# 3*256*256
		#print(x.size())
		return x

	def forward(self, x):
		x = self.encode(x)
		x = self.decode(x)
		return x

class ConvAutoencoder_deeper1(nn.Module):
	"""
	Convolutional style autoencoder.
	Here we implement upsampling by setting stride > 1 in the ConvTranspose2d layers.
	"""
	def __init__(self):
		super(ConvAutoencoder_deeper1, self).__init__()
		self.conv0 = nn.Conv2d(3, 16, 3, stride=1, padding=1)		
		self.conv1 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
		self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
		self.conv3 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
		self.conv4 = nn.Conv2d(32, 16, 3, stride=1, padding=1)
		self.relu = nn.LeakyReLU(0.1)
		self.pool = nn.MaxPool2d(2, 2)
		self.transconv1 = nn.ConvTranspose2d(16, 16, 4, stride=2, padding=1)
		self.transconv2 = nn.ConvTranspose2d(16, 32, 4, stride=2, padding=1)
		self.transconv3 = nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1)
		self.transconv4 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)
		self.transconv5 = nn.ConvTranspose2d(16, 3, 3, stride=1, padding=1)
		self.sigmoid = nn.Sigmoid()

	def encode(self, x):
		x = self.conv0(x) # 16*256*256
		x = self.relu(x)  # 16*256*256		
		x = self.conv1(x) # 16*256*256
		x = self.relu(x)  # 16*256*256
		x = self.pool(x)  # 16*128*128
		x = self.conv2(x) # 32*128*128
		x = self.relu(x)  # 32*128*128
		x = self.pool(x)  # 32*64*64
		x = self.conv3(x) # 32*64*64
		x = self.relu(x)  # 32*64*64
		x = self.pool(x)  # 32*32*32
		x = self.conv4(x) # 16*32*32
		x = self.relu(x)  # 16*32*32
		x = self.pool(x)  # 16*16*16
		return x

	def decode(self, x):
		x = self.transconv1(x) # 16*32*32
		x = self.relu(x)       # 16*32*32
		x = self.transconv2(x) # 32*64*64
		x = self.relu(x)	   # 32*64*64
		x = self.transconv3(x) # 32*128*128
		x = self.relu(x)	   # 32*128*128
		x = self.transconv4(x) # 16*256*256
		x = self.relu(x)	   # 16*256*256
		x = self.transconv5(x) # 3*256*256
		x = self.sigmoid(x)    # 3*256*256
		#print(x.size())
		return x

	def encode_vec(self, x):
		"""
		extract features, and then faltten the feature map as a vector
		"""
		x = encode(x)
		return x.view(x.size(0), -1)

	def forward(self, x):
		x = self.encode(x)
		x = self.decode(x)
		return x

def train(device, num_epochs, dataloader, model, criterion, optimizer, learningRateScheduler):
	"""
	Train the autoencoder
	"""
	train_loss_history = []

	# send model to GPU if available
	model = model.to(device)

	# need a deep copy here because weights will be updated in the future
	best_model_wts = copy.deepcopy(model.state_dict())
	best_loss = float("inf") # initial best loss
	for epoch in range(num_epochs):
		since = time.time()
		print(' ')
		print('Epoch {}/{}'.format(epoch+1, num_epochs))
		print('-' * 15)
		running_loss = 0.0
		batch_num = 0
		for images, _ in dataloader:
			batch_num = batch_num + 1
			if batch_num%200 == 0:
				print('training data on batch',batch_num)
			images = images.to(device) # send to GPU if available
			images_out = model(images)# forward
			#images = images.cpu()
			#images_out = images_out.cpu()
			loss = criterion(images,images_out)
			loss.backward() # back propagation
			optimizer.step() # update parameters
			optimizer.zero_grad() # zero out the gradients
			running_loss += loss.item() * images.size(0) # accumulate loss for this epoch
		epoch_loss = running_loss / len(dataloader.dataset)
		print('Training loss:{:4f}'.format(epoch_loss))
		train_loss_history.append(epoch_loss) # store loss of current epoch
		if epoch_loss <= best_loss:
			best_loss = epoch_loss
			best_model_wts = copy.deepcopy(model.state_dict())
		learningRateScheduler.step()

		time_elapsed = time.time() - since
		print( 'epoch complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
	#--------------------end of epoch loop--------------------------------------------------------
	print( 'Best training loss: {:4f}'.format(best_loss))

	# load best model weights
	model.load_state_dict(best_model_wts)
	return model, train_loss_history

def inference(device, image, model):
	"""
	Return the reconstructed image
	"""
	model = model.to(device)
	image = image.to(device)
	model.eval() # don't cache the intermediate values
	image_out = model(image)
	return image_out
