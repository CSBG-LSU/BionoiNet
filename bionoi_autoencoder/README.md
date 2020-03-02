# Bionoi Autoencoder

## Description
Autoencoders for bionoi datasets to generate feature vectors for future research. Different architectures are tested.

## Dependencies
* Python 3.6.8
* Numpy 1.16.2
* Pytorch 1.1 (if using GPU acceleration, use CUDA version)
* Matplotlib 3.0.3
* Torchvision 0.4
* Scikit-image 0.15.0

## Files
* dataset_statistics.py: calculate the mean and standard deviation of the dataset, then print out. The computed statistics are used to normalize data during training. Note that if the data is normalized during training, then the data should be normalized with the same mean and variance during feature generating or reconstructing.   
```
usage: python dataset_statistics.py -data_dir ../bae-data-images/   
```

* utils.py: contains essential modules to build an autoencoder. It contains:
  * customized dataset class
  * classes for different kinds of autoencoders
  * the function used to train autoencoder. 

* autoencoder_general.py: python scripts to train an autoencoder. User can specify what style of autoencoder to use.
```
usage: python autoencoder_general.py -style conv_1x1 -data_dir ../bae-data-images/ -model_file ./log/bionoi_autoencoder_conv.pt
```

* feature_gen.py: Save the generated latent space feature vectors into a specified folder.
```
usage: python feature_gen.py -index 0 -data_dir ../bindingsite-img/ -feature_dir ../bindingsite-feature/
```

* feature_gen_foldloop.py: Save the generated latent space feature vectors into a specified folder. Data directory takes the structure shown in file. User must specify a tar file to extract data from. Tar structure must take the form of
```
tar
│
└───fold0
│   │
│   └───train
│   │   │   file111.png
│   │   │   ...
│   │
│   └───val
│       │   file112.png
│       │   ...
│
└───fold1
    │   ...
```
```
usage: python feature_gen_foldloop.py -feature_dir ../bindingsite-feature/ -tar_extract_path /var/scratch/test -tar_directory /work/user/test.tar.gz
```

* feature_gen_bionoi.py: basically same as feature_gen.py, except that the input data directory is the bionoi dataset with a different folder structure instead of a single folder containing images:
```
/bionoi/train/control/
/bionoi/train/nucleotide/
/bionoi/val/control/
/bionoi/val/nucleotide/
/bionoi/test/control/
/bionoi/test/nucleotide/
```

* reconstruct.py: given the index of an image, take this image and feed it to a trained autoencoder, then plot the original image and reconstructed image by the autoencoder.
```
usage: python reconstruct.py -index 0 -data_dir ../bindingsite-img/ 
```
* verify_loss.py: given a  new dataset, calculate the loss of the autoencoder on this new dataset. If it is close to the loss on training set, then we are probably good.
```
usage: python verify_loss.py -data_dir ../bindingsite-img/
```
* helper.py: miscllaneous functions.   
