# BionoiNet
## Description
BionoiNet is a deep learning-based framework for ligand-binding site classification. It transforms 3-d structures of ligand-binding sites into 2-d images, and then these images are processed by a deep neural network (DNN) to generate classification results. The pipeline of BionoiNet is shown below:

![](https://github.com/CSBG-LSU/BionoiNet/blob/master/figures/BionoiNet.PNG)

The 2-d images of ligand-binding sites are generated using the [Bionoi](https://github.com/CSBG-LSU/BionoiNet/tree/master/bionoi) software and the DNN is implemented with the APIs of [Pytorch](https://pytorch.org/). The datasets are divided into 5 folds for cross-validation and the sequence identity between the training and validation sets is less than 20%.

## Dependency
Install the dependency package using this file: ```dependency/environment.yml```. This denpendency file is exported by [Anaconda](https://www.anaconda.com/). To install the environment:
```
conda env create -f environment.yml
```

## Folders
* bionoi: a software that transforms ligand-binding sites (.mol2 files) into Voronoi diagrams.
* bionoi_autoencoder: autoencoder that trained on Bionoi images in an un-supervised manner. The trained autoencoder can be used to produce latent space vectors of Bionoi images for machine learning applications.
* bionoi_cnn_homology_reduced: a convolutional neural network (CNN) trained on the Voronoi representations of ligand-binding sites for classification.
* bionoi_ml_homology_reduced: machine learning baselinies to classifiy binding sites.
* bionoi_roc: code to plot roc curves for classification.
* dependency: dependency python packages.
* homology_reduced_folds: files containing folds such that the sequence identity between train and validation is less than 20%.

## Usage
1. Unzip the .mol2 files located at ```/homology_reduced_folds/mols.zip```.
2. Download this repository and put it where you extract the .mol2 files. The layout of the project should be like following:   
```
    .
    ├── BionoiNet
    └── homology_reduced_mols      
        ├── control
            ├── fold_1   
            ├── fold_2
            ├── ...
            └── fold_5
        ├── heme
            ├── fold_1   
            ├── fold_2
            ├── ...
            └── fold_5
        └── nucleotide
            ├── fold_1   
            ├── fold_2
            ├── ...
            └── fold_5
```
3. Run Bionoi by executing ```/bionoi/img_gen.sh``` to transform .mol2 files to images. 
4. Train the convolutional neural network (CNN) at ```/bionoi_cnn_homology_reduced/``` for cross-validation:    
```
python homology_reduced_cnn_cv_resnet18.py -op control_vs_nucleotide -root_dir ../../bionoi_output/residue_type/ -batch_size 32 -result_file_suffix 1st
```
4. Grab a cup of coffee and wait for results.

## Citation
Shi, Wentao, et al. "BionoiNet: Ligand-binding site classification with off-the-shelf deep neural network." *Bioinformatics* (2020).
