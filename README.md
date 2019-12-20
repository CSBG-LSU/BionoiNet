# BionoiNet
## Description
BionoiNet is a deep learning-based framework for ligand-binding site classification. It converts 3-d structures of ligand-binding sites into 2-d images, and then these images were processed by a deep neural network (DNN) to generate classification results. The pipeline of BionoiNet is shown below:

![](https://github.com/CSBG-LSU/BionoiNet/blob/master/figures/BionoiNet.PNG)

The 2-d images of ligand-binding sites are generated using the [Bionoi](https://github.com/CSBG-LSU/BionoiNet/tree/master/bionoi) software and the DNN is implemented with the APIs of [Pytorch](https://pytorch.org/).

## Dependency
Install the dependency package using this file: ```dependency/pytorch_env.yml```. This denpendency file is exported by [Anaconda](https://www.anaconda.com/). To install the environment:
```
conda env create -f pytorch_env.yml
```

## Folders
* bionoi: a software that converts ligand-binding sites (.mol2 files) into Voronoi diagrams.
* bionoi_cnn: a convolutional neural network (CNN) trained on the Voronoi representations of ligand-binding sites for classification.
* bionoi_ml: machine learning baselinies to classifiy binding sites.
* homology_reduced_folds: files containing folds such that the sequence identity between train and validation is less than 20%.

## Cite our work
