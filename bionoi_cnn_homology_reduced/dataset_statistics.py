"""
Compute the mean and variance of the entire dataset for scaling before feeding data to CNN.
"""
import homology_reduced_cnn_cv_resnet18
from homology_reduced_cnn_cv_resnet18 import BionoiDatasetCV
import torch
from torchvision import transforms

if __name__ == "__main__":
    colorbys = ['atom_type', 'binding_prob', 'blended','center_dist','charge', 'hydrophobicity','residue_type','sasa','seq_entropy']
    ops = ['control_vs_nucleotide', 'control_vs_heme', 'heme_vs_nucleotide']        
    root_dir = '../../bionoi_output/'
    folds=[1,2,3,4,5]
    batch_size = 32
    transform = transforms.Compose([transforms.ToTensor()])
    for colorby in colorbys:
        data_dir = root_dir + colorby + '/'
        for op in ops:        
            dataset = BionoiDatasetCV(op=op, root_dir=data_dir, folds=folds, transform=transform)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=16)
            mean = 0
            std = 0
            for images, _ in dataloader:
                batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
                images = images.view(batch_samples, images.size(1), -1)
                mean += images.mean(2).sum(0)
                std += images.std(2).sum(0)

            mean /= len(dataloader.dataset)
            std /= len(dataloader.dataset)
            print('------------------------------------------------')
            print('colorby: {}, op: {}'.format(colorby, op))
            print('mean:')
            print(mean)
            print('std:')
            print(std)