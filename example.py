# -*- coding: utf-8 -*-
"""
Created on Mon June 8 11:18:24 2019 by Attila Lengyel - a.lengyel@tudelft.nl

Generates neuron features for Conv3-3 layer of VGG16 using the CIFAR100 dataset.

"""

import torch
import torchvision

from neuron_feature import get_neuron_features

# Load pretrained model
model = torchvision.models.vgg16(pretrained=True)
# Remove layers up to desired output layer
model = torch.nn.Sequential(*model.features[:16])

# Define dataset transform
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
tr = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                     torchvision.transforms.Normalize(mean,std)])

# Get dataset
dataset = torchvision.datasets.CIFAR100('./', train=False, transform=tr, download=True)

# Generate neuron features
get_neuron_features(model, dataset, batch_size=128, top_n=50,
                    out_dir='./output', mean=mean, std=std)