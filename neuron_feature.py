# -*- coding: utf-8 -*-
"""
Created on Mon May 18 11:04:37 2019 by Attila Lengyel - a.lengyel@tudelft.nl

"""

import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from receptive_field import receptive_field, receptive_field_for_unit

import matplotlib.pyplot as plt

def get_neuron_features(model, dataset, batch_size=128, top_n=100, 
                        out_dir='./output', mean=[0,0,0], std=[0,0,0]):
    """
    Generates neuron features for given model definition using given dataset.

    :param model: Pytorch model definition.
    :param dataset: Dataset used for generating NFs.
    :param batch_size: Batch size used for predicting feature maps.
    :top_n: Use top_n input patch activations for generating NF.
    :out_dir: Directory where generated images are stored.
    :mean: Dataset mean used for normalization in transform function.
    :std: Dataset std used for normalization in transform function.
    
    :return: returns nothing
    """

    # make output directory of not exists
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # set model in eval mode
    if torch.cuda.is_available():
        model = model.cuda()

    # Set model in eval mode
    model.eval()

    # Dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    mean = np.asarray(mean)
    std = np.asarray(std)

    # input shape (c,w,h)
    in_shape = next(iter(dataloader))[0].shape[1:]
    # receptive field info for entire model
    receptive_field_dict = receptive_field(model, in_shape)
    # output layer info
    output_layer_info = receptive_field_dict[str(list(receptive_field_dict.keys())[-2])]
    # check if fm has group convs
    fm_groups = output_layer_info['output_shape'][2] if len(output_layer_info['output_shape']) == 5 else 0
    # number of filters in last fm
    n_filters = output_layer_info['output_shape'][1]
    
    # Create placeholder for input patches
    rf = int(output_layer_info['r'])
    
    if fm_groups > 0:
        fm_im = np.zeros((top_n,n_filters,fm_groups,rf,rf,3))
        fm_w = -1e5*np.ones((top_n,n_filters,fm_groups))
    else:
        fm_im = np.zeros((top_n,n_filters,rf,rf,3))
        fm_w = -1e5*np.ones((top_n,n_filters))

    # Calculate amount of padding needed for input visualization
    # Get range for rf at position 0,0 in final feature map
    rf_range = receptive_field_for_unit(receptive_field_dict, str(list(receptive_field_dict.keys())[-2]), (0,0))
    pad_y = int(rf-(rf_range[0][1]-rf_range[0][0]))
    pad_x = int(rf-(rf_range[1][1]-rf_range[1][0]))
    
    
    # Print summary
    print('Group Convolutions: \t {}, {} elements'.format(fm_groups>0, fm_groups))
    print('Number of filters: \t {}'.format(n_filters))
    print('Receptive field size: \t {}'.format(rf))
    print('RF range at (0,0): \t {}'.format(rf_range))
    print('Input padding (x,y): \t {}, {}'.format(pad_x, pad_y))
    print('==============================================================================')

    # Iterate over all data samples to get input patch for highest neuron activation
    # for each filter and transformation
    with torch.no_grad():
        for inputs, _ in tqdm(dataloader, total=len(dataloader), desc='Extracting input patches: '):

            if torch.cuda.is_available():
                model_inputs = inputs.cuda()
            else:
                model_inputs = inputs

            # Predict feature map
            fm = model(model_inputs)

            # Convert inputs to numpy w,h,c for visualization
            inputs = inputs.permute((0,2,3,1)).numpy()
            # Unnormalize
            inputs *= std[None,None,None,:]
            inputs += mean[None,None,None,:]
            inputs = np.clip(inputs,0,1)
            # Pad inputs for visualization to compensate for padding in layers
            inputs = np.pad(inputs, ((0,0),(pad_y,pad_y),(pad_x,pad_x),(0,0)), mode='constant')

            # get batch shape
            fm_shape = fm.shape
            # if gconv: reshape groups into channels
            if fm_groups > 0:
                fm = fm.view((fm_shape[0],-1,fm_shape[3],fm_shape[4]))

            # Get max values and locations of feature maps
            # pool size = fm size = fm_shape[-1]
            a, b = F.max_pool2d(fm, (fm_shape[-2],fm_shape[-1]), return_indices=True)

            # if gconv: reshape groups back to own dimension
            if fm_groups > 0:
                a = a.view((fm_shape[0],fm_shape[1],fm_shape[2]))
                b = b.view((fm_shape[0],fm_shape[1],fm_shape[2]))

            a = a.cpu().numpy()
            b = b.cpu().numpy()

            # coordinates of max activations
            x = b % fm.shape[-1]
            y = b // fm.shape[-1]

            # store weight and input patches for each max position
            for i in range(inputs.shape[0]):
                for j in range(n_filters):

                    if fm_groups == 0:
                        # check if weight is higher than current lowest weight
                        if a[i,j] > np.min(fm_w[:,j]):
                            # replace lowest weight by current weight
                            m = np.argmin(fm_w[:,j])
                            fm_w[m,j] = a[i,j]
                            # store input patch
                            rf_range = receptive_field_for_unit(receptive_field_dict, str(list(receptive_field_dict.keys())[-2]), (y[i,j],x[i,j]), bound=False)
                            fm_im[m,j,:,:,:] = inputs[i,rf_range[0][0]+pad_y:rf_range[0][1]+pad_y,rf_range[1][0]+pad_x:rf_range[1][1]+pad_x,:]

                    else:
                        # loop over extra dimension for gconv
                        for k in range(fm_groups):
                            # check if weight is higher than current lowest weight
                            if a[i,j,k] > np.min(fm_w[:,j,k]):
                                # replace lowest weight by current weight
                                m = np.argmin(fm_w[:,j,k])
                                # store weight
                                fm_w[m,j,k] = a[i,j,k]
                                # store input patch
                                rf_range = receptive_field_for_unit(receptive_field_dict, str(list(receptive_field_dict.keys())[-2]), (y[i,j,k],x[i,j,k]), bound=False)
                                fm_im[m,j,k,:,:,:] = inputs[i,rf_range[0][0]+pad_y:rf_range[0][1]+pad_y,rf_range[1][0]+pad_x:rf_range[1][1]+pad_x,:]

    # Calculate and save neuron feature for each filter and transformation
    for i in tqdm(range(n_filters), total=n_filters, desc='Generating neuron features: '):
        if fm_groups == 0:
            w_sum = np.sum(fm_w[:,i])
            if w_sum > 0:
                # Sort patches in order of highest neuron activations
                idx = np.argsort(fm_w[:,i])[::-1] # ::-1 for high to low sort
                fm_w[:,i] = fm_w[idx,i]
                fm_im[:,i,:,:,:] = fm_im[idx,i,:,:,:]

                # Calculate neuron feature
                fm_nfw = fm_w[:,i,None,None,None]/w_sum
                nf = np.sum(fm_im[:,i,:,:,:]*fm_nfw, axis=0)

                # Plot 19 highest activated patches
                plt.figure(figsize=(40,2))
                for j in range(19):
                    plt.subplot(1,20,j+2)
                    plt.title('{:.3f}'.format(fm_w[j,i]))
                    plt.imshow(fm_im[j,i,:,:,:])
                # Plot NF
                plt.subplot(1,20,1)
                plt.imshow(nf)
                plt.title('NF')
                _=plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
                plt.savefig(os.path.join(out_dir,'f_{:02d}.png'.format(i)), bbox_inches='tight')
                plt.savefig(os.path.join(out_dir,'f_{:02d}.pdf'.format(i)), bbox_inches='tight')
                plt.close()

        else:
            plt.figure(figsize=(40,6))
            for k in range(fm_groups):
                w_sum = np.sum(fm_w[:,i,k])
                if w_sum > 0:
                    # Sort patches in order of highest neuron activations
                    idx = np.argsort(fm_w[:,i,k])[::-1] # ::-1 for high to low sort
                    fm_w[:,i,k] = fm_w[idx,i,k]
                    fm_im[:,i,k,:,:,:] = fm_im[idx,i,k,:,:,:]

                    # Calculate neuron feature
                    fm_nfw = fm_w[:,i,k,None,None,None]/w_sum
                    nf = np.sum(fm_im[:,i,k,:,:,:]*fm_nfw, axis=0)

                    # Plot 19 highest activated patches
                    for j in range(19):
                        plt.subplot(fm_groups,20,j+2+20*k)
                        plt.title('{:.3f}'.format(fm_w[j,i,k]))
                        plt.imshow(fm_im[j,i,k,:,:,:])
                    # Plot NF
                    plt.subplot(fm_groups,20,1+20*k)
                    plt.imshow(nf)
                    plt.title('NF')
            _=plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
            plt.savefig(os.path.join(out_dir,'f_{:02d}.png'.format(i)), bbox_inches='tight')
            plt.savefig(os.path.join(out_dir,'f_{:02d}.pdf'.format(i)), bbox_inches='tight')
            plt.close()
    print('Done!')