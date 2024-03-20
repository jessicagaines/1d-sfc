# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 11:25:40 2021

@author: Jessica Gaines
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.io 
from scipy.interpolate import interp1d
import pandas as pd

'''
Methods for reading in empirical data
'''

### Read in data from .mat format
def get_combined_data_jh(path,file):
    struct = scipy.io.loadmat(path + file, squeeze_me=True,simplify_cells=True).get(file)
    nsubj = len(struct.get('subj'))    
    taxis = struct.get('taxis')
    ntrials = 0
    for subj in range(nsubj):
        ntrials = ntrials + struct.get('subj')[subj].get('dat')[2].shape[0]
    control_data = np.ndarray([ntrials,len(taxis)])
    curr_trial = 0
    for pert,flip in enumerate([-1,1]):
        for subj in range(nsubj):
            subj_data = flip * struct.get('subj')[subj].get('dat')[pert]
            control_data[curr_trial:curr_trial + len(subj_data),:] = subj_data
            curr_trial = curr_trial + len(subj_data)
    means = np.nanmean(control_data, axis=0)
    stds = np.nanstd(control_data,axis=0)
    stde = stds/np.sqrt(control_data.shape[0])
    if np.isinf(means[0]) or np.isnan(means[0]): means[0] = 0
    for i in range(1,len(means)):
        if np.isinf(means[i]) or np.isnan(means[i]):
            means[i] = means[i-1]
    return downsample(taxis,300), downsample(means,300), downsample(stds,300), downsample(stde,300)

###Downsample data to match simulator output 
def downsample(array, npts):
    interpolated = interp1d(np.arange(len(array)), array, axis = 0, fill_value = 'extrapolate')
    downsampled = interpolated(np.linspace(0, len(array), npts))
    return downsampled

'''
Main method for reading in data
Determine data format, read in, downsample, and return empirical data 
Organize data into standard structure
'''   
def read_obs(path,condition,read_type,subj_type,color):
    observation = dict()
    if subj_type == "control":
        observation['name'] = "Controls"
    else:
        name = condition + " " + subj_type +'s'
        observation['name'] = name
    if read_type == 'jh':
        taxis, data_means, data_stdv, data_stde = get_combined_data_jh(path,subj_type + '_comb_subj_pertresp')
        pert_dur = 0.4
    # add other read types here
    observation['data'] = data_means
    observation['stdv'] = data_stdv
    observation['stde'] = data_stde
    observation['taxis'] = taxis
    observation['color'] = color
    observation['pert_dur'] = pert_dur
    return observation

    