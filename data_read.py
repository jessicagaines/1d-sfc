# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 11:25:40 2021

@author: Jessica
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.io 
from scipy.interpolate import interp1d

path = 'Cerebellar_Data/'
ca_control_file = 'control_comb_subj_pertresp'
ca_patient_file = 'patient_comb_subj_pertresp'

#path = 'lvPPA_Data/'
#lvppa_control_file = 'control'
#lvppa_patient_file = 'patient'

def get_combined_data(path,file):
    struct = scipy.io.loadmat(path + file, squeeze_me=True,simplify_cells=True).get(file)
    subj_data = struct.get('subj')[0].get('dat')[2]
    taxis = struct.get('taxis')
    nsubj = len(struct.get('subj'))        
    max_trials = 0
    trial_ns = np.zeros((1,nsubj))
    for subj in range(nsubj):
        ntrials = struct.get('subj')[subj].get('dat')[2].shape[0]
        trial_ns[0,subj] = ntrials
        if ntrials > max_trials: max_trials = ntrials
    control_data = np.ndarray([max_trials,subj_data.shape[1],nsubj])
    control_data[:,:,:] = np.nan
    for subj in range(nsubj):
        subj_data = struct.get('subj')[subj].get('dat')[2]
        control_data[:subj_data.shape[0],:subj_data.shape[1],subj] = subj_data
    subj_means = np.nanmean(control_data, axis=0)
    means = np.nanmean(subj_means, axis=-1)
    if np.isinf(means[0]): means[0] = 0
    for i in range(1,len(means)):
        if np.isinf(means[i]):
            means[i] = means[i-1]
    return taxis, means
'''    
def get_combined_data_jh(path,file):
    struct = scipy.io.loadmat(path + file, squeeze_me=True,simplify_cells=True).get(file)
    subj_data = struct.get('subj')[0].get('dat')[2]
    taxis = struct.get('taxis')
    nsubj = len(struct.get('subj'))        
    control_data = np.ndarray([1,subj_data.shape[1]])
    for pert,flip in enumerate([-1,1]):
        for subj in range(nsubj):
            subj_data = flip * struct.get('subj')[subj].get('dat')[pert]
            control_data = np.concatenate((control_data,subj_data),axis=0)
    means = np.nanmean(control_data, axis=0)
    stds = np.nanstd(control_data,axis=0)
    stde = stds/np.sqrt(control_data.shape[0])
    if np.isinf(means[0]) or np.isnan(means[0]): means[0] = 0
    for i in range(1,len(means)):
        if np.isinf(means[i]) or np.isnan(means[i]):
            means[i] = means[i-1]
    return taxis, means, stds, stde
'''

def get_combined_data_jh(path,file):
    struct = scipy.io.loadmat(path + file, squeeze_me=True,simplify_cells=True).get(file)
    subj_data = struct.get('subj')[0].get('dat')[2]
    taxis = struct.get('taxis')
    nsubj = len(struct.get('subj'))        
    control_data = np.ndarray([1,subj_data.shape[1]])
    for pert,flip in enumerate([-1,1]):
        for subj in range(nsubj):
            subj_data = flip * struct.get('subj')[subj].get('dat')[pert]
            control_data = np.concatenate((control_data,subj_data),axis=0)
    means = np.nanmean(control_data, axis=0)
    stds = np.nanstd(control_data,axis=0)
    stde = stds/np.sqrt(control_data.shape[0])
    if np.isinf(means[0]) or np.isnan(means[0]): means[0] = 0
    for i in range(1,len(means)):
        if np.isinf(means[i]) or np.isnan(means[i]):
            means[i] = means[i-1]
    return taxis, means, stds, stde
    
    
def get_combined_data_hk(path,file,variable):
    struct = scipy.io.loadmat(path + file, squeeze_me=True,simplify_cells=True).get(variable)
    taxis = struct.get('taxis')
    means = struct.get('mean')
    stds = struct.get('std')
    stde = struct.get('stde')
    return taxis, means, stds, stde

def get_combined_data_kr(path,file):
    taxis = scipy.io.loadmat(path + 'xaxisdata.mat', squeeze_me=True,simplify_cells=True).get('xs')
    if file == "cont":
        means = scipy.io.loadmat(path + file + 'pp.mat', squeeze_me=True,simplify_cells=True).get('cp')
        lower = scipy.io.loadmat(path + file + 'lower.mat', squeeze_me=True,simplify_cells=True).get('lowerc')
    else:
        means = scipy.io.loadmat(path + file + 'pp.mat', squeeze_me=True,simplify_cells=True).get('pp')
        lower = scipy.io.loadmat(path + file + 'lower.mat', squeeze_me=True,simplify_cells=True).get('lowerp')
    stds = np.zeros((len(means),1))
    stde = means - lower
    return taxis, means, stds, stde
    
    
def get_ind_data_hk(path,variable):
    struct = scipy.io.loadmat(path + 'centsdev_dat', squeeze_me=True,simplify_cells=True).get('centsdev_dat')
    taxis = struct.get('frame_taxis')
    nsubj = struct.get(variable).get('nsubj')
    means = np.ndarray([nsubj,len(taxis)])
    stdvs = np.ndarray([nsubj,len(taxis)])
    sterrs = np.ndarray([nsubj,len(taxis)])
    for i in range(nsubj):
        subj_mean = np.mean(struct.get(variable).get('subj')[i].get('absdat')[2],0)
        ntrials = struct.get(variable).get('subj')[i].get('absdat')[2].shape[0]
        subj_std = np.std(struct.get(variable).get('subj')[i].get('absdat')[2],0)
        subj_stde = subj_std/ntrials
        means[i,:] = subj_mean
        stdvs[i,:] = subj_std
        sterrs[i,:] = subj_stde
    return taxis, means, stdvs, sterrs
    
def downsample(array, npts):
    interpolated = interp1d(np.arange(len(array)), array, axis = 0, fill_value = 'extrapolate')
    downsampled = interpolated(np.linspace(0, len(array), npts))
    return downsampled
    
def read_obs(path,condition,read_type,subj_type,color):
    observation = dict()
    if subj_type == "control":
        observation['name'] = "Controls"
    else:
        name = condition + " " + subj_type +'s'
        observation['name'] = name
    if read_type == 'jh':
            taxis, data_means, data_stdv, data_stde = get_combined_data_jh(path,subj_type + '_comb_subj_pertresp')
    elif read_type == 'hk':
        taxis, data_means, data_stdv, data_stde = get_combined_data_hk(path,subj_type + '_avgs.mat',subj_type + '_absperttrial')
    elif read_type == 'kr':
        taxis, data_means, data_stdv, data_stde = get_combined_data_kr(path,condition.lower())
    observation['data'] = downsample(data_means,300)
    observation['stdv'] = downsample(data_stdv,300)
    observation['stde'] = downsample(data_stde,300)
    observation['taxis'] = downsample(taxis,300)
    observation['color'] = color
    return observation
    
def read_indiv_obs(path,condition,read_type,subj_type,color_list=None):
    observation_list = []
    if read_type == 'jh':
            pass
    elif read_type == 'hk':
        taxis, means, stdvs, sterrs = get_ind_data_hk(path,subj_type)
    for i in range(means.shape[0]):
        observation = dict()
        name = condition + "_" + subj_type + "_" + str(i)
        observation['name'] = name
        observation['data'] = downsample(means[i,:],300)
        observation['stdv'] = downsample(stdvs[i,:],300)
        observation['stde'] = downsample(sterrs[i,:],300)
        observation['taxis'] = downsample(taxis,300)
        if color_list:
            observation['color'] = color_list[i]
        else : observation['color'] = 'red'
        observation_list.append(observation)
    return observation_list
    