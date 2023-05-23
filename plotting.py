# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 12:12:45 2021

@author: Jessica
"""
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_trial_timecourse(t_axis,pitch_output,scale='hertz',baseline=np.array([])):
    plt.figure()
    if scale == 'cents':
        if baseline.size == 0: raise NameError('No baseline provided')
        pitch_output = get_cents(pitch_output,baseline)
    plt.plot(t_axis,pitch_output)
    plt.title('Pitch Output Response')
    plt.xlabel('Time (s)')
    plt.ylabel('Pitch (' + scale + ')')
    if not os.path.exists('figs/'): os.makedirs('figs/')
    plt.savefig('figs/trial_timecourse.png')
    plt.show()
    
def plot_adaptation(pitch_output,scale='hertz',baseline=np.array([]),hz_pert=-6.735,startframe=0,endframe=25):
    plt.figure()
    trial_means = np.mean(pitch_output[startframe:endframe],axis=0)
    if scale == 'cents':
        pitch_output = get_cents(pitch_output[startframe:endframe], baseline)
        trial_means = np.mean(pitch_output,axis=0)
        plt.ylim([-50,50])
    if scale == 'percent':
        pitch_output = -100 * (pitch_output - baseline)/hz_pert
        trial_means = np.mean(pitch_output[startframe:endframe],axis=0)
    plt.scatter(range(len(trial_means)),trial_means)
    plt.title('Adaptation')
    plt.xlabel('Trial')
    plt.ylabel('Pitch (' + scale + ')')
    if not os.path.exists('figs/'): os.makedirs('figs/')
    plt.savefig('figs/adaptation.png')
    plt.show()
    
def get_cents(hertz,baseline):
    return 1200*np.log2(np.divide(hertz,baseline))