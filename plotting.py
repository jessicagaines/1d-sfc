# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 12:12:45 2021

@author: Jessica
"""
import matplotlib.pyplot as plt
import numpy as np
import os

'''
Functions for plotting model output
'''
def plot_trial_timecourse(t_axis,pitch_output,scale='hertz',baseline=np.array([]),ax=None,color='blue',title=None,ylim=None):
    if ax is None:
        fig,ax = plt.subplots(1,1)
    if title is None:
        ax.set_title('Pitch Output Response',fontsize=18)
    else: ax.set_title(title,fontsize=18)
    ax.set_xlabel('Time (s)',fontsize=18)
    ax.set_ylabel('Pitch (' + scale + ')',fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    if scale == 'cents':
        if baseline.size == 0: raise NameError('No baseline provided')
        pitch_output = get_cents(pitch_output,baseline)
    ax.plot(t_axis,pitch_output,color=color)
    if ylim is not None:
        ax.set_ylim(ylim)
    #if not os.path.exists('figs/'): os.makedirs('figs/')
    #plt.savefig('figs/trial_timecourse.png')
    #plt.show()
    
def get_cents(hertz,baseline):
    return 1200*np.log2(np.divide(hertz,baseline))