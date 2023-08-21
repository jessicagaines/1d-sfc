import matplotlib.pyplot as plt
import numpy as np
import torch
from sbi import analysis as analysis
import pandas as pd
import seaborn as sns
import configparser
from model import Model
from scipy.stats import ks_2samp
from scipy.stats import ttest_ind
from scipy.stats import gaussian_kde
import copy
import os
    
def find_idx_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def violin_plots(obs_list,samples,prior_min,prior_max,effect_size_list=None,effect_size_stderr=None,labels=None,conf_int_level=None,show=True):
    fig, ax = plt.subplots(1,len(prior_min),figsize=(15*len(prior_min)/5,4))
    for i in range(len(ax)):
        data_all = pd.DataFrame()
        color_palette = {}
        conf_int = np.ndarray((len(obs_list),3))
        for j,obs in enumerate(obs_list):
            data_pop = pd.DataFrame()
            param_data = np.sort(samples[:,i,j])
            if conf_int_level:
                kde = gaussian_kde(param_data)
                pdf = kde.evaluate(param_data)
                cdf = np.cumsum(pdf)
                cdf = cdf/max(cdf)
                conf_int[j,0] = param_data[find_idx_nearest(cdf, (1-conf_int_level)/2)]
                conf_int[j,1] = param_data[find_idx_nearest(cdf, 0.5)]
                conf_int[j,2] = param_data[find_idx_nearest(cdf, 1-((1-conf_int_level)/2))]
            data_pop['data'] = param_data
            color_palette[obs.get('name')] = obs.get('color')
            data_pop['Population'] = obs.get('name')
            data_all = pd.concat([data_all,data_pop])
        sns.violinplot(x='Population',y='data',palette=color_palette,data=data_all, ax=ax[i],orient='v',inner=None)
        if effect_size_stderr is not None:
            ax[i].annotate("{:.2f}".format(effect_size_list[i]) + ' +/- ' + "{:.2f}".format(effect_size_stderr[i]),(0.17,0.90),xycoords='axes fraction',size=16)
        elif effect_size_list is not None:
            ax[i].annotate(str(round(effect_size_list[i],1)),(0.45,0.90),xycoords='axes fraction',size=20)
        if conf_int_level:
            for j,obs in enumerate(obs_list):
                #ax[i].scatter(j,data_list[j].get('max_likelihood')[i],marker='.',color='black',s=500)
                ax[i].scatter(j,conf_int[j,0],marker='_',color='black',s=300)
                ax[i].scatter(j,conf_int[j,1],marker='.',color='black',s=500)
                ax[i].scatter(j,conf_int[j,2],marker='_',color='black',s=300)
                ax[i].plot([j,j],[conf_int[j,0],conf_int[j,2]],color='black')
        ax[i].set_ylabel('')
        if labels: ax[i].set_title(labels[i],fontsize=15)
        buffer = (prior_max[i] - prior_min[i]) * 0.1
        ax[i].tick_params(axis='y',which='major',labelsize=15)
        ax[i].tick_params(axis='x',which='major',labelsize=15)
        ax[i].set_xlabel('')
        ax[i].set_ylim([prior_min[i]-buffer, prior_max[i]+buffer])
        ax[i].yaxis.offsetText.set_fontsize(15)
    ax[0].set_ylabel('Parameter value',fontsize=20)
    plt.tight_layout()
    return fig

def glassdelta_effect_size(samples1,samples2):
    mean1 = np.mean(samples1,axis=0)
    mean2 = np.mean(samples2,axis=0)
    var1 = np.var(samples1,axis=0)
    var2 = np.var(samples2,axis=0)
    glassdelta = np.abs((mean1-mean2)/np.sqrt(var1))
    return glassdelta
    
def rmse(array1,array2):
    return np.sqrt(rss(array1,array2)/len(array1))
    
def rss(array1,array2):
    return np.sum((array1 - array2)**2)
    
def marked_dist_plot(samples, prior_min, prior_max, labels=None, means=None, medians=None, modes=None, max_likelihood=None,actual_params=None,name=''):
    fig,ax = analysis.pairplot(samples, limits=np.array(list(zip(prior_min,prior_max))), figsize=(24,6))
    fontsize = 24
    for i in range(ax.shape[0]):
        if labels: ax[i,i].set_xlabel(labels[i],fontsize=fontsize)
        ax[i,i].tick_params(axis='both',which='major',labelsize=20)
        if actual_params is not None:
            ax[i,i].axvline(actual_params[i],color='r',linestyle=':')
        if means is not None:
            ax[i,i].axvline(means[i].item(),color='b')
        if medians is not None:
            ax[i,i].axvline(medians[i].item(),color='c')
        if modes is not None:
            ax[i,i].axvline(modes[i].item(),color='g')
        if max_likelihood is not None:
            ax[i,i].axvline(max_likelihood[i].item(),color='orange')
    return fig

def plot_actual_data(obs_list,ax=None,xlabel=False,ylabel=False,ylim=[-5,45],legend=False,show_pert=False,alpha=0.5):
    if ax is None:
        fig, ax = plt.subplots()
    if show_pert:
        ax.axvspan(0,0.4,alpha=0.08,color='gray')
        ax.annotate('Perturbation',(0.01,40),fontsize=14,color='gray')
    for i, obs in enumerate(obs_list):
        ax.plot(obs.get('taxis'),obs.get('data'),label=obs.get('name'),linewidth=5,linestyle=':',color=obs.get('color'),alpha=alpha)
        ax.plot(obs.get('taxis'),obs.get('data')+obs.get('stde'),linestyle=':',linewidth=2,color=obs.get('color'),alpha=alpha)
        ax.plot(obs.get('taxis'),obs.get('data')-obs.get('stde'),linestyle=':',linewidth=2,color=obs.get('color'),alpha=alpha)
        #plt.fill_between(obs.get('taxis'),
        #             obs.get('data')+obs.get('stde'),
        #             obs.get('data')-obs.get('stde'),
        #             linewidth=5,color='#A3A3A3')
        if xlabel:
            ax.set_xlabel('Time (s)',fontsize=18)
            ax.tick_params(axis='x',which='major',labelsize=16)
        else: ax.set_xticks([])
        if ylabel:
            ax.set_ylabel('Pitch (cents)',fontsize=18)
            ax.tick_params(axis='y',which='major',labelsize=16)
        else: ax.set_yticks([])
        if ylim is not None:
            ax.set_ylim(ylim)
        if legend:
            ax.annotate(obs.get('name'), (0.65, 36+4*i), fontsize=18, color=obs.get('color'))