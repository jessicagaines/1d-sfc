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
import copy
import os
    
def violin_plots(data_list,sig_list,effect_list,prior_min,prior_max,labels=None, name=None):
    fig, ax = plt.subplots(1,len(prior_min),figsize=(15,4))
    for i in range(len(ax)):
        data_all = pd.DataFrame()
        color_palette = {}
        conf_int = np.ndarray((len(data_list),3))
        for j,data in enumerate(data_list):
            data_pop = pd.DataFrame()
            data_pop['data'] = data.get('samples').numpy()[:,i]
            color_palette[data.get('name')] = data.get('color')
            data_pop['Population'] = data.get('name')
            data_all = pd.concat([data_all,data_pop])
            samples = np.sort(data['samples'].numpy(),axis=0)
            max_log_prob = data.get('max_likelihood')
        sns.violinplot(x='Population',y='data',palette=color_palette,data=data_all, ax=ax[i],orient='v',inner=None)
        facecolor = 1-(effect_list[i]/max(effect_list))
        ax[i].set_facecolor([facecolor]*3)
        if facecolor < 0.5 : textcolor = 'white'
        else: textcolor = 'black'
        if sig_list[i] > 0:
            ax[i].annotate('*',(0.5,0.9),xycoords='axes fraction',size=30,color=textcolor)
        ax[i].annotate(str(round(effect_list[i],1)),(0.45,0.05),xycoords='axes fraction',size=20,color=textcolor)
        for j,data in enumerate(data_list):
            ax[i].scatter(j,data_list[j].get('max_likelihood')[i],marker='_',color='black',s=300)
        ax[i].set_ylabel('')
        if labels: ax[i].set_xlabel(labels[i],fontsize=15)
        buffer = (prior_max[i] - prior_min[i]) * 0.1
        ax[i].set_xticklabels('')
        ax[i].tick_params(axis='y',which='major',labelsize=15)
        ax[i].yaxis.offsetText.set_fontsize(15)
    plt.tight_layout()
    return ax

def welch_ttest(data_list,fwer,nparams):
    results = [0]*nparams
    for param in range(nparams):
        dist1 = data_list[0].get('samples')[:,param].numpy()
        dist2 = data_list[1].get('samples')[:,param].numpy()
        stat,pval = ttest_ind(dist1,dist2,equal_var=False)
        if pval < fwer/nparams: results[param] = 1
    return results

def cohensd_effect_size(data_list,nparams,nsamples):
    results = [0]*nparams
    for param in range(nparams):
        dist1 = data_list[0].get('samples')[:,param].numpy()
        dist2 = data_list[1].get('samples')[:,param].numpy()
        mean1 = np.mean(dist1)
        mean2 = np.mean(dist2)
        var1 = np.var(dist1)
        var2 = np.var(dist2)
        pooled_s = np.sqrt((nsamples-1)*(var1+var2)/(nsamples*2-2))
        cohens_d = np.abs((mean1-mean2)/pooled_s)
        results[param] = cohens_d
    return results

def glassdelta_effect_size(data_list,nparams,nsamples):
    results = [0]*nparams
    for param in range(nparams):
        dist1 = data_list[0].get('samples')[:,param].numpy()
        dist2 = data_list[1].get('samples')[:,param].numpy()
        mean1 = np.mean(dist1)
        mean2 = np.mean(dist2)
        var1 = np.var(dist1)
        var2 = np.var(dist2)
        glassdelta = np.abs((mean1-mean2)/np.sqrt(var1))
        results[param] = glassdelta
    return results
    
def mse(array1,array2):
    return ((array1 - array2)**2).mean()
    
def marked_dist_plot(samples, prior_min, prior_max, labels=None, means=None, modes=None, max_likelihood=None,actual_params=None,name=''):
    fig,ax = analysis.pairplot(samples, limits=np.array(list(zip(prior_min,prior_max))), figsize=(24,6))
    fontsize = 24
    for i in range(ax.shape[0]):
        if labels: ax[i,i].set_xlabel(labels[i],fontsize=fontsize)
        ax[i,i].tick_params(axis='both',which='major',labelsize=20)
        if actual_params is not None:
            ax[i,i].axvline(actual_params[i],color='r',linestyle=':')
        if means is not None:
            ax[i,i].axvline(means[i].item(),color='b')
        if modes is not None:
            ax[i,i].axvline(modes[i].item(),color='g')
        if max_likelihood is not None:
            ax[i,i].axvline(max_likelihood[i].item(),color='orange')
    if not os.path.exists('figs/'): os.makedirs('figs/')
    plt.savefig('figs/param_dists_' + name + '.png',format='png')
    plt.show()
