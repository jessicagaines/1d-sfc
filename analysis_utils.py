import matplotlib.pyplot as plt
import numpy as np
import torch
from sbi import analysis as analysis
from sbi import utils as utils
from sbi.inference.base import infer
import pandas as pd
import seaborn as sns
import configparser
from model import Model
from scipy.stats import gaussian_kde
import copy
import os
import pickle
    
def find_idx_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def violin_plots(obs_list,samples,prior_min,prior_max,effect_size_list=None,effect_size_stderr=None,labels=None,conf_int_level=None,show=True):
    fig, ax = plt.subplots(1,len(prior_min),figsize=(20*len(prior_min)/5,4))
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
            ax[i].annotate("{:.2f}".format(effect_size_list[i]) + ' +/- ' + "{:.2f}".format(effect_size_stderr[i]),(0.17,0.90),xycoords='axes fraction',size=18)
        elif effect_size_list is not None:
            ax[i].annotate(str(round(effect_size_list[i],1)),(0.45,0.90),xycoords='axes fraction',size=18)
        if conf_int_level:
            for j,obs in enumerate(obs_list):
                #ax[i].scatter(j,data_list[j].get('max_likelihood')[i],marker='.',color='black',s=500)
                ax[i].scatter(j,conf_int[j,0],marker='_',color='black',s=400)
                ax[i].scatter(j,conf_int[j,1],marker='.',color='black',s=500)
                ax[i].scatter(j,conf_int[j,2],marker='_',color='black',s=400)
                ax[i].plot([j,j],[conf_int[j,0],conf_int[j,2]],color='black',linewidth=2)
        ax[i].set_ylabel('')
        if labels: ax[i].set_title(labels[i],fontsize=20)
        buffer = (prior_max[i] - prior_min[i]) * 0.1
        ax[i].tick_params(axis='y',which='major',labelsize=17)
        ax[i].tick_params(axis='x',which='major',labelsize=17)
        ax[i].set_xlabel('')
        ax[i].set_ylim([prior_min[i]-buffer, prior_max[i]+2*buffer])
        ax[i].yaxis.offsetText.set_fontsize(15)
    if effect_size_list is not None:
        plt.subplots_adjust(wspace=0.3)
        sort_indeces = np.argsort(-effect_size_list)
        positions = []
        for x in range(len(labels)):
            positions.append(ax[x].get_position())
        for x in range(len(labels)):
            ax[sort_indeces[x]].set_position(positions[x])
    else:
        plt.tight_layout()
    ax[sort_indeces[0]].set_ylabel('Parameter value',fontsize=18)
    return fig

def glassdelta_effect_size(samples1,samples2):
    mean1 = np.mean(samples1,axis=0)
    mean2 = np.mean(samples2,axis=0)
    var1 = np.var(samples1,axis=0)
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

def build_simulator(training_noise_scale, ablate_values=None, ablate_index=None):
    config = configparser.ConfigParser()
    config.read('pitch_pert_configs.ini')
    default_model = Model(config)
    return lambda parameter_set: simulator_configurable(parameter_set, default_model, training_noise_scale, ablate_values, ablate_index)

def simulator_configurable(parameter_set, model, training_noise_scale, ablate_values, ablate_index):
    if ablate_values is not None and ablate_index is not None:
        parameter_set_new = np.insert(parameter_set,ablate_index,ablate_values[ablate_index])
    else: parameter_set_new = parameter_set
    model.set_tunable_params(parameter_set_new)
    y_output,errors = model.run()
    pitch_output = y_output[:,0,0]
    pitch_output[pitch_output < 50] = 50
    pitch_baseline = np.mean(pitch_output[0:50])
    pitch_output_cents = 1200 * np.log2(pitch_output/pitch_baseline)
    pitch_output_cents_wnoise = pitch_output_cents + ((np.random.rand(len(pitch_output_cents)) - 0.5) * training_noise_scale)
    return pitch_output_cents_wnoise

def sbi_train(path,subdir,simulator,prior_min,prior_max,seed,n_simulations):
    prior = utils.torchutils.BoxUniform(low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max))
    np.random.seed(seed)
    torch.manual_seed(seed)
    posterior = infer(simulator,prior,method='SNPE', num_simulations=n_simulations, num_workers=4)
    posterior_save = {'prior_min':prior_min,'prior_max':prior_max,'posterior':posterior,'seed':seed}
    if not os.path.exists(os.path.join(path,subdir,'posterior')): 
        os.mkdir(os.path.join(path,subdir,'posterior'))
    with open(os.path.join(path,subdir,'posterior','seed' + str(seed) + '.pkl'), "wb") as handle:
        pickle.dump(posterior_save,handle)
    
def sample_posterior(path,subdir,simulator,obs_list,seed,n_samples,labels,plot=True):
    np.random.seed(seed)
    torch.manual_seed(seed)
    with open(os.path.join(path,subdir,'posterior','seed' + str(seed) + '.pkl'), "rb") as handle:
        posterior_dict = pickle.load(handle)
    posterior_new = posterior_dict.get('posterior')
    prior_min = posterior_dict.get('prior_min')
    prior_max = posterior_dict.get('prior_max')
    all_samples = np.ndarray([n_samples,len(labels),len(obs_list)])
    for i, observation in enumerate(obs_list):
        samples = posterior_new.sample((n_samples,), x=observation.get('data'))
        all_samples[:,:,i] = samples.numpy()
        log_probability = posterior_new.log_prob(samples, x=observation.get('data'))
        max_joint_likelihood = samples[torch.argmax(log_probability)]
        medians = torch.median(samples,axis=0).values
        if plot: 
            fig = marked_dist_plot(samples, prior_min, prior_max, labels, max_likelihood=max_joint_likelihood, medians=medians, name=observation.get('name'))
            if not os.path.exists(os.path.join(path,subdir,'marginal_dist')): os.mkdir(os.path.join(path,subdir,'marginal_dist'))
            fig.savefig(os.path.join(path,subdir,'marginal_dist', 'marginal_dist_' + subdir + "_" + observation.get('name') + '_seed' + str(seed) + '.png'),format='png')
    if plot:
        effect_size = glassdelta_effect_size(all_samples[:,:,0],all_samples[:,:,1])
        fig = violin_plots(obs_list,all_samples,prior_min,prior_max,effect_size_list=effect_size,labels=labels,show=False)
        if not os.path.exists(os.path.join(path,subdir,'violin_plots')): os.mkdir(os.path.join(path,subdir,'violin_plots'))
        fig.savefig(os.path.join(path,subdir,'violin_plots', 'violin_plots_' + subdir + '_seed' + str(seed) + '.png'),format='png')
        if not os.path.exists(os.path.join(path,subdir,'pitch_plots')): os.mkdir(os.path.join(path,subdir,'pitch_plots'))
        fig, rmse_mean, rmse_sterr = plot_actual_inferred_data(simulator,obs_list,np.median(all_samples,axis=0),xlabel=True,ylabel=True,legend=True)
        fig.savefig(os.path.join(path,subdir,'pitch_plots', 'pitch_plots_' + subdir + '_seed' + str(seed) + '.png'),format='png')
    return all_samples

def plot_actual_data(obs_list,ax=None,xlabel=False,ylabel=False,ylim=None,legend=False,show_pert=False):
    if ax is None:
        fig, ax = plt.subplots()
    if show_pert:
        ax.axvspan(0,obs_list[0].get('pert_dur'),alpha=0.08,color='#C8C8C8')
        ax.annotate('Perturbation',(0.01,40),fontsize=14,color='#646464')
    for i, obs in enumerate(obs_list):
        ax.plot(obs.get('taxis'),obs.get('data'),label=obs.get('name'),linewidth=5,linestyle=':',color=obs.get('color'))
        ax.plot(obs.get('taxis'),obs.get('data')+obs.get('stde'),linestyle=':',linewidth=2,color=obs.get('color'))
        ax.plot(obs.get('taxis'),obs.get('data')-obs.get('stde'),linestyle=':',linewidth=2,color=obs.get('color'))
        #plt.fill_between(obs.get('taxis'),
        #             obs.get('data')+obs.get('stde'),
        #             obs.get('data')-obs.get('stde'),
        #             linewidth=5,color='#A3A3A3')
        if xlabel:
            ax.set_xlabel('Time (s)',fontsize=18)
            ax.tick_params(axis='x',which='major',labelsize=17)
        else: ax.set_xticks([])
        if ylabel:
            ax.set_ylabel('Pitch (cents)',fontsize=18)
            ax.tick_params(axis='y',which='major',labelsize=17)
        else: ax.set_yticks([])
        if ylim is not None:
            ax.set_ylim(ylim)
        if legend:
            ax.annotate(obs.get('name'), (0.65, 36+4*i), fontsize=18, color=obs.get('color'))

def plot_actual_inferred_data(simulator, obs_list,max_likelihood_params,name=None,xlabel=False,ylabel=False,ylim=None,legend=True,title='',figlabel=''):
    ntrials = 100
    rmse_all = np.ndarray((len(obs_list), ntrials))
    fig, ax = plt.subplots(figsize=(8,3.7))
    ax.text(-0.6,18,title,ha='center',va='center',rotation=90,size=20,fontweight='bold')
    ax.text(-0.8,40,figlabel,ha='center',va='center',size=45)
    plot_actual_data(obs_list,ax,xlabel,ylabel,ylim,legend=True)
    for i, obs in enumerate(obs_list):
        plot_color = obs.get('color')
        taxis = obs_list[0].get('taxis')
        inferred_all = np.ndarray((ntrials,len(taxis)))
        for j in range(ntrials):
            sim_output = simulator(max_likelihood_params[:,i])
            inferred_all[j,:] = sim_output
            error = rmse(obs.get('data'), sim_output)
            rmse_all[i,j] = error
        inferred = np.mean(inferred_all,axis=0)
        sterr = np.std(inferred_all,axis=0) / np.sqrt(ntrials)
        ax.plot(taxis,inferred,label=obs.get('name')+' inferred',linewidth=5,color=plot_color)
        ax.plot(taxis,inferred-sterr, linewidth=2,color=plot_color)
        ax.plot(taxis,inferred+sterr, linewidth=2, color=plot_color)
    rmse_mean = np.mean(rmse_all,axis=1)
    rmse_sterr = np.std(rmse_all,axis=1) / np.sqrt(ntrials)
    return fig, rmse_mean, rmse_sterr

def run_sbi(path,subdir,obs_list,n_simulations,n_samples,n_reps,prior_min_all,prior_max_all,all_labels,train=True,ablate_index=None,ablate_values=None,verbose=True):
    if not os.path.exists(os.path.join(path,subdir)): os.mkdir(os.path.join(path,subdir))
    labels = copy.deepcopy(all_labels)
    prior_min = copy.deepcopy(prior_min_all)
    prior_max = copy.deepcopy(prior_max_all)
    if ablate_index is not None:
        ablated_label = labels.pop(ablate_index)
        prior_min.pop(ablate_index)
        prior_max.pop(ablate_index)
        title = 'Fixed ' + ablated_label.split('(')[0]
        figlabel = ['C','E','D','A','B'][ablate_index]
        ylim = [-5,45]
    else: 
        title = ''
        figlabel = ''
        ylim = None
    combined_samples = np.ndarray((n_reps*n_samples,len(labels),len(obs_list)))
    for seed in range(n_reps):
        # train
        if ablate_index is None: simulator = build_simulator(training_noise_scale=7)
        if ablate_index is not None: simulator = build_simulator(training_noise_scale=7, ablate_values=ablate_values, ablate_index=ablate_index)
        if train: sbi_train(path,subdir,simulator,prior_min,prior_max,seed,n_simulations)
        # sample
        all_samples = sample_posterior(path,subdir,simulator,obs_list,seed,n_samples,labels,plot=verbose)
        for i in range(len(obs_list)):
            combined_samples[n_samples*seed:n_samples*(seed+1),:,i] = all_samples[:,:,i]
    inferred_values = np.median(combined_samples, axis=0)
    # bootstrap
    n_bootstrap = 100
    size_bootstrap = 1000
    combined_effect_size = np.ndarray([len(labels),n_bootstrap])
    for b in range(n_bootstrap):
        indeces = np.random.randint(0,n_samples*n_reps,size_bootstrap)
        subset = combined_samples[indeces,:,:]
        combined_effect_size[:,b] = glassdelta_effect_size(np.squeeze(subset[:,:,0]),np.squeeze(subset[:,:,1]))
    effect_size_list = np.mean(combined_effect_size, axis=1)
    effect_size_stderr = np.std(combined_effect_size,axis=1) / np.sqrt(n_bootstrap)
    if ablate_index is None: simulator = build_simulator(training_noise_scale=0)
    else: simulator = build_simulator(training_noise_scale=0, ablate_values=ablate_values, ablate_index=ablate_index)
    fig, rmse_mean, rmse_sterr = plot_actual_inferred_data(simulator,obs_list,inferred_values,xlabel=False,ylabel=True,legend=True,title=title,figlabel=figlabel,ylim=ylim)
    plt.tight_layout()
    fig.savefig(os.path.join(path,subdir,'pitch_plots' + '.eps'),format='eps',dpi=600)
    fig = violin_plots(obs_list,combined_samples,prior_min,prior_max,effect_size_list=effect_size_list,effect_size_stderr=effect_size_stderr,labels=labels,conf_int_level=0.95,show=True)
    fig.savefig(os.path.join(path,subdir,'violin_plots' + '.eps'),format='eps',dpi=600)
    return inferred_values, rmse_mean, rmse_sterr

def bar_plot(path,all_labels):
    labels = copy.deepcopy(all_labels)
    labels = ['Fixed ' + label for label in labels]
    labels = [label.split('(')[0] for label in labels]
    labels.insert(0,'Full model')
    
    with open(os.path.join(path,'results_' + str(0) + '.pkl'), "rb") as handle:
        results = pickle.load(handle)
    obs_list = results.get('observation_list')
    
    rmse_means_all = np.ndarray([len(all_labels)+1,len(obs_list)])
    rmse_stderr_all = np.ndarray([len(all_labels)+1,len(obs_list)])
    
    for k,label in enumerate(labels):
        with open(os.path.join(path,'results_' + str(k) + '.pkl'), "rb") as handle:
            results = pickle.load(handle)
        rmse_means_all[k,:] = results.get('rmse_means')
        rmse_stderr_all[k,:] = results.get('rmse_stderr')
    
    bar_df = pd.DataFrame(rmse_means_all,columns=[obs_list[0].get('name'),obs_list[1].get('name')])
    se_df = pd.DataFrame(rmse_stderr_all,columns = [obs_list[0].get('name'),obs_list[1].get('name')])
    se_df['CA means for sorting'] = bar_df['CA patients']
    bar_df['Labels'] = labels
    se_df['Labels'] = labels
    bar_df = bar_df.sort_values('CA patients',ascending=False)
    se_df = se_df.sort_values('CA means for sorting',ascending=False)
    ax = bar_df.plot(kind='bar',yerr=se_df,rot=0,color=[obs_list[0].get('color'),obs_list[1].get('color')],figsize=(5,5))
    ax.set_ylabel('RMSE (cents)',fontsize=12)
    ax.set_xlabel('Reduced model',fontweight='bold',fontsize=12)
    ax.set_xticks(ticks=range(len(labels)),labels=bar_df["Labels"],fontsize=12,rotation=30,ha='right')
    tick_labels = ax.get_xticklabels()
    tick_labels[bar_df['Labels'].tolist().index('Full model')].set_fontweight('bold')
    plt.tight_layout()
    plt.savefig(os.path.join(path,'bar_plot.eps'),format='eps',dpi=600)


bar_plot(path = os.path.join(os.getcwd(),'SBI_results'), all_labels = ['Aud Delay (ms)',
          'Somat Delay (ms)',
          'Fb Noise Var (log)',
          'Fb Noise Ratio (Aud:Som)',
          'Controller Gain',
          ])