# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 11:06:39 2023

@author: jgaines
"""

import numpy as np
from data_read import read_obs
import time
from analysis_utils import plot_actual_data, run_sbi, bar_plot
import matplotlib.pyplot as plt
import os
import copy
import sys
import pickle
import torch


def main(argv):
    start = time.time()
    path = os.path.join(os.getcwd(),'SBI_results_final_for_paper')
    if not os.path.exists(path): os.mkdir(path)
    logpath = os.path.join(path,'log' + time.strftime("%Y%m%d%H%M%S", time.localtime(start)) + '.txt')
    
    with open(logpath, 'w') as logfile:
        logfile.write("00:00:00 Start\n")
    
    all_labels = ['Aud Delay (ms)',
              'Somat Delay (ms)',
              'Fb Noise Var (log)',
              'Fb Noise Ratio (Aud:Som)',
              'Controller Gain',
              ]
    
    observation_list = []
        
    observation_list.append(read_obs('pitch_pert_data/CA_Data/','CA','jh','control',"#C7221F"))
    observation_list.append(read_obs('pitch_pert_data/CA_Data/','CA','jh','patient',"#456990"))
    print('Range of data set 1 in cents')
    print(max(observation_list[0].get('data'))-min(observation_list[0].get('data')))
    print('Range of data set 2 in cents')
    print(max(observation_list[1].get('data'))-min(observation_list[1].get('data')))
    plot_actual_data(observation_list,xlabel='Time (s)',ylabel='Pitch (cents)',legend=True,show_pert=True,ylim=None)
    plt.tight_layout()
    plt.savefig(os.path.join(path,'actual_data.eps'),format='eps',dpi=600)
    with open(logpath, 'a') as logfile:
        logfile.write(time.strftime("%H:%M:%S", time.gmtime(time.time()-start)) + " Actual data plot saved in " + path +"\n")
    
    prior_min_all = [50, 3, -6.5, 0.1, 0.1]
    prior_max_all = [200, 80, -3, 6, 8]
    
    
    n_simulations=int(argv[0])
    n_samples=10000
    n_reps = int(argv[1])
    train=bool(argv[2])
    
    with open(logpath, 'a') as logfile:
        logfile.write('Priors: \n')
        for i, label in enumerate(all_labels):
            logfile.write(label + " Min: " + str(prior_min_all[i]) + ' Max: ' + str(prior_max_all[i]) + "\n")
        logfile.write('Simulations: ' + str(n_simulations) + "\n")
        logfile.write('Samples: ' + str(n_samples)+ "\n")
        logfile.write('Repetitions: ' + str(n_reps)+ "\n")
        logfile.write('Train: ' + str(train)+ "\n")
        logfile.write(time.strftime("%H:%M:%S", time.gmtime(time.time()-start)) + " Begin inference\n")
    
    if len(argv) == 3:
        inferred_values, rmse_means, rmse_stderr = run_sbi(path,'all_params',observation_list,n_simulations,n_samples,n_reps,prior_min_all,prior_max_all,all_labels,train=train)
        inferred_control_values = inferred_values[:,0]
        print(inferred_control_values)
        
        with open(logpath, 'a') as logfile:
            logfile.write(time.strftime("%H:%M:%S", time.gmtime(time.time()-start)) + " Inference completed\n\n")
            logfile.write('Inferred values: \n')
            for i, obs in enumerate(observation_list):
                logfile.write('\n\t' + obs.get('name') + "\n")
                for j, label in enumerate(all_labels):
                    logfile.write('\t' + label + ": " + str(inferred_values[j,i])+ "\n")
            logfile.write('RMSE: \n')
            for i, obs in enumerate(observation_list):
                logfile.write('\t' + obs.get('name') + ": " + "{:.4f}".format(rmse_means[i]) + "+/-" + "{:.4f}".format(rmse_stderr[i]) + "\n")
        results = {}
        results['inferred_values'] = inferred_values
        results['rmse_means'] = rmse_means
        results['rmse_stderr'] = rmse_stderr
        results['label'] = 'Full Model'
        results['observation_list'] = observation_list
        with open(os.path.join(path,'results_' + str(0)+'.pkl'), "wb") as handle:
            pickle.dump(results,handle)
    
    if len(argv) == 4:
        with open(os.path.join(path,'results_0.pkl'), "rb") as handle:
            full_model_results = pickle.load(handle)
        inferred_control_values = full_model_results.get('inferred_values')[:,0]
        print(inferred_control_values)
        
        k = int(argv[3])
        label = all_labels[k]
        label = label.split('(')[0]
        label = label.strip()
        inferred_values, rmse_means, rmse_stderr = run_sbi(path,'Fix ' + label,observation_list,n_simulations,n_samples,n_reps,prior_min_all,prior_max_all,all_labels,train=train,ablate_index=k,ablate_values=inferred_control_values)
        results = {}
        results['inferred_values'] = inferred_values
        results['rmse_means'] = rmse_means
        results['rmse_stderr'] = rmse_stderr
        results['label'] = label
        with open(os.path.join(path,'results_' + str(k+1)+'.pkl'), "wb") as handle:
            pickle.dump(results,handle)
        
        
        labels = copy.deepcopy(all_labels)
        del labels[k]
        
        with open(logpath, 'a') as logfile:
            logfile.write('\n***Ablate ' + all_labels[k] + '***\n')
            logfile.write(time.strftime("%H:%M:%S", time.gmtime(time.time()-start)) + " Inference completed\n\n")
            logfile.write('Inferred values: \n')
            for i, obs in enumerate(observation_list):
                logfile.write('\n\t' + obs.get('name') + "\n")
                for j, label in enumerate(labels):
                    logfile.write('\t' + label + ": " + str(inferred_values[j,i])+ "\n")
            logfile.write('RMSE: \n')
            for i, obs in enumerate(observation_list):
                logfile.write('\t' + obs.get('name') + ": " + "{:.4f}".format(rmse_means[i]) + " +/- " + "{:.4f}".format(rmse_stderr[i]) + "\n")
        
if __name__ == "__main__":
    main(sys.argv[1:])