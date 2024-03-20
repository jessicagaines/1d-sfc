# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 09:12:32 2021

@author: Jessica Gaines

Run state feedback control model defined by a given config file
Usage: python run_pitch_model.py [config file]
This file creates a plot of the simulated response to a feedback perturbation of F0.
"""
import configparser
from model import Model
import numpy as np
import plotting

def main(argv):
    config = configparser.ConfigParser()
    config.read(argv[0])
    np.random.seed(100)
    ts = float(config['Experiment']['sampling_time'])
    ntrials = int(config['Experiment']['n_trials'])
    nframes = round(float(config['Experiment']['end_time'])/ts)
    model = Model(config)
    #model.set_tunable_params_list([102.7, 35.3, -5, 2.0, 1.9])
    #print(model.observer.kal_gain_scaled)
    #print(model.plant.R)
    y_output,errors = model.run()
    make_plots(y_output,errors,model.feedback_alteration.onset,ts)

def make_plots(y_output,errors,alt_onset,ts):
    # Plotting
    pitch_output = y_output[:,0,:]
    nframes = y_output.shape[0]
    ntrials = y_output.shape[2]
    starting_pitch= y_output[0,0,0]
    t_axis = (np.arange(nframes) * ts) - alt_onset
    if alt_onset > 0:
        baseline = np.mean(pitch_output[:int(np.floor(alt_onset/ts)),:],axis=0) 
    else: baseline = np.mean(pitch_output[:int(np.floor(0.1/ts)),:],axis=0)
    baseline = np.asarray([starting_pitch]*ntrials)
    #plotting.plot_trial_timecourse(t_axis,pitch_output,scale='Hz',baseline=baseline)
    plotting.plot_trial_timecourse(t_axis,pitch_output,scale='cents',baseline=baseline)
    
main(['pitch_pert_configs.ini'])