# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 12:37:32 2021

@author: Jessica Gaines
"""
from plant import VocalTract
from feedback_alteration import FeedbackAlteration
import factories
import numpy as np

'''
Dynamically create state feedback control model composed of a plant,
target, feedback alteration, observer, and controller. Run one frame of the 
composed model.
'''
class Model():
    def __init__(self, config):
        self.config = config
        self.ts = float(config['Experiment']['sampling_time'])
        self.nframes = round(float(config['Experiment']['end_time'])/self.ts)
        self.ntrials = int(config['Experiment']['n_trials'])
        self.plant = VocalTract(config['Vocal_Tract'],self.ts)
        self.target = factories.TargetFactory(config['Target'],self.plant)
        self.feedback_alteration = FeedbackAlteration(config['Alteration'], self.nframes, self.ts, self.target.target_pitch)
        self.observer = factories.ObserverFactory(config['Observer'],self.plant, self.ts)
        self.controller = factories.ControlLawFactory(config['Controller'],self.plant)
        
    def run(self):
        y_output = np.ndarray((self.nframes,2,self.ntrials))
        errors = np.ndarray((self.nframes,2,self.ntrials))
        for i in range(self.ntrials):
            x_prev = np.array([np.squeeze(self.target.get_xtarg()[1]),np.squeeze(self.target.get_xtarg()[1]),0]).reshape((3,1))
            u_prev = 0
            xest_prev = x_prev.copy()
            self.observer.clear_buffers()
            for j in range(self.nframes):
                x,y = self.plant.run(x_prev,u_prev)
                yalt = self.feedback_alteration.run(y,j)
                xest,err,y_delayed,y_predict = self.observer.run(xest_prev,u_prev,yalt)
                u_control = self.controller.run(self.target.get_xtarg(),xest,self.target.get_ytarg(),y_predict)
                x_prev = x
                u_prev = u_control
                xest_prev = xest
                y_output[j,:,i] = np.squeeze(y)
                errors[j,:,i] = err
        return y_output,errors
        
    '''
    Methods for overriding parameters in various formats
    '''
    # Override feedback noise in the plant with a tensor
    def set_tunable_params(self,parameter_set):
        self.plant = VocalTract(self.config['Vocal_Tract'],self.ts,arn=10**parameter_set[2].item(),srn=(10**parameter_set[2].item())/parameter_set[3].item())
        self.observer = factories.ObserverFactory(self.config['Observer'],self.plant, self.ts,aud_delay=parameter_set[0].item(),som_delay=parameter_set[1].item())
        self.controller.ugain = parameter_set[4].item()
    # Override feedback noise in the observer with a tensor
    def set_tunable_params_noise_est(self,parameter_set):
        self.observer = factories.ObserverFactory(self.config['Observer'],self.plant, self.ts,aud_delay=parameter_set[0].item(),som_delay=parameter_set[1].item(), 
                                                  estimated_arn=10**parameter_set[2].item(), estimated_srn=(10**parameter_set[2].item())/parameter_set[3].item())
        self.controller.ugain = parameter_set[4].item()
    # Override feedback noise in the plant with a list
    def set_tunable_params_list(self,parameter_set):
        self.plant = VocalTract(self.config['Vocal_Tract'],self.ts,arn=10**parameter_set[2],srn=10**parameter_set[2]/parameter_set[3])
        self.observer = factories.ObserverFactory(self.config['Observer'],self.plant, self.ts,aud_delay=parameter_set[0],som_delay=parameter_set[1])
        self.controller.ugain = parameter_set[4]
    # Override feedback noise in the observer with a list
    def set_tunable_params_noise_est_list(self,parameter_set):
        self.observer = factories.ObserverFactory(self.config['Observer'],self.plant, self.ts,aud_delay=parameter_set[0],som_delay=parameter_set[1], 
                                                  estimated_arn=10**parameter_set[2], estimated_srn=(10**parameter_set[2])/parameter_set[3])
        self.controller.ugain = parameter_set[4]
    # Override feedback noise in the observer but don't use feedback noise ratio -- set auditory noise to param2 and somat noise to param3
    def set_tunable_params_sa_est_list(self,parameter_set):
        self.observer = factories.ObserverFactory(self.config['Observer'],self.plant, self.ts,aud_delay=parameter_set[0],som_delay=parameter_set[1], 
                                                  estimated_arn=parameter_set[2], estimated_srn=parameter_set[3])
        self.controller.ugain = parameter_set[4]
    # Override feedback noise in the plant but don't use feedback noise ratio -- set auditory noise to param2 and somat noise to param3
    def set_tunable_params_sa_list(self,parameter_set):
        self.plant = VocalTract(self.config['Vocal_Tract'],self.ts,arn=parameter_set[2],srn=parameter_set[3])
        self.observer = factories.ObserverFactory(self.config['Observer'],self.plant, self.ts,aud_delay=parameter_set[0],som_delay=parameter_set[1])
        self.controller.ugain = parameter_set[4]
        