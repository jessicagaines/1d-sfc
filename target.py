# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 22:11:18 2021

@author: John Houde, Jessica Gaines
"""
import numpy as np

'''
Defines target F0 from config file
X is state target
Y is sensory feedback target
'''

class Target():
    def __init__(self,target_config,plant):
        self.target_pitch = float(target_config['target_pitch'])
        self.starting_pitch= float(target_config['starting_pitch'])
        self.target_alpha = float(target_config['target_alpha'])
        self.C = plant.sysd.C
        target_state = (1/plant.sysd.C[0,1])*self.target_pitch
        self.x_targ = np.array([[0],[target_state],[0]])
        self.y_targ = np.array([[self.target_pitch],[(self.C[1,:]*self.x_targ).item()]])
        
    def get_xtarg(self):
        return self.x_targ
    
    def get_ytarg(self):
        return self.y_targ
