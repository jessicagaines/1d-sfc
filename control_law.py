# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 09:13:48 2021

@author: John Houde, Jessica Gaines
"""
import numpy as np

'''
Defines the controller of the system -- calculate motor commands based on prediction error
'''
class ControlLaw():
    def __init__(self,control_params,plant,ugain=None):
        self.ugain = float(control_params['controller_gain'])
        self.u_noise = float(control_params['controller_noise'])
        aud_fdbk_alpha = float(control_params['aud_fdbk_alpha'])
        somat_fdbk_alpha = float(control_params['somat_fdbk_alpha'])
        self.xtargerr_gain = float(control_params['fdfwd_alpha'])
        self.ytargerr_gain = np.multiply(1/plant.sysd.C[:,1],np.array([[aud_fdbk_alpha],[somat_fdbk_alpha]]))
    
    def run(self,x_targ,xest,y_targ,y_predict):
        un = np.random.normal(0,1)*self.u_noise
        u_control = self.get_controller_output(x_targ,xest,y_targ,y_predict,un)
        return u_control.item()
    
    def get_controller_output(self,x_targ,xest,y_targ,y_predict,un):
        return self.ugain*(self.xtargerr_gain*(x_targ[1]-xest[1]) + self.ytargerr_gain.T*(y_targ-y_predict) + un)