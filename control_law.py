# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 09:13:48 2021

@author: Jessica Gaines
"""
import numpy as np

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
    
    def adapt(self,err):
        return 0
    
class AdaptiveControlLaw(ControlLaw):
    def __init__(self, control_params, plant):
        super().__init__(control_params,plant)
        self.weight = 1
        self.bias = 0
        self.adaptation_rate = float(control_params['adaptation_rate'])
        
    def get_controller_output(self, x_targ, xest, y_targ, y_predict, un):
        u_control = self.weight * super().get_controller_output(x_targ, xest, un) + self.bias
        return u_control
    
    def adapt(self,err):
        self.weight = self.weight + self.adaptation_rate * -err
        
class AdaptiveControlLaw_Target(AdaptiveControlLaw):
    def get_controller_output(self, x_targ, xest, y_targ, y_predict, un):
        u_control = ((self.weight * x_targ[0] + self.bias)-(self.ugain*xest[0])) + un
        return u_control