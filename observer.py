# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 09:14:31 2021

@author: Jessica Gaines
"""
from util import string2dtype_array
from collections import deque
import numpy as np
import control.matlab as ctrl

class Observer():
    def __init__(self,observer_params, plant, ts):
        kalfact_base = float(observer_params['kalfact_base'])
        kalfact_balance = string2dtype_array(observer_params['kalfact_balance'],'float')
        self.aud_fdbk_delay = float(observer_params['aud_fdbk_delay'])
        self.somat_fdbk_delay = float(observer_params['somat_fdbk_delay'])
        self.ts = ts
        Kalfact = kalfact_base * kalfact_balance
        self.plant = plant
        self.aud_fdbk_buffer = deque(maxlen=round(self.aud_fdbk_delay/1000/self.ts))
        self.somat_fdbk_buffer = deque(maxlen=round(self.somat_fdbk_delay/1000/self.ts))
        self.aud_pred_buffer = deque(maxlen=round(self.aud_fdbk_delay/1000/self.ts))
        self.somat_pred_buffer = deque(maxlen=round(self.somat_fdbk_delay/1000/self.ts))
        [X,L,G] = ctrl.dare(plant.sysd.A.T,plant.sysd.C.T,plant.Q,plant.R)
        kal_gain = X*plant.sysd.C.T * np.linalg.inv(plant.R + plant.sysd.C * X * plant.sysd.C.T)
        self.kal_gain_scaled = np.multiply(Kalfact,kal_gain)
        
    def run(self, x_est,uprev,y_alt):
        y_delayed = delay_sensory_feedback([self.aud_fdbk_buffer,self.somat_fdbk_buffer],y_alt)
        x_predict = self.predict_state(x_est,uprev)
        y_predict = self.predict_feedback(x_predict)
        # always add sensory prediction to the buffer
        self.aud_pred_buffer.appendleft(y_predict[0])
        self.somat_pred_buffer.appendleft(y_predict[1])
        # if feedback exists, begin comparing to prediction
        if not np.isnan(y_delayed[0]): 
            aud_err = y_delayed[0] - self.aud_pred_buffer.pop()
        else: aud_err = 0
        if not np.isnan(y_delayed[1]):
            somat_err = y_delayed[1] - self.somat_pred_buffer.pop()
        else: somat_err = 0
        # calculate sensory errors
        err = np.array((2,1),dtype='float'); err[0]=aud_err; err[1]=somat_err
        # update state estimate
        x_est_new = self.update_state_estimate(x_est,self.kal_gain_scaled,err.reshape((2,1)))
        return x_est_new, err, y_delayed, y_predict
    
    def predict_state(self,x_est,uprev):
        return self.plant.sysd.A*x_est + self.plant.sysd.B*uprev
    
    def predict_feedback(self,x_predict):
        return self.plant.sysd.C * x_predict
    
    def update_state_estimate(self,x_est,kalman_gain,errors):
        return x_est + np.dot(kalman_gain,errors)
    
    def clear_buffers(self):
        self.aud_fdbk_buffer.clear()
        self.somat_fdbk_buffer.clear()
        self.aud_pred_buffer.clear()
        self.somat_pred_buffer.clear()
    
    def adapt(self,err):
        return 0
    
    def get_bias(self):
        return np.array([0,0]).reshape((2,1))

def delay_sensory_feedback(buffers,y_alt):
    delayed_fdbk = np.zeros(len(buffers))
    # always add incoming feedback to each buffer
    for i in range(len(buffers)):
        buffers[i].appendleft(y_alt[i])
    # if delay has passed, begin dequeuing feedback
        if len(buffers[i]) >= buffers[i].maxlen:
            delayed_fdbk[i] = buffers[i].pop()
        else: delayed_fdbk[i] = np.nan
    return delayed_fdbk
    
class ObserverFixed(Observer):
    def __init__(self,observer_params, plant, ts):
        super().__init__(observer_params, plant, ts)
        self.kal_gain_scaled = np.array([[0.00060795, 0.00109431],
                                            [0.00071428, 0.00128571],
                                            [0.04397977, 0.07916358]])

class AdaptiveObserverState(Observer):
    def __init__(self,observer_params, plant, ts):
        super().__init__(observer_params, plant, ts)
        self.weight = np.ones(3).reshape((3,1))
        self.bias = np.zeros(3).reshape((3,1))
        self.adaptation_rate = float(observer_params['adaptation_rate'])
    
    def predict_state(self,x_est,uprev):
        return np.multiply(self.weight,super().predict_state(x_est,uprev)) + self.bias
    
    def adapt(self,err):
        self.bias = self.bias + np.array([0,1,0]).reshape((3,1)) * self.adaptation_rate * -err
        
    def get_bias(self):
        return self.bias
    
class AdaptiveObserverFeedback(Observer):
    def __init__(self,observer_params, plant, ts):
        super().__init__(observer_params, plant, ts)
        self.weight = np.ones(2).reshape((2,1))
        self.bias = np.zeros(2).reshape((2,1))
        self.adaptation_rate = string2dtype_array(observer_params['adaptation_rate'],'float')
    
    def predict_feedback(self,x_predict):
        #return super().predict_feedback(x_predict)
        return np.multiply(self.weight,super().predict_feedback(x_predict)) + self.bias
    
    def adapt(self,err):
        self.bias[0] = self.bias[0] + (self.adaptation_rate[0] * err[0])
        self.bias[1] = self.bias[1] + (self.adaptation_rate[1] * err[1])
        
    def get_bias(self):
        return self.bias