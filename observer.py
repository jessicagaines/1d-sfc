# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 09:14:31 2021

@author: John Houde, Jessica Gaines
"""
from util import string2dtype_array
from collections import deque
import numpy as np
import control.matlab as ctrl

'''
Defines the observer which updates the estimated state based on sensory feedback
'''
class Observer():
    def __init__(self,observer_params, plant, ts,aud_delay=None,som_delay=None, estimated_arn=None, estimated_srn=None):
        kalfact_base = float(observer_params['kalfact_base'])
        kalfact_balance = string2dtype_array(observer_params['kalfact_balance'],'float')
        if aud_delay is None: self.aud_fdbk_delay = float(observer_params['aud_fdbk_delay'])
        else: self.aud_fdbk_delay = aud_delay
        if som_delay is None: self.somat_fdbk_delay = float(observer_params['somat_fdbk_delay'])
        else: self.somat_fdbk_delay = som_delay
        self.ts = ts
        self.plant = plant
        self.Kalfact = kalfact_base * kalfact_balance
        self.aud_fdbk_buffer = deque(maxlen=round(self.aud_fdbk_delay/1000/self.ts))
        self.somat_fdbk_buffer = deque(maxlen=round(self.somat_fdbk_delay/1000/self.ts))
        self.aud_pred_buffer = deque(maxlen=round(self.aud_fdbk_delay/1000/self.ts))
        self.somat_pred_buffer = deque(maxlen=round(self.somat_fdbk_delay/1000/self.ts))
        # Hierarchy of sensory noise values -- highest priority constructor input, second priority config file. 
        # If these are unavailable then use plant noise
        if estimated_arn is not None and estimated_srn is not None:
            R = np.diagflat([estimated_arn, estimated_srn])
        elif 'estimated_arn' in observer_params.keys() and 'estimated_srn' in observer_params.keys():
            R = np.diagflat([float(observer_params['estimated_arn']),float(observer_params['estimated_srn'])])
        else:
            R = self.plant.R
        # Calculate Kalman gain
        [X,L,G] = ctrl.dare(self.plant.sysd.A.T,self.plant.sysd.C.T,self.plant.Q,R)
        kal_gain = X*plant.sysd.C.T * np.linalg.inv(R + self.plant.sysd.C * X * self.plant.sysd.C.T)
        self.kal_gain_scaled = np.multiply(self.Kalfact,kal_gain)
        
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