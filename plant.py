# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 09:14:16 2021

@author: Jessica Gaines
"""
import numpy as np
import control.matlab as ctrl

class VocalTract():
    def __init__(self,vocal_tract_params,ts,arn=None,srn=None):
        damping_ratio = float(vocal_tract_params['damping_ratio'])
        k = float(vocal_tract_params['spring_constant'])
        m = float(vocal_tract_params['mass'])
        if arn is None: self.arn = float(vocal_tract_params['aud_noise_covariance'])
        else: self.arn = arn
        if srn is None: self.srn = float(vocal_tract_params['somat_noise_covariance'])
        else: self.srn = srn
        self.qn = float(vocal_tract_params['state_noise_covariance'])
        damping = damping_ratio*2*np.sqrt(m*k)
        A = np.array([[0, 1], [-k/m, -damping/m]])
        B = np.array([[0, k/m]]).T
        #C = np.array([[100, 0], [100, 0]])
        C = np.array([[1, 0], [1, 0]])
        D = np.array([[0,0]]).T # auditory and somatosensory feedback
        sys1 = ctrl.ss(A,B,C,D) 
        sys2 = ctrl.ss(0,1,1,0); 
        # note: only in descrete version of integrator is A = 1
        sys = ctrl.series(sys2,sys1)
        self.sysd = sys.sample(ts)
        self.Q = self.qn*np.identity(self.sysd.A.shape[0])
        self.R = np.diagflat([self.arn,self.srn])
    def run(self,xprev,uprev):
        #add system dependent noise
        state_noise = np.matmul(np.random.normal(0,1,self.sysd.A.shape[0]),np.linalg.cholesky(self.Q)).reshape((3,1))
        obs_noise = np.matmul(np.random.normal(0,1,self.sysd.C.shape[0]),np.linalg.cholesky(self.R)).reshape((2,1))
        x = self.sysd.A*xprev + self.sysd.B*uprev + state_noise
        y = self.sysd.C*x + obs_noise
        return x,y