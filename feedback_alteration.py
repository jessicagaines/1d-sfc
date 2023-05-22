# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 14:11:17 2021

@author: JLG
"""
import numpy as np

class FeedbackAlteration():
    def __init__(self,alteration_params,nframes,ts,baseline):
        alt_type = alteration_params['type']
        alt_vec = np.zeros((2,nframes))
        if alt_type == 'aud_perturbation':
            self.onset = float(alteration_params['onset'])
            duration = float(alteration_params['duration'])
            cents_pert = float(alteration_params['cents_pert'])
            hertz_pert = (2**(cents_pert/1200)-1)*baseline
            onset_idx = round(self.onset/ts)
            offset_idx = round((self.onset+duration)/ts)
            alt_vec[0,onset_idx:offset_idx] = hertz_pert
            self.alt_vec = alt_vec
    def run(self,y,iframe):
        alt_y = y + self.alt_vec[:,iframe].reshape((2,1))
        return alt_y