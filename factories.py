# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 15:55:32 2021

@author: Jessica Gaines
"""
from control_law import ControlLaw
from observer import Observer
from target import Target

'''
Factory methods for dynamic creation of different implementations of model components
'''

def ControlLawFactory(controller_config,plant):
    controller = None
    if 'adaptation_type' in controller_config:
        print('implementation in progress')
    else: 
        controller = ControlLaw(controller_config,plant)
    return controller

def ObserverFactory(observer_config,plant,ts,aud_delay=None,som_delay=None,estimated_arn=None,estimated_srn=None):
    observer = None
    observer = Observer(observer_config,plant,ts,aud_delay,som_delay,estimated_arn,estimated_srn)
    return observer

def TargetFactory(target_config,plant):
    target = None
    if 'adaptation_type' in target_config:
        print('implementation in progress')
    else: 
        target = Target(target_config,plant)
    return target