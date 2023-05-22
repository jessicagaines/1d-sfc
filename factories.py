# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 15:55:32 2021

@author: Jessica
"""
from control_law import ControlLaw,AdaptiveControlLaw,AdaptiveControlLaw_Target
from observer import Observer,AdaptiveObserverState,AdaptiveObserverFeedback
from target import Target,AdaptiveTarget,AdaptiveTargetSimple

def ControlLawFactory(controller_config,plant):
    controller = None
    if 'adaptation_type' in controller_config:
        if controller_config['adaptation_type'] == 'control_output':
            controller = AdaptiveControlLaw(controller_config,plant)
        elif controller_config['adaptation_type'] == 'control_target':
            controller = AdaptiveControlLaw_Target(controller_config,plant)
        else: print('Unrecognized controller type')
    else: 
        controller = ControlLaw(controller_config,plant)
    return controller

def ObserverFactory(observer_config,plant,ts):
    observer = None
    if 'adaptation_type' in observer_config:
        if observer_config['adaptation_type'] == 'prediction':
            observer = AdaptiveObserverState(observer_config,plant,ts)
        elif observer_config['adaptation_type'] == 'simple_target':
            observer = Observer(observer_config,plant,ts)
        elif observer_config['adaptation_type'] == 'matlab':
            observer = AdaptiveObserverFeedback(observer_config,plant,ts)
        else: print('Unrecognized observer type')
    else: 
        observer = Observer(observer_config,plant,ts)
    return observer

def TargetFactory(target_config,plant):
    target = None
    if 'adaptation_type' in target_config:
        if target_config['adaptation_type'] == 'matlab':
            target = AdaptiveTarget(target_config,plant)
        elif target_config['adaptation_type'] == 'simple_target':
            target = AdaptiveTargetSimple(target_config,plant)
        else: 
            print('Unrecognized target type. Creating normal target')
            target = Target(target_config,plant)
    else: 
        target = Target(target_config,plant)
    return target