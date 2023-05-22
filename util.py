# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 09:40:59 2021

@author: JLG
"""
import numpy as np
import re
    
def string2dtype_array(string,dtype):
    string = re.sub(r'[\s\[\]]', '', string)
    string_array = np.array(re.split(',', string))
    type_array = string_array.astype(dtype)
    return type_array

def split_data(data_x,data_y,prop):
    shuffled_index = np.arange(data_x.shape[0])
    np.random.shuffle(shuffled_index)
    len_test_set = np.round(data_x.shape[0] * prop).astype(int)
    test_index = shuffled_index[0:len_test_set]
    training_index = shuffled_index[len_test_set:]
    ## assign data to each set
    x_test = data_x.iloc[test_index].to_numpy()
    y_test = data_y.iloc[test_index].to_numpy()
    x_train = data_x.iloc[training_index].to_numpy()
    y_train = data_y.iloc[training_index].to_numpy()
    return x_train,y_train,x_test,y_test