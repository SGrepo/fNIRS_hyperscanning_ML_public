#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 13:36:31 2023

@author: sg
"""

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mat73
import pickle
from fastdtw import fastdtw  
from scipy.spatial.distance import euclidean  
from scipy.signal import hilbert
 
 
with open('UCL_S06_split.pickle', 'rb') as handle3:
    UCL_data_split = pickle.load(handle3)    

with open('Yale_S06_split.pickle', 'rb') as handle4:
    Yale_data_split = pickle.load(handle4) 
    
print(handle3)
    

def transform_data(data):
    transformed_data = []
    for i in range(32):
        block_data = []
        for j in range(48):
            channel_data = data[j][i]
            block_data.append(channel_data)
        transformed_data.append(block_data)
    return transformed_data

UCL_input_data = transform_data(UCL_data_split)
Yale_input_data = transform_data(Yale_data_split)


from scipy.interpolate import interp1d

def circular_correlation_coefficient(a, b):
    # Resample arrays to the same length
    n = np.lcm(a.shape[0], b.shape[0])
    a_resampled = np.interp(np.linspace(0, a.shape[0] - 1, n),
                            np.arange(a.shape[0]), a.ravel())
    b_resampled = np.interp(np.linspace(0, b.shape[0] - 1, n),
                            np.arange(b.shape[0]), b.ravel())

    # Compute circular Pearson's correlation coefficient
    a_mean = np.mean(a_resampled)
    b_mean = np.mean(b_resampled)
    a_std = np.std(a_resampled)
    b_std = np.std(b_resampled)
    r = np.sum(np.sin(a_resampled - a_mean) * np.sin(b_resampled - b_mean)) / (n * a_std * b_std)

    return r


def calculate_matrices(UCL_input_data, Yale_input_data,measure):
    def compute_z_pairwise(list1, list2):
        n = len(list1)
        m = len(list2)
        z_matrix = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                z_matrix[i, j] = measure(list1[i], list2[j])
        return z_matrix

    matrices = []
    for i, value in enumerate(UCL_input_data):
        print(i)
        dataU = UCL_input_data[i]
        dataY = Yale_input_data[i]
        matrix = compute_z_pairwise(dataU, dataY)
        plt.imshow(matrix, cmap='viridis', interpolation='nearest')
        plt.show()
        matrices.append(matrix)
    return matrices


#cPCC_matrices = calculate_matrices(UCL_input_data, Yale_input_data, circular_correlation_coefficient)

# with open('S06_cPCC_matrices.pickle', 'wb') as handle:
#       pickle.dump(cPCC_matrices, handle, protocol=pickle.HIGHEST_PROTOCOL)


datau = mat73.loadmat('preproc_S06U_bp_mc.mat')
of_u_clone = mat73.loadmat('S06U_onsets.mat')
of_y_clone = mat73.loadmat('S06Y_onsets.mat')


