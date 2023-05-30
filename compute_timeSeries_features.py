#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 14:40:41 2023

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
 
 
with open('UCL_S09_split.pickle', 'rb') as handle3:
    UCL_data_split = pickle.load(handle3)    

with open('Yale_S09_split.pickle', 'rb') as handle4:
    Yale_data_split = pickle.load(handle4) 




def calcfeatures(data, frequency):
    means = []
    variances = []
    t2ps = []
    for arr in data:
        mean = np.mean(arr)
        means.append(mean)
              
        std = np.std(arr)
        var = std/mean
        variances.append(var)
        
        max_magnitude = float('-inf')
        max_index = None
    
        for i in range(len(arr)):
            if abs(arr[i]) > max_magnitude and i != 0:
                max_magnitude = abs(arr[i])
                max_index = i
                # UCL frequency = 12.3Hz
                # Yale frequency = 8.547Hz
                t2p = max_index/frequency
        t2ps.append(t2p) 
    return means, variances, t2ps

def calculate_features_split(arrays, frequency):
    features = []
    for array in arrays:
        feature = calcfeatures(array, frequency)
        features.append(feature)
    return features

UCL_frequency = 12.3
Yale_frequency = 8.547
UCL_general_features_array = calculate_features_split(UCL_data_split, UCL_frequency)
Yale_general_features_array = calculate_features_split(Yale_data_split, Yale_frequency)



# Calculation of phase lockimg value


def phase_locking_value(x, y):
    # Resize x and y to the same shape
    n = max(len(x), len(y))
    x = np.resize(x, (n, 1))
    y = np.resize(y, (n, 1))
    
    # Compute the phase locking value between two time series x and y  
    
    # Apply the Hilbert transform to x and y to obtain their analytic signals
    x_analytic = hilbert(x)
    y_analytic = hilbert(y)

    # Compute the phase difference between the analytic signals
    phase_diff = np.angle(x_analytic / y_analytic)

    # Compute the phase locking value (PLV)
    plv = np.abs(np.mean(np.exp(1j * phase_diff)))

    return plv


def calculate_PLV(UCL_data, Yale_data):
    PLVs = []
    for i, value in enumerate(UCL_data):
        datau = UCL_data[i]
        datay = Yale_data[i]
        PLV = phase_locking_value(datau, datay)
        #print(PLV)
        PLVs.append(PLV)  
    return(PLVs)
        

def calculate_PLV_split(arrays1, arrays2):
    results = []
    for i in range(len(arrays1)):
        result = calculate_PLV(arrays1[i], arrays2[i])
        results.append(result)
    return results


PLV_array = calculate_PLV_split(UCL_data_split, Yale_data_split)



# calculation of circular pearsons correlation coef

def circular_pearson(x, y):
    
    """
    Computes the circular Pearson's correlation coefficient between two regular time series x and y,
    assumed to represent angles or directions.
    """
    # Resize x and y to the same shape
    n = max(len(x), len(y))
    x = np.resize(x, (n, 1))
    y = np.resize(y, (n, 1))
    # Convert x and y to phase information
    phase_x = np.angle(np.fft.fft(x))
    phase_y = np.angle(np.fft.fft(y))

    # Convert phase information to circular variables on the unit circle
    z_x = np.exp(1j * phase_x)
    z_y = np.exp(1j * phase_y)

    # Compute the circular Pearson's correlation coefficient
    r = np.abs(np.mean(z_x * np.conj(z_y))) / (np.std(z_x) * np.std(z_y))

    return r


def calculate_cPCC(UCL_data, Yale_data):
    cPCCs = []
    for i, value in enumerate(UCL_data):
        #print(i)
        datau = UCL_data[i]
        datay = Yale_data[i]
        cPCC = circular_pearson(datau, datay)
        #print(cPCC)
        cPCCs.append([cPCC])
    return(cPCCs)

def calculate_cPCC_split(arrays1, arrays2):
    results = []
    for i in range(len(arrays1)):
        result = calculate_cPCC(arrays1[i], arrays2[i])
        results.append(result)
    return results


cPCC_array = calculate_cPCC_split(UCL_data_split, Yale_data_split)
        
   


# Calculating dtw matrix

def calculate_distance_matrix(UCL_data, Yale_data):
    distances = []
    for i, value1 in enumerate(UCL_data):
        #print(i)
        #print(i)
        for j, value2 in enumerate(Yale_data):
            #print(j)
            #print(j)
            datau = UCL_data[i]
            datay = Yale_data[j]
            distance, path = fastdtw(datau, datay, dist=euclidean)
            #print(distance)
            #print(distance)
            distances.append([distance])


    return distances 

# After running this code, the variable transformed_data should contain the 
#desired data structure. The first index of transformed_data corresponds to 
#the block number, and the second index corresponds to the channel number. 
#So, for example, transformed_data[0][0] would give you the data for the first 
#channel in the first block.

def transform_data(data):
    transformed_data = []
    for i in range(32):
        block_data = []
        for j in range(48):
            channel_data = data[j][i]
            block_data.append(channel_data)
        transformed_data.append(block_data)
    return transformed_data

def computedtws(UCL_dtw_input_data, Yale_dtw_input_data):
    dtws = []
    #for i, value in enumerate(UCL_dtw_input_data):
    for i in range(0,len(UCL_dtw_input_data)):
        print(i)
        print('********************************')
        dtwu_data = UCL_dtw_input_data[i]
        dtwy_data = Yale_dtw_input_data[i]
        dtw = calculate_distance_matrix(dtwu_data,dtwy_data)
        dtw = np.array(dtw)
        dtws.append(dtw)
    return dtws


UCL_dtw_input_data = transform_data(UCL_data_split)
Yale_dtw_input_data = transform_data(Yale_data_split)
#DTW_arrays = computedtws(UCL_dtw_input_data,Yale_dtw_input_data)

def dtw_matrices(data):
    def array_group(arr, n):
        return np.column_stack([arr[i:i+n] for i in range(0, len(arr), n)])
    matrices = []
    for i in range(0,len(data)):
        grouped_arr = array_group(data[i], 48)

        plt.imshow(grouped_arr, cmap='viridis', interpolation='nearest')
        plt.show()

        matrices.append(grouped_arr)
    return matrices


#DTW_matrices = dtw_matrices(DTW_arrays)

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



# # with open('S14_U_general_features.pickle', 'wb') as handle:
# #     pickle.dump(UCL_general_features_array, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('S14_Y_general_features.pickle', 'wb') as handle:
#     pickle.dump(Yale_general_features_array, handle, protocol=pickle.HIGHEST_PROTOCOL)

# # with open('S14_S07_PLV_array.pickle', 'wb') as handle:
# #     pickle.dump(PLV_array, handle, protocol=pickle.HIGHEST_PROTOCOL)

# # with open('S14_S07_cPCC_array.pickle', 'wb') as handle:
# #     pickle.dump(cPCC_array, handle, protocol=pickle.HIGHEST_PROTOCOL)

# # with open('S14_S07_cDTW_matrices.pickle', 'wb') as handle:
# #     pickle.dump(DTW_matrices, handle, protocol=pickle.HIGHEST_PROTOCOL)









