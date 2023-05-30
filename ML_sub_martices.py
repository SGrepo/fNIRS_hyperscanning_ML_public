#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 17:14:01 2023

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
import sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.metrics import accuracy_score
from xgboost import plot_importance
from matplotlib import pyplot
from sklearn.metrics import classification_report
import csv

def apply_cutoff_filter(image_array, factor=3):
    # Reshape the array to make each image a separate element
    reshaped_array = image_array.reshape(image_array.shape[0], -1)

    # Find the median pixel value for each image
    median_values = np.median(reshaped_array, axis=1)

    # Apply the cutoff filter and set values to median
    filtered_array = np.where(reshaped_array > factor * median_values[:, np.newaxis], median_values[:, np.newaxis], reshaped_array)

    # Reshape the filtered array back to its original shape
    filtered_image_array = filtered_array.reshape(image_array.shape)

    return filtered_image_array



with open('fNIRS_hyperscanning_time_series_dataset1.pickle', 'rb') as handle:
    dataset = pickle.load(handle)


target = dataset['target - 1=paired,0=rand']

# Open the .csv file in read mode.
with open('taskclass.csv', 'r') as csvfile:

    # Create a csv reader object.
    csvreader = csv.reader(csvfile)

    # Load the contents of the .csv file into a list.
    datacsv = list(csvreader)

task_classes = np.array(datacsv)
task_classes = [int(string) for string in task_classes]
task_classes = [x - 1 for x in task_classes]

DTWs_raw = dataset['DTW_matrix']





# this line takes us to the first half of the data, i.e. true dyads
DTWs = np.array(DTWs_raw[4:544])
DTWs = apply_cutoff_filter(DTWs)


# This function splits the 48x48 matrix into 16 12x12 matrices, for all 540
def split_matrices(matrices):
    # Create an empty list for the output
    output = []
    
    # Loop over each matrix in the input list
    for matrix in matrices:
        # Create an empty list for the sub-arrays
        subarrays = []
        
        # Loop over each 12x12 block in the matrix
        for i in range(0, 48, 12):
            for j in range(0, 48, 12):
                # Extract the 12x12 block
                block = matrix[i:i+12, j:j+12]
                # Append the block to the list of sub-arrays
                subarrays.append(block)
        
        # Append the list of sub-arrays to the output
        output.append(subarrays)
    
    return output

DTW_sub = split_matrices(DTWs)





def split_lists(data):
    # Create a list of 16 empty lists
    output = [[] for i in range(144)]
    
    # Loop over each index in the input list
    for i in range(len(data)):
        # Loop over each array in the inner list
        for j in range(len(data[i])):
            # Append the matrix to the corresponding output list
            output[j].append(data[i][j])
    
    return output



# This line generates 16 (48x48 matrix split into 16 12x12 matrices) x N
# Each row is 540 12x12 matrices i.e. the sub matrix for each task block
# Each row may be used as an input to a machine learning classifier

DTW_sub_sub = split_lists(DTW_sub)[0:16]








importances_mat=[]
for i in range(16):
    print(i)
    test_dtw = np.array(DTW_sub_sub[i])



    X = test_dtw
    X = np.reshape(X, (X.shape[0], -1))

    y = task_classes[4:544]

    ############### ML starts

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=(7)) # 70% training and 30% test

    params = {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 200}

    xgb_model = xgb.XGBClassifier(**params)

    xgb_model.fit(X_train, y_train)

    # dict on the test set
    y_pred = xgb_model.predict(X_test)

    #plot_importance(xgb_model)
    #pyplot.show()
    accuracy_score_ = accuracy_score(y_test, y_pred)

    #print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))
    
    importances_mat.append(accuracy_score_)
    












    
def list_to_matrix(list):
  """Converts a 16-long list into a 4x4 matrix.

  Args:
    list: A 16-long list.

  Returns:
    A 4x4 matrix.
  """

  matrix = []
  for i in range(0, len(list), 4):
    matrix.append(list[i:i + 4])

  return matrix

importances_matrix = list_to_matrix(importances_mat)
    

def plot_colormap(matrix, title='', xlabel='', ylabel='', cmap='viridis', vmin=None, vmax=None):
    fig, ax = plt.subplots()
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Value')  # Add a label to the color bar
    plt.show()


plot_colormap(importances_matrix)



# for i in range(len(DTWs)):
#     #print(i)
#     data = DTWs[i]
#     row0_col0 = [row[:12] for row in data[:12]]
#     row0_col1 = [row[12:24] for row in data[:12]]
#     row0_col2 = [row[24:36] for row in data[:12]]
#     row0_col3 = [row[36:48] for row in data[:12]]
    
#     row1_col0 = [row[:12] for row in data[12:24]]
#     row1_col1 = [row[12:24] for row in data[12:24]]
#     row1_col2 = [row[24:36] for row in data[12:24]]
#     row1_col3 = [row[36:48] for row in data[12:24]]
    
#     row2_col0 = [row[:12] for row in data[24:36]]
#     row2_col1 = [row[12:24] for row in data[24:36]]
#     row2_col2 = [row[24:36] for row in data[24:36]]
#     row2_col3 = [row[36:48] for row in data[24:36]]
    
#     row3_col0 = [row[:12] for row in data[36:48]]
#     row3_col1 = [row[12:24] for row in data[36:48]]
#     row3_col2 = [row[24:36] for row in data[36:48]]
#     row3_col3 = [row[36:48] for row in data[36:48]]
    
#     #plot_colormap(row0_col1)


    

    
    
    
    
    
    
    
    
    
    
    
    


