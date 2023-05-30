#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 16:04:06 2023

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



# Load HbO data
with open('UCL_S07.pickle', 'rb') as handle:
    UCL_data = pickle.load(handle)    
with open('Yale_S07.pickle', 'rb') as handle2:
    Yale_data = pickle.load(handle2)  

#Load time data
with open('UCL_S07_t.pickle', 'rb') as handle:
    UCL_data_t = pickle.load(handle)
with open('Yale_S07_t.pickle', 'rb') as handle2:
    Yale_data_t = pickle.load(handle2)
    
# Load offset files
of_u = mat73.loadmat('S07U_onsets.mat')
of_y = mat73.loadmat('S07Y_onsets.mat')

of_u_clone = mat73.loadmat('S07U_onsets.mat')
of_y_clone = mat73.loadmat('S07Y_onsets.mat')
of_y_clone = of_y_clone['onsets']
of_u_clone = of_u_clone['onsets']


# Trim off any unneeded channels
UCL_data = UCL_data[0:48]
Yale_data = Yale_data[0:48]




# extract offsets from their dict and order them numerically

of_y = of_y['onsets']
of_u = of_u['onsets']

of_y = np.concatenate(of_y, axis=None)
of_u = np.concatenate(of_u, axis=None)

of_y = np.sort(of_y)
of_u = np.sort(of_u)

of_y = of_y[0:len(of_u)]


# now create 2 new arrays called 'end times' which contain the
# end times of each block 30 (N) seconds later

end_y = of_y + 15
end_u = of_u + 15

# Now we have calculated the start and end times of each block, we can
# now work on seperating each dataset into smaller blocks

#####################################

# we can do this by finding the indices of the time vector which
# match the start and end points 

def find_closest_timepoints(start_times, end_times, timepoints):
    """
    Find the index of the timepoints closest 
    to each start and end point value.

    Returns:
    --------
    ndarray, ndarray
        Two arrays containing the indices 
        of the timepoints closest to each start and end point value,
        respectively.
    """
    start_indices = np.searchsorted(timepoints, start_times)
    end_indices = np.searchsorted(timepoints, end_times)
    return start_indices, end_indices

yaleinds=find_closest_timepoints(of_y, end_y, Yale_data_t)
UCLinds=find_closest_timepoints(of_u, end_u, UCL_data_t)




startyale = yaleinds[0]
startUCL = UCLinds[0]

endyale = yaleinds[1]
endUCL = UCLinds[1]


# Next, knowing the indices of the start and end point of each block
# we can split the main time series UCL_data\Yale_data
# into smaller pieces

test_arr = UCL_data[0]


def split_array_by_indices(arr, start, end):
    """
    inputs: 'arr' The input array. 'start' The array of start indices.
    'end' The array of end indices.

    Returns: A list of smaller arrays of the input array, starting with the indices of the start array
        and ending with the indices of the end array.
    """
    result = []
    for i, j in zip(start, end):
        result.append(arr[i:j+1])
    return result



UCL_split_t = split_array_by_indices(UCL_data_t, startUCL, endUCL)
Yale_split_t = split_array_by_indices(Yale_data_t, startyale, endyale)






def split_data(arr_list, startUCL, endUCL):
    """
    Takes a list of arrays and splits them based on start and end time indices
    using split_array_by_indices func
    """
    new_list = []
    for arr in arr_list:
        new_arr = split_array_by_indices(arr, startUCL, endUCL)
        new_list.append(new_arr)
    return new_list


UCL_data_split = split_data(UCL_data, startUCL, endUCL)
Yale_data_split = split_data(Yale_data, startyale, endyale)


# test by plotting 
#plt.plot(Yale_split_t[2],Yale_data_split[0][2])



# so now we can return the Split up data, N*arrays containing 30 secs worth
# of data, as well as the split up timepoints



# uncomment when saving files

with open('UCL_S07_split.pickle', 'wb') as handle:
    pickle.dump(UCL_data_split, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('Yale_S07_split.pickle', 'wb') as handle:
    pickle.dump(Yale_data_split, handle, protocol=pickle.HIGHEST_PROTOCOL)



with open('UCL_S07_t_split.pickle', 'wb') as handle:
    pickle.dump(UCL_split_t, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('Yale_S07_t_split.pickle', 'wb') as handle:
    pickle.dump(Yale_split_t, handle, protocol=pickle.HIGHEST_PROTOCOL)





