#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 11:19:59 2023

@author: sg
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score
from xgboost import plot_importance
from matplotlib import pyplot
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns




with open('fNIRS_hyperscanning_time_series_dataset1.pickle', 'rb') as handle:
    dataset = pickle.load(handle)


target = dataset['target - 1=paired,0=rand']

DTWs_raw = dataset['DTW_matrix']
DTWs = np.array(DTWs_raw)

def subtract_one(array):
  """Subtracts 1 from every value in the array.

  Args:
    array: The array to subtract 1 from.

  Returns:
    The array with 1 subtracted from every value.
  """

  # Get the shape of the array.
  shape = array.shape

  # Create a new array with the same shape.
  new_array = np.zeros(shape)

  # Iterate over the array.
  for i in range(shape[0]):
    new_array[i] = array[i] - 1

  return new_array

import csv

# Open the .csv file in read mode.
with open('taskclass.csv', 'r') as csvfile:

    # Create a csv reader object.
    csvreader = csv.reader(csvfile)

    # Load the contents of the .csv file into a list.
    datacsv = list(csvreader)

task_classes = np.array(datacsv)

DTWs_2 = DTWs[0:544]


# PLVs = dataset['PLV']
# PLVs2 = np.vstack(PLVs)

# cPCCs = dataset['cPCC']
# cPCCs1=np.array(cPCCs)
# cPCCs2 = cPCCs1.reshape((1056, 48))
# # Find and replace NaN and inf values with -1
# cPCCs2[np.isnan(cPCCs2) | np.isinf(cPCCs2)] = 0
# cPCC_mats = dataset['cPCC_matrices']
# cPCC_mats = np.array(cPCC_mats)

# meansU = np.array(dataset['block_mean_U'])
# meansY = np.array(dataset['block_mean_Y'])
# meansUY = np.concatenate((meansU, meansY), axis=1)

# varU = np.array(dataset['block_variance_U'])
# varY = np.array(dataset['block_variance_Y'])
# varUY = np.concatenate((varU, varY), axis=1)

# ttpU = np.array(dataset['block_time2peak_U'])
# ttpY = np.array(dataset['block_time2peak_Y'])
# ttpUY = np.concatenate((ttpU, ttpY), axis=1)





X = DTWs_2
X = np.reshape(X, (X.shape[0], -1))

task_classes = [int(string) for string in task_classes]
task_classes = [x - 1 for x in task_classes]
live_class = [0 if x == 0 or x == 1 else 1 for x in task_classes]
mask_class = [0 if x == 0 or x == 3 else 1 for x in task_classes]

y= task_classes[0:544]  # Labels


# Split dataset into training set and test set 7
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=(7)) # 70% training and 30% test


param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.1, 0.01, 0.001],
    'n_estimators': [50, 100, 200],
    #'min_child_weight': [1, 5, 10],
    #'gamma': [0.5, 1, 1.5, 2, 5],
    #'subsample': [0.6, 0.8, 1.0],
    #'colsample_bytree': [0.6, 0.8, 1.0],
}

#xgb_model = xgb.XGBClassifier()

# Use grid search to find the best hyperparameters
# grid_search = GridSearchCV(xgb_model, param_grid, cv=5)
# grid_search.fit(X_train, y_train)

#
# Print the best hyperparameters
# print("Best hyperparameters: ", grid_search.best_params_)

params = {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 200}




# Create the XGBoost classifier model

xgb_model = xgb.XGBClassifier(**params)

xgb_model.fit(X_train, y_train)

# dict on the test set
y_pred = xgb_model.predict(X_test)

plot_importance(xgb_model)
pyplot.show()

print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

# Print the confusion matrix
print("Confusion matrix:")
print(cm)

# Plot the confusion matrix as a heatmap
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')

# Add title and axis labels
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')

# Show the plot
plt.show()



# Define the number of folds for cross-validation
num_folds = 20

# Create a KFold cross-validation object
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Perform K-fold cross-validation
scores = cross_val_score(xgb_model, X_train, y_train, cv=kf, scoring='accuracy')

# Print the accuracy scores for each fold
for fold_idx, score in enumerate(scores):
    print(f"Fold {fold_idx+1}: Accuracy = {score:.4f}")

# Calculate and print the average accuracy across all folds
average_accuracy = np.mean(scores)
print("\nAverage Accuracy:", average_accuracy)

# Train the model on the full training set and evaluate on the test set
xgb_model.fit(X_train, y_train)
test_accuracy = xgb_model.score(X_test, y_test)
print("Test Accuracy:", test_accuracy)











