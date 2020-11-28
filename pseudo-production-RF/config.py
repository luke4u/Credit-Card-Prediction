# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 21:38:47 2020

@author: KX764QE
"""


# data
DATA_PATH = 'creditcard.csv'

RANDOM_SEED = 101
# target
TARGET = 'Class'

# input features
FEATURES = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 
            'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17',
            'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 
            'V26', 'V27', 'V28', 'Amount']

# to-be-scaled feature
SCALING_FEATURE = ['Amount']

# model
ESTIMATOR_SIZE = 100
MODEL_PATH = 'rf_model.h5'
ENCODER_PATH = 'scaler.pkl'
PIPELINE_NAME = 'rf_pipe.pkl'
CLASSES_PATH = 'classes.pkl'