# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 21:35:47 2020

@author: KX764QE
"""
import pandas as pd
import config
from sklearn.model_selection import train_test_split
import joblib

def read_dataset(data_folder):
    """
    read raw data
    """
    df = pd.read_csv(config.DATA_PATH)
    return df

def split_dataset(df):
    """
    split read-in data into train and test datasets
    """
    X_train, X_test, y_train, y_test = train_test_split(df[config.FEATURES], 
                                                        df[config.TARGET], 
                                                        test_size = 0.20,
                                                        random_state = config.RANDOM_SEED)
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)
    
    return X_train, X_test, y_train, y_test

def save_pipeline(pipe):
    #pipe.named_steps['model'].model.save(config.MODEL_PATH)
    joblib.dump(pipe, config.PIPELINE_NAME)
    

def load_pipeline():
    pipe = joblib.load(filename=config.PIPELINE_NAME)
    

if __name__ == '__main__':
    """
    read and split dataset
    """
    df = read_dataset(config.DATA_PATH)
    print(df.head())
    
    X_train, X_test, y_train, y_test = split_dataset(df)
    print(X_train.shape, X_test.shape)
    print(X_train.head())
    print(y_train.head())