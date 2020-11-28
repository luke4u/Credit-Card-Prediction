# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 21:55:15 2020

@author: KX764QE
"""

import config
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

transformer = ColumnTransformer(transformers=[('scaler', StandardScaler(), config.SCALING_FEATURE)], 
                                remainder='passthrough')

pipe = Pipeline([
                ('scaler', transformer),
                ('model', RandomForestClassifier(n_estimators = config.ESTIMATOR_SIZE, 
                                                 random_state = config.RANDOM_SEED))
                ])


if __name__ == '__main__':
    """
    read dataset
    train, evaluate and save pipeline
    """
    
    from sklearn.metrics import  accuracy_score, precision_score, recall_score, f1_score
    import data_management as dm
    
    
    df = dm.read_dataset(config.DATA_PATH)
    X_train, X_test, y_train, y_test = dm.split_dataset(df)
    
    pipe.fit(X_train, y_train)
    
    y_pred = pipe.predict(X_test)
    
    print(accuracy_score(y_test, y_pred.round()))
    print(precision_score(y_test, y_pred.round()))
    print(recall_score(y_test, y_pred.round()))
    print(f1_score(y_test, y_pred.round()))
    
    dm.save_pipeline(pipe)