# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 22:38:41 2020

@author: KX764QE
"""
import joblib
import config

def make_prediction(input_data):
    """
    make prediction using save pipeline model
    """
    _pipe_price = joblib.load(filename=config.PIPELINE_NAME)
    
    results = _pipe_price.predict(input_data)

    return results


if __name__ == '__main__':
    
    """
    test pipeline using test set
    """ 
    
    from sklearn.metrics import  accuracy_score, precision_score, recall_score, f1_score
    import data_management as dm
    
    df = dm.read_dataset(config.DATA_PATH)
    X_train, X_test, y_train, y_test = dm.split_dataset(df)
    
    y_pred = make_prediction(X_test)
    
    print(accuracy_score(y_test, y_pred.round()))
    print(precision_score(y_test, y_pred.round()))
    print(recall_score(y_test, y_pred.round()))
    print(f1_score(y_test, y_pred.round()))