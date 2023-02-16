'''
Utility (auxiliary) functions

Author: Christopher Bonham
Date: 16th February 2023
'''
import json
import os
import pickle


def read_config(pth, display=False):
    '''Read in a config file (JSON format) and return a dictionary object

    Inputs:
        pth (string)
            Path to config file
        display (boolean default = False
            Display contents of config JSON to scree
    Outputs:
        dict
            Contents of config file
    '''
    with open(pth, 'r') as fp:
        config = json.load(fp)

    if display:
        print(json.dumps(config, indent=3))

    return config


def load_model(in_path):
    '''Load model
    Inputs:
        in_path (string)
            Path to model directory

    Outputs:
        sklearn.linear_model._logistic.LogisticRegression
            Logistic regression model
    '''
    with open(os.path.join(in_path, "trainedmodel.pkl"), 'rb') as file:
        lr = pickle.load(file)

    return lr


def apply_model(df, lr):
    '''Apply a logistic regression model to a pandas dataframe

    Inputs:
        df (pandas.DataFrame)
            Dataframe to apply model to
        lr (sklearn.linear_model._logistic.LogisticRegression
            Logistic regression model)

    Outputs:
        numpy.array
            Model scores
    '''
    # Extract features
    X = df[["lastmonth_activity", "lastyear_activity", "number_of_employees"]]

    # Get model prediction
    y_pred = lr.predict(X)

    return y_pred
