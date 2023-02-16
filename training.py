'''
Functionality to train the model

Author: Christopher Bonham
Date: 16th February 2023
'''
import pandas as pd
import os
import logging
import pickle
from utils.io import read_config
from sklearn.linear_model import LogisticRegression


# Create a logger
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def load_training_data(in_path):
    '''Load training data to dataframe
    Inputs:
        in_path (string)
            Path to training data

    Outputs:
        pandas.dataframe
            Training data
    '''
    logger.info(f"training.py: Input folder path: {in_path}")

    # Load training data
    df = pd.read_csv(os.path.join(in_path, "finaldata.csv"))
    logger.info(f"training.py: Training data shape: {df.shape}")

    return df


def train_model(df, out_path):
    '''Train logistic regression model and pickle model file

    Inputs:
        df (pandas.dataframe)
            Training data
        out_path (string)
            Path to store the pickled model artifact
    Outputs:
        None
    '''
    # Extract labels and features
    y = df["exited"]
    X = df[["lastmonth_activity", "lastyear_activity", "number_of_employees"]]
    logger.info(f"training.py: y shape: {y.shape}")
    logger.info(f"training.py: X shape: {X.shape}")

    # Instantiate a Logistic regression model object
    lr = LogisticRegression(
        C=1.0, class_weight=None, dual=False, fit_intercept=True,
        intercept_scaling=1, l1_ratio=None, max_iter=100,
        multi_class='auto', n_jobs=None, penalty='l2',
        random_state=0, solver='liblinear', tol=0.0001, verbose=0,
        warm_start=False
    )

    # Fit the logistic regression to your data
    lr.fit(X, y)

    # Write trained model to pkl file
    # Create output directory if it doesnt exist
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    with open(os.path.join(out_path, "trainedmodel.pkl"), 'wb') as fp:
        pickle.dump(lr, fp)
    logger.info(f"training.py: Trained model written to "
                f"{os.path.join(out_path, 'trainedmodel.pkl')}")


def main():
    '''Main functionality call

    Inputs:
        None
    Outputs:
        None
    '''
    # Read the configuration file
    config = read_config(r".\config.json")
    # config = read_config(r".\config.json", display = True)
    logger.info("training.py: Configuration file read")

    # Load the trainiing data
    df = load_training_data(
        os.path.join(os.getcwd(), config["output_folder_path"])
    )

    # Train the model
    train_model(
        df,
        os.path.join(os.getcwd(), config["output_model_path"])
    )


# Top level script entry point
if __name__ == '__main__':
    main()
