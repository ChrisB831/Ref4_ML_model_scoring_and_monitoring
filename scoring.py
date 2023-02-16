'''
Functionality to apply the model to new data

Author: Christopher Bonham
Date: 16th February 2023
'''
import pandas as pd
import os
import logging
from utils.io import read_config, load_model
from sklearn.metrics import f1_score


# Create a logger
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def load_test_data(in_path):
    '''Load test data to dataframe
    Inputs:
        in_path (string)
            Path to test data

    Outputs:
        pandas.dataframe
            Training data
    '''
    logger.info(f"scoring.py: Input folder path: {in_path}")

    # Load training data
    df = pd.read_csv(os.path.join(in_path, "testdata.csv"))
    logger.info(f"scoring.py: Test data shape: {df.shape}")

    return df


def apply_model(df, lr, out_path):
    '''Load test data to dataframe
    Inputs:
        df (Pandas.datafrane)
            Data to score
        lr (sklearn.linear_model._logistic.LogisticRegression)
            Logistic regression model
        out_path (string)
            Path to store the F1 score
    Outputs:
        None
    '''
    logger.info(f"scoring.py: Output folder path: {out_path}")

    # Extract features
    y = df["exited"]
    X = df[["lastmonth_activity", "lastyear_activity", "number_of_employees"]]
    logger.info(f"scoring.py: y shape: {y.shape}")
    logger.info(f"scoring.py: X shape: {X.shape}")

    # Get model prediction and F1 score
    y_pred = lr.predict(X)
    f1 = f1_score(y, y_pred)
    logger.info(f"scoring.py: f1 score: {f1}")

    # Write f1 score to file
    with open(os.path.join(out_path, "latestscore.txt"), "w") as fp:
        fp.write(str(f1))
    logger.info(f"scoring.py: f1 score written to"
                f" {os.path.join(out_path, 'latestscore.txt')}")


def main():
    '''Main functionality call

    Inputs:
        None
    Outputs:
        None
    '''
    # Read the configuration file
    config = read_config(r".\config.json")
    logger.info("scoring.py: Configuration file read")

    # Load the test data
    df = load_test_data(
        os.path.join(os.getcwd(), config["test_data_path"])
    )

    # Load the model
    lr = load_model(
        os.path.join(os.getcwd(), config["output_model_path"])
    )

    # Apply the model to the test data and save F1 score
    apply_model(
        df, lr,
        os.path.join(os.getcwd(), config["output_model_path"])
    )


# Top level script entry point
if __name__ == '__main__':
    main()
