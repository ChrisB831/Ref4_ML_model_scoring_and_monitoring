'''
Functionality to apply the model to new data and get F1 performance

Author: Christopher Bonham
Date: 16th February 2023
'''
import pandas as pd
import os
import logging
from utils.io import read_config, load_model, apply_model
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


def get_f1_score(df, lr, out_path):
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

    # Get model scores
    y_pred = apply_model(df, lr)

    # Extract labels
    y = df["exited"]

    # Get model F1 score
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

    # Get F1 score
    get_f1_score(
        df, lr,
        os.path.join(os.getcwd(), config["output_model_path"])
    )


# Top level script entry point
if __name__ == '__main__':
    main()
