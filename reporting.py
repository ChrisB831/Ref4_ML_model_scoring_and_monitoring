'''
Functionality to plot performance

Author: Christopher Bonham
Date: 17th February 2023
'''
import pandas as pd
import os
import logging
from utils.io import read_config, load_model, apply_model
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt


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


def plot_conf_mat(df, lr, out_path):
    '''Load test data to dataframe
    Inputs:
        df (Pandas.datafrane)
            Data to score
        lr (sklearn.linear_model._logistic.LogisticRegression)
            Logistic regression model
        out_path (string)
            Path to store confusion matrix plot
    Outputs:
        None
    '''
    logger.info(f"scoring.py: Output folder path: {out_path}")

    # Get model scores
    y_pred = apply_model(df, lr)

    # Extract labels
    y = df["exited"]

    # Get confusion matrix
    cm = confusion_matrix(y, y_pred)
    print(y)
    print(y_pred)
    print(type(cm))
    print(cm)

    # Plot confusion matrix
    plt.matshow(cm, cmap=plt.cm.Greens)
    plt.title("Confusion matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    # plt.show()

    # Save plot to file
    plt.savefig(os.path.join(out_path, 'confusionmatrix.png'))


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

    # Get the confusion matrix plot
    plot_conf_mat(
        df, lr,
        os.path.join(os.getcwd(), config["output_model_path"])
    )


# Top level script entry point
if __name__ == '__main__':
    main()
