'''
Functionality to execute data diagnostics

Author: Christopher Bonham
Date: 16th February 2023
'''
import pandas as pd
import os
import timeit
import logging
import subprocess
from utils.io import read_config, load_model, apply_model


# Create a logger
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def model_predictions(df, config):
    '''Get predictions using the deployed production model

    Inputs:
        df (Pandas.datafrane)
            Data to score
        config (Dict)
            Configuration
    Outputs:
        list
            Model scores
    '''
    # Load deployed model
    lr = load_model(
        os.path.join(os.getcwd(), config["prod_deployment_path"])
    )

    # Get predictions (this must be a list)
    preds = list(apply_model(df, lr))

    return preds


def load_training_data(in_path):
    '''Load training data to dataframe
    Inputs:
        in_path (string)
            Path to training data

    Outputs:
        pandas.dataframe
            Training data
    '''
    logger.info(f"diagnostics.py: Input folder path: {in_path}")

    # Load training data
    df = pd.read_csv(os.path.join(in_path, "finaldata.csv"))
    logger.info(f"diagnostics.py: Training data shape: {df.shape}")

    return df


def dataframe_summary(df):
    '''Get summary stats (mean, media, stddev for numeric fields in dataframe

    Inputs:
        df (Pandas.datafrane)
            Dataframe to get statistics
    Outputs:
        list
            Dataframe statistics
            Format is [mean, median, standard deviation] repeated for
            every numeric field
    '''
    # Get a list of all the numeric columns in the dataframe
    numeric_vnames = list(df.select_dtypes(include='number').columns.values)
    logger.info(f"diagnostics.py: Numeric fields are: {numeric_vnames}")

    # Get summary statistics list
    summary_stats = []
    for var in numeric_vnames:
        summary_stats.append(df[var].mean())
        summary_stats.append(df[var].median())
        summary_stats.append(df[var].std())

    return summary_stats


def missing_data(df):
    '''Get the percentage of missing values for eack field

    Inputs:
        df (Pandas.datafrane)
            Dataframe to get statistics
    Outputs:
        list
            Percentage of missing values
    '''
    logger.info(f"diagnostics.py: Fields are: {list(df.columns.values)}")
    return list(df.isnull().sum() / df.shape[0])


def execution_time():
    '''Time the ingestion.py and training.py modules

    Inputs:
        None
    Outputs:
        list
            Training times in seconds
    '''

    # Time the ingestion.py script
    starttime = timeit.default_timer()
    os.system("python ingestion.py")
    runtime_ingestion = timeit.default_timer() - starttime

    # Time the training.py script
    starttime = timeit.default_timer()
    os.system("python training.py")
    runtime_training = timeit.default_timer() - starttime

    return [runtime_ingestion, runtime_training]


def outdated_packages_list():
    '''Get a list of all outdated packages

    Inputs:
        None
    Outputs:
        str
            Outdated packages
    '''
    response = subprocess.run(["python", "-m", "pip", "list", "--outdated"],
                              capture_output=True).stdout

    response = response.decode("utf-8")
    return(response)
    logger.info(f"diagnostics.py: Outdated dependencies are:\n{response_decoded}")


def main():
    '''Main functionality call

    Inputs:
        None
    Outputs:
        None
    '''
    # Read the configuration file
    config = read_config(r".\config.json")
    logger.info("diagnostics.py: Configuration file read")

    # Load training data
    df = load_training_data(
        os.path.join(os.getcwd(), config["output_folder_path"])
    )

    # Get summary statistics
    summary_stats = dataframe_summary(df)
    logger.info(f"diagnostics.py: Summary statistics [mean, medias, sd] are"
                f" {summary_stats}")

    # Get missing values
    missing_values = missing_data(df)
    logger.info(f"diagnostics.py: Missing value percentages are:"
                f" {missing_values}")

    # Time the ingestion.py and training.py modules
    timings = execution_time()
    logger.info(f"diagnostics.py: Runtime for ingestion and training "
                f"modules are: {timings}")

    # Get a table of all outdated packages
    outdated_packages = outdated_packages_list()


# Top level script entry point
if __name__ == '__main__':
    main()
