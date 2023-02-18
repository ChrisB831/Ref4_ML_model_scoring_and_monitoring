'''
Functionality to automate the ML scoring and monitoring

Warning for some reson sklearn turns off the logging functionality?


Author: Christopher Bonham
Date: 18th February 2023
'''
import logging
import ast
import os
import pandas as pd
import ingestion
from scoring import get_f1_score
import training
import deployment
import reporting
from utils.io import read_config, load_model


# Create a logger
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def main():
    '''Main functionality call

    Inputs:
        None
    Outputs:
        None
    '''
    # Read the configuration file
    config = read_config(r".\config.json")
    logger.info("fullprocess.py: Configuration file read")

    # Get the names of the previously ingested files from production
    fpath = os.path.join(os.getcwd(),
                         config["prod_deployment_path"],
                         "ingestedfiles.txt")
    with open(fpath, 'r') as fp:
        prev_files = ast.literal_eval(fp.read())
    logger.info(f"fullprocess.py: Previously ingested files were {prev_files}")

    # Get a list of files in the input folder
    path = os.path.join(os.getcwd(), config["input_folder_path"])
    input_files = os.listdir(path)
    logger.info(f"fullprocess.py: Input files to ingest are {input_files}")

    # Get a list of new files
    new_files = list(set(input_files) - set(prev_files))
    logger.info(f"fullprocess.py: New files to ingest are {new_files}")

    # Run only if there are new files to export
    if (len(new_files) > 0):
        logger.info("fullprocess.py: There are new files to ingest")

        # Ingest new data
        ingestion.main()

        # Read in F1 score from deployed model
        fpath = os.path.join(os.getcwd(),
                             config["prod_deployment_path"],
                             "latestscore.txt")
        with open(fpath, 'r') as fp:
            live_f1 = ast.literal_eval(fp.read())
        logger.info(f"fullprocess.py: F1 score of live model {live_f1}")

        # Get F1 score with the new data using deployment model
        lr = load_model(
            os.path.join(os.getcwd(), config["prod_deployment_path"])
        )
        df = pd.read_csv(os.path.join(os.getcwd(),
                                      config["output_folder_path"],
                                      "finaldata.csv"))

        get_f1_score(
            df, lr,
            os.path.join(os.getcwd(), config["output_model_path"])
        )

        with open(os.path.join(os.getcwd(),
                               config["output_model_path"],
                               "latestscore.txt"), 'r') as fp:
            new_f1 = ast.literal_eval(fp.read())
        logger.info(f"fullprocess.py: F1 score of new model {new_f1}")

        # Rebuild only if model drift has occured
        if new_f1 < live_f1:
            logger.info("fullprocess.py: Model drift has occurred")

            # Retrain
            logger.info("Retrain model with new data")
            training.main()

            # Redeploy
            logger.info("Deploy new model into live")
            deployment.main()

            # Reporting
            logger.info("Run reporting")
            reporting.main()

        else:
            logger.info("fullprocess.py: Model drift has NOT occurred")
    else:
        logger.info("fullprocess.py: No files to ingest")


# Top level script entry point
if __name__ == '__main__':
    main()
