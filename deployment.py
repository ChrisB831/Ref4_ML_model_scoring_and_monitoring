'''
Functionality to apply the model to new data

Author: Christopher Bonham
Date: 16th February 2023
'''
import pandas as pd
import os
import logging
from utils.io import read_config
import shutil



# Create a logger
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def deploy_artifacts_to_prod(model_path, ingested_files_path, deploy_path):
    '''Copy files to production
    trainedmodel.pkl
    latestscore.txt
    ingestedfiles.txt

    Inputs:
        model_path (string)
            Path to model directory
        ingested_files_path (string)
            Path to model directory
        deploy_path (string)
            Path to deployment (prod) directory
    Outputs:
        None
    '''
    logger.info(f"deployment.py: Model path: {model_path}")
    logger.info(f"deployment.py: Ingested files path: {ingested_files_path}")
    logger.info(f"deployment.py: Deployment (prod) path: {deploy_path}")

    # Create deployment directory if it doesnt exist
    if not os.path.exists(deploy_path):
        os.makedirs(deploy_path)

    # Copy files to production
    # WE MAY NEED TO MODIFY THIS WHEN WE MOVE TO LINUX
    shutil.copyfile(
        os.path.join(model_path, "latestscore.txt"),
        os.path.join(deploy_path, "latestscore.txt"),
    )
    shutil.copyfile(
        os.path.join(model_path, "trainedmodel.pkl"),
        os.path.join(deploy_path, "trainedmodel.pkl"),
    )
    shutil.copyfile(
        os.path.join(ingested_files_path, "ingestedfiles.txt"),
        os.path.join(deploy_path, "ingestedfiles.txt"),
    )

    logger.info(f"deployment.py: Model artifacts deployed to live")



def main():
    '''Main functionality call

    Inputs:
        None
    Outputs:
        None
    '''
    # Read the configuration file
    config = read_config(r".\config.json")
    logger.info("deployment.py: Configuration file read")


    # Copy relevant files to production
    deploy_artifacts_to_prod(
        os.path.join(os.getcwd(), config["output_model_path"]),
        os.path.join(os.getcwd(), config["output_folder_path"]),
        os.path.join(os.getcwd(), config["prod_deployment_path"])
    )


# Top level script entry point
if __name__ == '__main__':
    main()
