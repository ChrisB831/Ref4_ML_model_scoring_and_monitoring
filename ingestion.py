import pandas as pd
import numpy as np
import os
import json
import logging
from datetime import datetime


# Create a logger
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def read_config(pth, display = False):
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

    if display == True:
        print(json.dumps(config, indent = 3))

    return config


def ingest_data(in_path, out_path):
    '''Ingest data, combine into a single dataset, Read in a config file (JSON format) and return a dictionary object

    Inputs:
        in_path (string)
            Path to data to ingest
        out_path (string)
            Path to write ingested data

    Outputs:
        None
    '''
    logger.info(f"ingestion.py: Input folder path: {in_path}")
    logger.info(f"ingestion.py: Output folder path: {out_path}")

    # Get a list of files in the input folder
    fnames = os.listdir(in_path)

    # Read in the csv files, combine into a single dedupled dataframe
    for idx, fname in enumerate(fnames):

        # Use the first csv file to set the df metadata
        if idx ==0:
            df = pd.read_csv(os.path.join(in_path, fname))
        else:
            _ = pd.read_csv(os.path.join(in_path, fname))
            df = pd.concat([df, _]).reset_index(drop=True)
    df = df.drop_duplicates()
    logger.info(f"ingestion.py: Input data ingested, shape: {df.shape}")

    # Write ingested df to csv
    # Create output directory if it doesnt exist
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    df.to_csv(os.path.join(out_path, "finaldata.csv"), index=False)
    logger.info(f"ingestion.py: Ingested data written to {os.path.join(out_path, 'finaldata.csv')}")

    # Write names of ingested files
    with open(os.path.join(out_path, "ingestedfiles"), "w") as fp:
        fp.write(str(fnames))
    logger.info(f"ingestion.py: Ingested file list written to {os.path.join(out_path, 'ingestedfiles.txt')}")



def main():
    '''Main functionality call

    Inputs:
        None
    Outputs:
        None
    '''

    # Get the configuration
#    config = read_config(".\config.json", display=True)
    config = read_config(".\config.json")
    logger.info("ingestion.py: Configuration file read")


    # Ingest the data
    ingest_data(
        os.path.join(os.getcwd(), config["input_folder_path"]),
        os.path.join(os.getcwd(), config["output_folder_path"]),
    )






    # output_folder_path = config["output_folder_path"]
    # logger.info(f"ingestion.py: output_folder_path: {output_folder_path}")




# Top level script entry point
if __name__ == '__main__':
    main()



