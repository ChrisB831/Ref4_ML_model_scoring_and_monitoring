'''
Functionality to ingest the development data

Author: Christopher Bonham
Date: 16th February 2023
'''
import pandas as pd
import os
import logging
from utils.io import read_config


# Create a logger
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def ingest_data(in_path, out_path):
    '''Ingest training data, combine into a single dataset and write to csv

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
        if idx == 0:
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
    logger.info(f"ingestion.py: Ingested data written to "
                f"{os.path.join(out_path, 'finaldata.csv')}")

    # Write names of ingested files
    with open(os.path.join(out_path, "ingestedfiles.txt"), "w") as fp:
        fp.write(str(fnames))
    logger.info(f"ingestion.py: Ingested file list written to"
                f" {os.path.join(out_path, 'ingestedfiles.txt')}")


def main():
    '''Main functionality call

    Inputs:
        None
    Outputs:
        None
    '''
    # Read the configuration file
    config = read_config(r".\config.json")
    logger.info("ingestion.py: Configuration file read")

    # Ingest the data
    ingest_data(
        os.path.join(os.getcwd(), config["input_folder_path"]),
        os.path.join(os.getcwd(), config["output_folder_path"]),
    )


# Top level script entry point
if __name__ == '__main__':
    main()
