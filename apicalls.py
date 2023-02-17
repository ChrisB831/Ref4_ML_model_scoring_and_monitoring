'''
Functionality to call the API

NB For this code to execute the app MUST be running
python app.py

Author: Christopher Bonham
Date: 17th February 2023
'''
import requests
import os
import logging
from utils.io import read_config

# Domain of dev server
domain_dev = "http://127.0.0.1:8000"


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
    logger.info("apicalls.py: Configuration file read")

    with open(os.path.join(os.getcwd(), config["output_model_path"],
                           "apireturns.txt"), "w") as fp:

        # Call the prediction endpoint and write the status code and
        # response body to file
        pth = os.path.join(os.getcwd(), "testdata", "testdata.csv")
        url = f"{domain_dev}/prediction?fname={pth}"
        rc = requests.get(url)
        fp.write("Prediction endpoint")
        fp.write(f"\nStatus code: {rc.status_code}")
        fp.write(f"\n{str(rc.content)}")

        # Call the scoring endpoint and print the status code and response body
        url = f"{domain_dev}/scoring"
        rc = requests.get(url)
        fp.write("\n\nScoring endpoint")
        fp.write(f"\nStatus code: {rc.status_code}")
        fp.write(f"\n{str(rc.content)}")

        # Call the summarystats endpoint and print the status code
        # and response body
        url = f"{domain_dev}/summarystats"
        rc = requests.get(url)
        fp.write("\n\nSummary stats endpoint")
        fp.write(f"\nStatus code: {rc.status_code}")
        fp.write(f"\n{str(rc.content)}")

        # Call the diagnostics endpoint and print the status code and
        # response body
        url = f"{domain_dev}/diagnostics"
        rc = requests.get(url)
        fp.write("\n\nDiagnostics endpoint")
        fp.write(f"\nStatus code: {rc.status_code}")
        fp.write(f"\n{str(rc.content)}")


# Top level script entry point
if __name__ == '__main__':
    main()
