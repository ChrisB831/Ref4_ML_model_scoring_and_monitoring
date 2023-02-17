'''
Functionality for Flask app

Author: Christopher Bonham
Date: 17th February 2023
'''
import os
import pandas as pd
from flask import Flask, request
from utils.io import read_config
from diagnostics import model_predictions, load_training_data, \
                        dataframe_summary, missing_data, execution_time, \
                        outdated_packages_list


# Instantiate app instance
app = Flask(__name__)


# Read the configuration file
global config
config = read_config(r".\config.json")


@app.route("/prediction")
def prediction_ep():
    '''Prediction endpoint takes the csv (path defined as a query parameter)
    and applies the production model to it

    Example call
    NB We dont need the double quotes here
    http://127.0.0.1:8000/prediction?fname=G:\\My Drive\\Work\\002 Code store\\
        Python\\PyCharm\\Udacity\\p4\\testdata\\testdata.csv

    Inputs:
        None
    Outputs:
        Str
            Model predictions
    '''
    # Get the query parameter and load the coresponding data
    fname = request.args.get('fname')
    df = pd.read_csv(fname)

    # Get model prediction using the production model
    preds = model_predictions(df, config)

    return str(preds)


@app.route("/scoring")
def scoring_ep():
    '''Scoring endpoint that runs the scoring.py script and return the F1 score

    Example call
    http://127.0.0.1:8000/scoring

    Inputs:
        None
    Outputs:
        string
            F1 score
    '''
    # Run the script
    os.system("python scoring.py")

    # Get the F1 score from the file and return it
    with open(os.path.join(os.getcwd(),
                           config["output_model_path"],
                           "latestscore.txt")) as fp:
        f1 = fp.read()
    return str(f1)


@app.route("/summarystats")
def summarystats_ep():
    '''Summary stats endpoint that runs the dataframe_summary
    function from the diagnostics.py

    Example call
    http://127.0.0.1:8000/summarystats

    Inputs:
        None
    Outputs:
        str
            Summary statistics
    '''

    # Load training data
    df = load_training_data(
        os.path.join(os.getcwd(), config["output_folder_path"])
    )

    # Get summary statistics
    summary_stats = dataframe_summary(df)

    return summary_stats


@app.route("/diagnostics")
def diagnostics_ep():
    '''Summary stats endpoint that runs the following functions
    from diagnostics.py
    missing_data
    execution_time
    outdated_packages_list

    Example call
    http://127.0.0.1:8000/diagnostics

    Inputs:
        None
    Outputs:
        str
            Summary statistics
    '''

    # Load training data
    df = load_training_data(
        os.path.join(os.getcwd(), config["output_folder_path"])
    )

    # Get missing values
    missing_values = missing_data(df)

    # Time the ingestion.py and training.py modules
    timings = execution_time()

    # Get a table of all outdated packages
    # NB this string has \n characters as we are printing html
    # we need to replace with <br>
    outdated_packages = outdated_packages_list()

    # Create and return response body
    rb = f"Missing values:<br>{missing_values}" \
         f"<br><br>Timings:<br>{timings}" \
         f"<br><br>Outdated packages:<br>{outdated_packages}"
    return rb


# Top level script entry point
# Run the app on a local development server
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
