'''
Utility (auxiliary) functions

Author: Christopher Bonham
Date: 16th February 2023
'''
import json


def read_config(pth, display=False):
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

    if display:
        print(json.dumps(config, indent=3))

    return config
