import yaml
import os
import logging

def load_config(config_name):

    #DEBUG WORK
    files = [f for f in os.listdir('.') if os.path.isfile(f)]
    for f in files:
        logging.error(f)

    with open(config_name) as file:
        config = yaml.safe_load(file)

    return config