from os.path import join
import yaml

CONFIG_PATH = "../config"

def load_config(config_name):

    with open(join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config