import yaml

def load_config(config_name):

    with open(config_name) as file:
        config = yaml.safe_load(file)

    return config