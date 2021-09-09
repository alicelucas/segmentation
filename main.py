from test import test
from train import train
from utils.load import load_config
import sys

if __name__ == '__main__':

    config_filename = sys.argv[1]
    config = load_config(config_filename)

    if not config["train"]:
        test.test_unet(config)

    else:
        train.train_unet(config)
