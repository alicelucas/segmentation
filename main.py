from test import test
from train import train
from utils.load import load_config

if __name__ == '__main__':

    config = load_config("config.yaml")

    if not config["train"]:
        test.test_unet(config)

    else:
        train.train_unet(config)
