from test import test
from train import train
from utils.load import load_config
import sys
from utils.to_semantic import decode_DSB_test_set

if __name__ == '__main__':

    config_filename = sys.argv[1]
    config = load_config(config_filename)

    #TMP
    decode_DSB_test_set()
    #ENd of tmp code
    exit(0)

    if not config["train"]:
        test.test_unet(config)

    else:
        train.train_unet(config)
