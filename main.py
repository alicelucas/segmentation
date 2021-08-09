from test import test
from train import train
import numpy
import sys

if __name__ == '__main__':

    #Image to do a forward pass on
    # image_dir = "data/maddox/images"
    # filename = "018.png"

    if sys.argv[1] == "test":
        if not sys.argv[2]:
            print("Please provide an image_dir and filename to test")
            exit(0)
        else:
            image_dir = sys.argv[2]
        if not sys.argv[3]:
            print("Please provide a filename to test")
            exit(0)
        else:
            filename = sys.argv[3]
            test.test_unet(image_dir, filename)

    elif sys.argv[1] == "train":
        train.train_unet()

    else:
        print("Please specify whether you would like to train or test the Unet model")


