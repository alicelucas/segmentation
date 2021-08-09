from test import test
from train import train
import numpy
import sys

if __name__ == '__main__':

    #For testing:
    #python main.py test data/maddox/images/018.png
    #For training
    #python main.py train

    if sys.argv[1] == "test":
        if not sys.argv[2]:
            print("Please provide an image file path to test")
            exit(0)
        else:
            filepath = sys.argv[2]
            test.test_unet(filepath)

    elif sys.argv[1] == "train":
        train.train_unet()

    else:
        print("Please specify whether you would like to train or test the Unet model")


