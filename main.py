from test import test
from train import train
import numpy

if __name__ == '__main__':

    #Image to do a forward pass on
    image_dir = "data/maddox/images"
    filename = "018.png"

    # test.test_unet(image_dir, filename)
    train.train_unet()