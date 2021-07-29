from os import listdir
from os.path import isfile, join

from skimage import io, color

from test import test
from utils import preprocessing

from PIL import Image

import numpy

if __name__ == '__main__':

    #Load input
    batch_size = 4
    img_size = 1024
    num_classes = 3

    #Get list of input files and target masks
    image_dir = "data/maddox/images"
    target_dir = "data/maddox/masks"
    input_img_paths = [join(image_dir, f) for f in listdir(image_dir) if isfile(join(image_dir, f))]
    target_paths = [join(target_dir, f) for f in listdir(target_dir) if isfile(join(target_dir, f))]

    filename = "020.png"
    x_image = Image.open(join(image_dir, "x." + filename))
    x_grey = numpy.asarray(x_image, dtype="float32")
    x = numpy.stack((x_grey,) * 3, axis=-1)

    y_image = Image.open(join(target_dir, "x." + filename))
    y_pre = numpy.asarray(y_image, dtype="uint8")
    y = preprocessing.convert_labels(y_pre)


    #Visualize input image and ground-truth output
    io.imsave(f"./images/x.{filename}", x[:, :, 0])
    io.imsave(f"./images/y.{filename}", color.label2rgb(y))

    # Make inference pass
    probs = test.forward_pass_full_image(x[numpy.newaxis, :, :, :], "unet", num_classes)

    # Convert whole probability map to color mask for each example in image
    mask = preprocessing.prob_to_mask(probs[0])
    io.imsave(f'images/mask.{filename}', mask)