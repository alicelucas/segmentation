from os.path import join, exists
from os import makedirs

import numpy
from PIL import Image
from skimage import io, color

from models import model
from utils import preprocessing

from tensorflow.keras import models


def test_unet(config):
    """
    Given file path, do a forward pass
    :param config: configuration params
    :return: nothing. Saves output prediction to an "images" directory.
    """
    # Load input
    num_classes = 3

    filepath = config["test_filepath"]
    input_size = config["input_size"]
    pad_size = config["pad_size"]

    #Parse image dir and filename:
    slash = filepath.rfind("/")
    image_dir = filepath[:slash]
    filename = filepath[slash + 1:]

    #Parse filenumber (follows maddox dataset structure)
    nindex = filename.rfind(".")
    filenumber = filename[nindex-3:nindex]

    # Get list of input files and target mask
    # Here we assume directory structue is name/masks/*.png and name/images/*.png
    slash = image_dir.rfind("/")
    target_dir = join(image_dir[:slash], "masks")

    x_image = Image.open(join(image_dir, filename))
    x_grey = numpy.asarray(x_image, dtype="float32")
    x = numpy.stack((x_grey,) * 3, axis=-1)

    y_image = Image.open(join(target_dir, filename))
    y_pre = numpy.asarray(y_image, dtype="uint8")
    y = preprocessing.convert_labels(y_pre)

    save_dir = config["save_dir"]
    if not exists(save_dir):
        makedirs(save_dir)

    # Visualize input image and ground-truth output
    io.imsave(f"{save_dir}/{filename}", x[:, :, 0])
    io.imsave(f"{save_dir}/y.{filenumber}.png", color.label2rgb(y, bg_label=0))

    # Make inference pass
    pretrained = config["use_saved_model"] #If we want to make a prediction using an already trained model (trained by us)
    probs = forward_pass(x[numpy.newaxis, :, :, :], input_size, pad_size, num_classes, pretrained=pretrained)


    # Convert whole probability map to color mask for each example in image
    mask = preprocessing.prob_to_mask(probs[0])
    io.imsave(f'{save_dir}/mask.{filenumber}.png', mask)


def forward_pass(x, input_size, pad_size, num_classes, pretrained=False):
    """
    Given input, forward pass through model. If needed, patch up image.
    :param input: (B, N, M, C) input (full image)
    :param input_size: the input size of the patch that goes to neural net
    :param pad_size: the pad size used to pad the input to avoid border effects
    :param num_classes
    :param pretrained: if False, then we use the random head decoder. If true, we use the unet.h5 weights that have been saved to home
    :return: The probability map of the size of the input image
    """
    # Initialize model
    if pretrained:
        unet = models.load_model('./unet.h5')
    else:
        unet = model.unet_model()


    # Code below prepares patch extraction process for inference when testing
    patch_size = input_size - 2 * pad_size  # account for padding

    # number of patches in row and column directions
    n_row = x.shape[2] // patch_size
    n_col = x.shape[1] // patch_size

    # pad whole image so that we can account for border effects
    pad_row = int(numpy.floor((n_row + 1) * patch_size - x.shape[2]) / 2)
    pad_col = int(numpy.floor((n_col + 1) * patch_size - x.shape[1]) / 2)

    im = numpy.pad(x, ((0, 0), (pad_col, pad_col), (pad_row, pad_row), (0, 0)))

    probs = numpy.zeros((1, (n_col + 1) * patch_size, (n_row + 1) * patch_size,
                         num_classes))  # Probability map for the whole image

    # Extract patches over image
    for i in range(n_row + 1):
        print(f"Prediction row {i} out of {n_row} rows.")
        for j in range(n_col + 1):
            patch = im[:, patch_size * i:patch_size * (i + 1), patch_size * j:patch_size * (j + 1),
                    :]  # extract patch
            patch = numpy.pad(patch, ((0, 0), (8, 8), (8, 8), (0, 0)))  # add padding
            patch_prob = unet.predict(patch)  # forward pass
            probs[:, patch_size * i:patch_size * (i + 1), patch_size * j:patch_size * (j + 1), :] = patch_prob[:,
                                                                                                    pad_size: input_size - pad_size,
                                                                                                    pad_size: input_size - pad_size,
                                                                                                    :]

    # Crop probability mask to original image size
    probs = probs[:, pad_col:-pad_col, pad_row:-pad_row, :]

    return probs