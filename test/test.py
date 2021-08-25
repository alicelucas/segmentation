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
    crop_size = config["crop_border"]
    model_path = config["model_filepath"]

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

    # x_image = Image.open(join(image_dir, filename))
    x_image = Image.open(filepath)
    x = numpy.asarray(x_image, dtype="float32")

    if len(x.shape) == 2:
        x = numpy.stack((x,) * 3, axis=-1)

    save_dir = config["test_save_dir"]
    if not exists(save_dir):
        makedirs(save_dir)

    # y_image = Image.open(join(target_dir, filename)) #Ground-truth label
    # y_pre = numpy.asarray(y_image, dtype="uint8")
    # y = preprocessing.convert_labels(y_pre)
    #

    #
    # # Visualize input image and ground-truth output
    # io.imsave(f"{save_dir}/{filename}", x[:, :, 0])
    # io.imsave(f"{save_dir}/y.{filenumber}.png", color.label2rgb(y, bg_label=0))

    # Make inference pass
    pretrained = config["use_saved_model"] #If we want to make a prediction using the whole trained model (trained by us), vs the random decoder head
    probs = forward_pass(x[numpy.newaxis, :, :, :], input_size, num_classes, crop_size, model_path=model_path, pretrained=pretrained)


    # Convert whole probability map to color mask for each example in image
    mask = preprocessing.prob_to_mask(probs[0])
    io.imsave(f'{save_dir}/mask.{filenumber}.png', mask)


def forward_pass(x, input_size, num_classes, crop_size, model_path="", dropout=False, pretrained=False):
    """
    Given input, forward pass through model. If needed, patch up image.
    :param input: (B, N, M, C) input (full image)
    :param input_size: the input size of the patch that goes to neural net
    :param pad_size: the pad size used to pad the input to avoid border effects
    :param num_classes
    :param pretrained: if False, then we use the random head decoder. If true, we use the unet.h5 weights that have been saved to home
    :return: The probability map of the size of the input image
    """
    print("Image shape:", x.shape)

    # Initialize model
    if pretrained:
        unet = models.load_model(model_path)
    else:
        unet = model.unet_model(numpy.array([x.shape[1], x.shape[2], x.shape[3]]))

    print(unet.summary())

    probs = numpy.zeros((1, x.shape[1] - 2 * crop_size, x.shape[2] - 2*crop_size,
                         num_classes))  # Probability map for the whole image

    # Extract patches over image since MobileNet expects 224x224 input
    start_row, start_col = 0 , 0 #Start index
    end_row, end_col = input_size, input_size #End index
    overflow_row, overflow_col = 0, 0

    #Iterate over image and send patches individually to MobileNet
    while start_row < x.shape[1] - 2 * crop_size:
        overflow_col = 0
        start_col = 0
        end_col = input_size
        while start_col < x.shape[2] - 2 * crop_size:
            #Pad if we are outside of boundary of image
            if end_col > x.shape[2] and end_row < x.shape[1]:
                overflow_col = end_col - x.shape[2]
                patch = numpy.pad(x[:, start_row: end_row, start_col: x.shape[2], :], ((0, 0), (0, 0), (0, overflow_col), (0, 0)))
                patch_prob = unet.predict(patch)  # forward pass
                probs[:, start_row:end_row - 2 * crop_size, start_col:end_col, :] = patch_prob[:, : input_size - overflow_row, : input_size - overflow_col - 2 * crop_size,:]
            elif end_row > x.shape[1] and end_col < x.shape[2]:
                overflow_row = end_row - x.shape[1]
                patch = numpy.pad(x[:, start_row: x.shape[1], start_col: end_col, :], ((0, 0), (0, overflow_row), (0, 0), (0, 0)))
                patch_prob = unet.predict(patch)  # forward pass
                probs[:, start_row:end_row, start_col:end_col - 2 * crop_size, :] = patch_prob[:, : input_size - overflow_row - 2 * crop_size, : input_size - overflow_col,:]
            elif end_row > x.shape[1] and end_col > x.shape[2]:
                overflow_row = end_row - x.shape[1]
                overflow_col = end_col - x.shape[2]
                patch = numpy.pad(x[:, start_row: x.shape[1], start_col: x.shape[2], :], ((0, 0), (0, overflow_row), (0, overflow_col), (0, 0)))
                patch_prob = unet.predict(patch)  # forward pass
                probs[:, start_row:end_row, start_col:end_col, :] = patch_prob[:, : input_size - overflow_row - 2 * crop_size, : input_size - overflow_col - 2 * crop_size,:]

            else:
                patch = x[:, start_row: end_row, start_col: end_col, :]  # extract patch
                patch_prob = unet.predict(patch)  # forward pass
                probs[:, start_row:end_row - 2 * crop_size, start_col:end_col - 2 * crop_size, :] = patch_prob[:,
                                                                                                    : input_size - overflow_row,
                                                                                                    : input_size - overflow_col,
                                                                                                    :]

            start_col += input_size - 2 * crop_size
            end_col = start_col + input_size
        start_row += input_size - 2 * crop_size
        end_row = start_row + input_size

    return probs