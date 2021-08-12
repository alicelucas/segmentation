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

    # x_image = Image.open(join(image_dir, filename))
    x_image = Image.open(filepath)
    x = numpy.asarray(x_image, dtype="float32")

    if len(x.shape) == 2:
        x = numpy.stack((x,) * 3, axis=-1)

    #FIXME Uncomment this when you are done with your overfitting experiment

    #
    # y_image = Image.open(join(target_dir, filename))
    # y_pre = numpy.asarray(y_image, dtype="uint8")
    # y = preprocessing.convert_labels(y_pre)
    #
    # save_dir = config["save_dir"]
    # if not exists(save_dir):
    #     makedirs(save_dir)
    #
    # # Visualize input image and ground-truth output
    # io.imsave(f"{save_dir}/{filename}", x[:, :, 0])
    # io.imsave(f"{save_dir}/y.{filenumber}.png", color.label2rgb(y, bg_label=0))

    # Make inference pass
    pretrained = config["use_saved_model"] #If we want to make a prediction using an already trained model (trained by us)
    probs = forward_pass(x[numpy.newaxis, :, :, :], input_size, pad_size, num_classes, pretrained=pretrained)

    print(probs.shape)

    #FIXME : Comment this when done with overfitting experiment (Save dir is defined above)
    save_dir = config["save_dir"]
    if not exists(save_dir):
        makedirs(save_dir)


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
    # Code below prepares patch extraction process for inference when testing
    patch_size = input_size - 2 * pad_size  # account for padding

    # number of patches in row and column directions
    n_row = (x.shape[2] // input_size)
    n_col = (x.shape[1] // input_size)

    im = numpy.pad(x, ((0, 0), (pad_size, pad_size), (pad_size, pad_size), (0, 0)))

    print("Image shape:", im.shape)


    # Initialize model
    if pretrained:
        unet = models.load_model('./unet.h5')
    else:
        unet = model.unet_model(numpy.array([im.shape[1], im.shape[2], im.shape[3]]))


    probs = numpy.zeros((1, n_col * input_size, n_row * input_size,
                         num_classes))  # Probability map for the whole image

    # Extract patches over image
    for i in range(n_row):
        print(f"Prediction row {i} out of {n_row} rows.")
        for j in range(n_col):
            patch = im[:, pad_size + input_size * i:input_size * (i + 1) - pad_size, pad_size + input_size * j:input_size * (j + 1) - pad_size,
                    :]  # extract center patch
            patch = numpy.pad(patch, ((0, 0), (pad_size, pad_size), (pad_size, pad_size), (0, 0)))  # add padding
            patch_prob = unet.predict(patch)  # forward pass
            print("Patch prob", patch_prob.shape)
            probs[:, patch_size * i:patch_size * (i + 1), patch_size * j:patch_size * (j + 1), :] = patch_prob[:,
                                                                                                    pad_size: input_size - pad_size,
                                                                                                    pad_size: input_size - pad_size,
                                                                                                    :]


    return probs