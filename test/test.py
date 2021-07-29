import numpy
from models import model
from input import Cells
from utils import preprocessing
from skimage import io, color

def forward_pass(model_name, num_classes):
    """
    Given input, forward pass through model. If needed, patch up image.
    :param input: (B, N, M, C) input
    :param model: string for model name
    :param batch_size
    :param num_classes
    :return: The probability map of the size of the input image
    """

    if model_name == "unet":
        # Initialize model
        unet = model.unet_model()

        #Test image
        input_img_paths = ["data/maddox/images/x.016.png"]
        target_paths = ["data/maddox/masks/x.016.png"]

        batch_size = 1
        patch_size = 224
        image_size = 1024
        pad_size = 8

        data = Cells.CellsGenerator(input_img_paths, target_paths, batch_size, patch_size, image_size, pad_size)

        filenames = data.map_filename_indices(0)

        for i, patch in enumerate(data.x_patches):
            patch_prob = unet.predict(patch[numpy.newaxis, :, :, :])  # forward pass
            patch_prob = patch_prob[:, pad_size: patch.shape[1] - pad_size, pad_size: patch.shape[2] - pad_size, :]

            mask = preprocessing.prob_to_mask(patch_prob[0])
            io.imsave(f'images/mask.{filenames[0]}_{i}.png', mask)

