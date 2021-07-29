import numpy
from models import model

def forward_pass_full_image(x, model_name, num_classes):
    """
    Given input, forward pass through model. If needed, patch up image.
    :param input: (B, N, M, C) input
    :param model: string for model name
    :param num_classes
    :return: The probability map of the size of the input image
    """

    if model_name == "unet":
        # Initialize model
        unet = model.unet_model()

        # Code below prepares patch extraction process for inference when testing
        pad_size = 8
        input_size = 224 #MobileNet encoder expects 225 for input size
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