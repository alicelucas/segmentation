from utils import preprocessing
from models import model
from skimage import io
import numpy
import sys

if __name__ == '__main__':
    """
    This simple script accepts an image filename as input, patches it up in the desired patch size, and feeds
    each patch to a UNet model (with a pre-trained MobileNet backbone but random decoded)
    """

    filename = sys.argv[1]
    im = io.imread(filename, as_gray=True)
    im = preprocessing.normalize(im) #Convert to float and range [0, 1]

    unet = model.unet_model()

    #Code below prepares patch extraction process
    pad_size = 8
    input_size = 224
    patch_size = input_size - 2 * pad_size # account for padding

    # number of patches in row and column directions
    n_row = im.shape[1] // patch_size
    n_col = im.shape[0] // patch_size

    #pad whole image so that we can account for border effects
    pad_row = int(numpy.floor((n_row + 1) * patch_size - im.shape[1]) / 2)
    pad_col = int(numpy.floor((n_col + 1) * patch_size - im.shape[0]) / 2)

    im = numpy.pad(im, (pad_col, pad_row))

    probs = numpy.zeros((1, (n_col + 1) * patch_size, (n_row + 1) * patch_size, 3)) #Probability map for the whole image

    #Extract patches over image
    for i in range(n_row + 1):
        print(f"Prediction row {i} out of {n_row} rows.")
        for j in range(n_col + 1):
            patch = im[patch_size*i:patch_size*(i+1), patch_size*j:patch_size*(j+1)] #extract patch
            patch = numpy.pad(patch, (8, 8)) #add padding
            x = preprocessing.to_tensor(patch) #convert to shape appropriate to tensorflow
            patch_prob = unet.predict(x) #forward pass
            probs[0, patch_size*i:patch_size*(i+1), patch_size*j:patch_size*(j+1), :] = patch_prob[:, pad_size: input_size - pad_size, pad_size: input_size - pad_size, :]


    #Crop probability mask to original image size
    probs = probs[:, pad_col:-pad_col, pad_row:-pad_row, :]
    #Convert whole probability map to color mask
    mask = preprocessing.create_mask(probs)
    io.imsave("mask.png", mask)
