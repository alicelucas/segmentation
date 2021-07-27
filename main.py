from os import listdir
from os.path import isfile, join

import numpy
from skimage import io, color

from models import model
from preprocessing import Cells
from utils import preprocessing

if __name__ == '__main__':

    #Load preprocessing
    batch_size = 4
    img_size = 1024
    num_classes = 3

    #Get list of input files and target masks
    image_dir = "data/maddox/images"
    target_dir = "data/maddox/masks"
    input_img_paths = [join(image_dir, f) for f in listdir(image_dir) if isfile(join(image_dir, f))]
    target_paths = [join(target_dir, f) for f in listdir(target_dir) if isfile(join(target_dir, f))]


    data = Cells.CellsSequence(input_img_paths, target_paths, batch_size, img_size)
    idx = 12
    x, y = data.__getitem__(idx)
    filenames = data.map_filename_indices(idx)

    #Visualize input image and ground-truth output
    for i in range(x.shape[0]):
        io.imsave(f"./images/x.{filenames[i]}.png", x[i] * 255)
        # print(y.shape)
        # print(color.label2rgb(y[0,:,:,0]).shape)
        io.imsave(f"./images/y.{filenames[i]}_tmp.png", color.label2rgb(y[i,:,:,0]))

    #Initialize model
    unet = model.unet_model()

    #Code below prepares patch extraction process
    pad_size = 8
    input_size = 224
    patch_size = input_size - 2 * pad_size # account for padding

    # number of patches in row and column directions
    n_row = x.shape[2] // patch_size
    n_col = x.shape[1] // patch_size

    #pad whole image so that we can account for border effects
    pad_row = int(numpy.floor((n_row + 1) * patch_size - x.shape[2]) / 2)
    pad_col = int(numpy.floor((n_col + 1) * patch_size - x.shape[1]) / 2)

    im = numpy.pad(x, ((0, 0), (pad_col, pad_col), (pad_row, pad_row), (0, 0)))

    probs = numpy.zeros((batch_size, (n_col + 1) * patch_size, (n_row + 1) * patch_size, num_classes)) #Probability map for the whole image

    #Extract patches over image
    for i in range(n_row + 1):
        print(f"Prediction row {i} out of {n_row} rows.")
        for j in range(n_col + 1):
            patch = im[:, patch_size*i:patch_size*(i+1), patch_size*j:patch_size*(j+1), :] #extract patch
            patch = numpy.pad(patch, ((0, 0), (8, 8), (8, 8), (0, 0))) #add padding
            patch_prob = unet.predict(patch) #forward pass
            probs[:, patch_size*i:patch_size*(i+1), patch_size*j:patch_size*(j+1), :] = patch_prob[:, pad_size: input_size - pad_size, pad_size: input_size - pad_size, :]


    #Crop probability mask to original image size
    probs = probs[:, pad_col:-pad_col, pad_row:-pad_row, :]

    #Convert whole probability map to color mask
    for i in range(probs.shape[0]):
        mask = preprocessing.prob_to_mask(probs[i])
        io.imsave(f'images/mask.{filenames[i]}.png', mask)
