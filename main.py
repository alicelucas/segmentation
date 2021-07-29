from os import listdir
from os.path import isfile, join

from skimage import io, color

from input import Cells
from test import test
from train import train
from utils import preprocessing

if __name__ == '__main__':

    #Load input
    batch_size = 4
    img_size = 1024
    num_classes = 3
    patch_size = 224

    #Get list of input files and target masks
    image_dir = "data/maddox/images"
    target_dir = "data/maddox/masks"
    input_img_paths = [join(image_dir, f) for f in listdir(image_dir) if isfile(join(image_dir, f))]
    target_paths = [join(target_dir, f) for f in listdir(target_dir) if isfile(join(target_dir, f))]

    test.forward_pass("unet", num_classes)
    #FIXME Make code above work before attempting train.train()

    # train.train()
    #
    # exit()

    data = Cells.CellsGenerator(input_img_paths, target_paths, batch_size, patch_size, image_size=img_size)
    idx = 11
    x, y = data.__getitem__(idx)
    filenames = data.map_filename_indices(idx)

    #Visualize input image and ground-truth output
    for i in range(x.shape[0]):
        io.imsave(f"./images/x.{filenames[i]}.png", x[i])
        # print(y.shape)
        # print(color.label2rgb(y[0,:,:,0]).shape)
        io.imsave(f"./images/y.{filenames[i]}_tmp.png", color.label2rgb(y[i,:,:,0]))

    # Make inference pass
    probs = test.forward_pass(x, "unet", batch_size, num_classes)

    # Convert whole probability map to color mask for each example in image
    for i in range(probs.shape[0]):
        mask = preprocessing.prob_to_mask(probs[i])
        io.imsave(f'images/mask.{filenames[i]}.png', mask)
