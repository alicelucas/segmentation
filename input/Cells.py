import random

import numpy as np
from PIL import Image
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from utils import preprocessing


class CellsGenerator(keras.utils.Sequence):
    """
    Helper class to iterate over the input (from file paths to numpy arrays)
    """

    def __init__(self, x_paths, y_paths, batch_size, patch_size, should_augment):
        self.do_augment = should_augment

        self.batch_size = batch_size
        self.patch_size = patch_size

        self.pad_size = 8

        self.x_paths = []
        self.y_paths = []

        self.x_patches, self.y_patches = self.create_patches(x_paths, y_paths)



        #Randomize the dataset here
        random.Random(1337).shuffle(self.x_patches)
        random.Random(1337).shuffle(self.y_patches)
        random.Random(1337).shuffle(self.x_paths)
        random.Random(1337).shuffle(self.y_paths)



    def __len__(self):
        """
        Returns number of batches
        :return:
        """
        #FIXEME
        return 1
        # return len(self.x_patches) // self.batch_size


    def create_patches(self, x_paths, y_paths):
        """
        :param x_paths: list of paths of image
        :param y_paths: list of paths of label masks
        :return: list of patches of size 224 x 224
        """
        # Code below prepares patch extraction process for inference when testing
        patch_size = self.patch_size - 2 * self.pad_size  # account for padding

        x_patches = []
        y_patches = []

        pad_size = 8

        ##FIXME the overwriting of x_paths is for the overfitting experiment
        x_paths = ["data/maddox/images/x.018.png"]
        y_paths = ["data/maddox/masks/x.018.png"]

        # Extract image shape by reading shape of first image
        x_image = Image.open(x_paths[0])
        image_rows = np.asarray(x_image, dtype="float32").shape[0]
        image_cols = np.asarray(x_image, dtype="float32").shape[1]

        for idx, x_path in enumerate(x_paths):
            baz = load_img(x_path, target_size=(image_rows, image_cols))
            x = np.array(img_to_array(baz), dtype="float32")

            foo = load_img(y_paths[idx], target_size=(image_rows, image_cols))
            data = np.array([img_to_array(foo)], dtype="uint8")
            mask = preprocessing.convert_labels(data[0])

            y = np.expand_dims(mask, 2)

            if self.do_augment:
                x, y = self.augment((x, y))  # Place augmented patch in batch

            # number of column directions
            n_row = x.shape[1] // (patch_size + 2 * pad_size)
            n_col = x.shape[0] // (patch_size + 2 * pad_size)

            print(n_row, n_col)
            # Extract patches over image
            for i in range(n_row):
                for j in range(n_col):

                    #Extract y-patch
                    patch = y[patch_size * i:patch_size * (i + 1), patch_size * j:patch_size * (j + 1),
                            :]  # extract patch

                    #Only keep the patch if it has non-zero labels (i.e., not just black)
                    if 1 not in patch[:, :, 0]:
                        continue

                    patch = np.pad(patch, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)))  # add padding
                    y_patches.append(patch)

                    # Extract x-patch
                    patch = x[patch_size * i:patch_size * (i + 1), patch_size * j:patch_size * (j + 1), :]  #extract patch
                    patch = np.pad(patch, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)))  # add padding
                    x_patches.append(patch)

                    # keep track of which patch belongs to which image
                    self.x_paths.append(x_paths[idx])
                    self.y_paths.append(y_paths[idx])

        return x_patches, y_patches


    def map_filename_indices(self, idx):
        """
        Given an idx that for the data point and the batch size, return the indices in the original filenames.
        This is necessary since the data sequence was shuffled randomly, so index 0 does not actually correspond to file 0.
        :param idx:
        :return: the list of actual indices corresponding to the filenames selected for that batch.
        """
        batch_x_paths = self.x_paths[idx: idx + self.batch_size]
        names = []
        for path in batch_x_paths:
            a, b = path.split(".png")
            names.append(a[-3:])
        return names


    def augment(self, x_and_y):
        """
        Given a batch of images and their gt mask, augment it by flipping it (horizontally or vertically), and doing a rotation
        :return: the augmented batch of data
        """
        x_and_y = preprocessing.flip(x_and_y)
        return preprocessing.rotate(x_and_y)


    def __getitem__(self, batch_idx):
        """Return (input, target) numpy array corresponding to batch idx"""


        x_patches = self.x_patches[batch_idx * self.batch_size: batch_idx * self.batch_size + self.batch_size]
        y_patches = self.y_patches[batch_idx * self.batch_size: batch_idx * self.batch_size + self.batch_size]

        #FIXME Warning! Uncomment the following. This is for an experimentw here you force the network to overfit.
        x_patches = self.x_patches[0:1] #Independent of batch index
        y_patches = self.y_patches[0:1]

        x_batch = np.zeros((self.batch_size, self.patch_size, self.patch_size, 3), dtype="float32") #Input images are RGB
        y_batch = np.zeros((self.batch_size, self.patch_size, self.patch_size, 1), dtype="uint8")

        #Go through each patch in batch and augment it
        for i in range(len(x_patches)):
            x_batch[i], y_batch[i] = x_patches[i], y_patches[i]

        # io.imsave(f"TMP2_x.png", x_batch[0])
        # io.imsave(f"TMP2_y.png", color.label2rgb(y_batch[0, :, :, 0], bg_label=0))


        return x_batch, y_batch

