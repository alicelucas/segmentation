import random

import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from utils import preprocessing


class CellsGenerator(keras.utils.Sequence):
    """
    Helper class to iterate over the input (from file paths to numpy arrays)
    """

    def __init__(self, x_paths, y_paths, batch_size, patch_size, image_size, pad_size):
        self.batch_size = batch_size
        self.patch_size = patch_size

        self.image_size = image_size
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
        return len(self.x_patches) // self.batch_size


    def create_patches(self, x_paths, y_paths):
        """
        :param x_paths: list of paths of image
        :param y_paths: list of paths of label masks
        :return: list of patches
        """
        # Code below prepares patch extraction process for inference when testing
        patch_size = self.patch_size - 2 * self.pad_size  # account for padding

        x_patches = []
        y_patches = []

        for idx, x_path in enumerate(x_paths):
            baz = load_img(x_path, target_size=(self.image_size, self.image_size))
            x = np.array(img_to_array(baz), dtype="float32")

            foo = load_img(y_paths[idx], target_size=(self.image_size, self.image_size))
            data = np.array([img_to_array(foo)], dtype="uint8")
            mask = preprocessing.convert_labels(data[0])
            y = np.expand_dims(mask, 2)

            # number of column directions
            n_row = x.shape[1] // patch_size
            n_col = x.shape[0] // patch_size

            # pad whole image so that we can account for border effects
            pad_row = int(np.floor((n_row + 1) * patch_size - x.shape[2]) / 2)
            pad_col = int(np.floor((n_col + 1) * patch_size - x.shape[1]) / 2)

            x_padded = np.pad(x, ((pad_col, pad_col), (pad_row, pad_row), (0, 0)))
            y_padded = np.pad(y, ((pad_col, pad_col), (pad_row, pad_row), (0, 0)))


            # Extract patches over image
            for i in range(n_row + 1):
                for j in range(n_col + 1):
                    patch = x_padded[patch_size * i:patch_size * (i + 1), patch_size * j:patch_size * (j + 1), :]  # extract patch
                    patch = np.pad(patch, ((8, 8), (8, 8), (0, 0)))  # add padding
                    x_patches.append(patch)

                    patch = y_padded[patch_size * i:patch_size * (i + 1), patch_size * j:patch_size * (j + 1),
                            :]  # extract patch
                    patch = np.pad(patch, ((8, 8), (8, 8), (0, 0)))  # add padding
                    y_patches.append(patch)

                    self.x_paths.append(x_paths[idx]) #keep track of which patch belongs to which image
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

        x_batch = np.zeros((self.batch_size, self.patch_size, self.patch_size, 3), dtype="float32") #Input images are RGB
        y_batch = np.zeros((self.batch_size, self.patch_size, self.patch_size, 1), dtype="uint8")

        #Go through each patch in batch and augment it
        for i in range(len(x_patches)):
            x_batch[i], y_batch[i] = self.augment((x_patches[i], y_patches[i])) #Place augmented patch in batch


        return x_batch, y_batch

