import random

import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from utils import preprocessing


class CellsSequence(keras.utils.Sequence):
    """
    Helper class to iterate over the preprocessing (from file paths to numpy arrays)
    """

    def __init__(self, x_paths, y_paths, batch_size, image_size):
        self.x_paths, self.y_paths = x_paths, y_paths
        #Randomize the dataset here
        random.Random(1337).shuffle(self.x_paths)
        random.Random(1337).shuffle(self.y_paths)

        self.batch_size = batch_size
        self.image_size = image_size

    def __len__(self):
        return len(self.x_paths) // self.batch_size

    def convert_labels(self, data):
        """
        From gt segmentation masks for instance segmentations to 3-class semantic segmentation
        :return: The three class segmentation mask with (0, 1, 2) labels for background, border, inside cell
        """

        y = np.zeros((self.image_size, self.image_size), dtype="uint8") #Greyscale

        #Look at R, G, B channels in current mask
        red, green, blue = data[:, :, 0], data[:, :, 1], data[:, :, 2]

        #Replace "inside cell labels with 2"
        cell_label = 178, 178, 178
        cell_pixels = (red == cell_label[0]) & (green == cell_label[1]) & (blue == cell_label[2]) #Get pixel indices that have that color
        #pixels is a (image_size, image_size) array with True and False values
        y[cell_pixels] = 2

        #Replace background label with 0
        background_label = 0, 0, 0
        background_pixels = (red == background_label[0]) & (green == background_label[1]) & (
                    blue == background_label[2])
        y[background_pixels] = 0

        # Replace border label with 1
        #Here making the assumption that everything that is not background or cell was labeled as border
        border_pixels = np.logical_not(np.logical_or(background_pixels, cell_pixels))
        y[border_pixels] = 1

        return y


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


    def augment(self, x):
        """
        Given a batch of images and their gt mask, augment it by flipping it (horizontally or vertically), and doing a rotation
        :return: the augmented batch of data
        """
        x = preprocessing.flip(x)
        return preprocessing.rotate(x)


    def __getitem__(self, idx):
        """Return (input, target) numpy array corresponding to batch idx"""

        batch_x_paths = self.x_paths[idx: idx + self.batch_size]
        batch_y_paths = self.y_paths[idx: idx + self.batch_size]

        x = np.zeros((self.batch_size, self.image_size, self.image_size, 3), dtype="float32") #Input images are RGB
        y = np.zeros((self.batch_size, self.image_size, self.image_size, 1), dtype="uint8")

        for i, path in enumerate(batch_x_paths):
            x[i] = load_img(path, target_size=(self.image_size, self.image_size))

        for i, path in enumerate(batch_y_paths):
            img = load_img(path, target_size=(self.image_size, self.image_size))
            data = np.array([img_to_array(img)], dtype="uint8")
            mask = self.convert_labels(data[0])
            y[i] = np.expand_dims(mask, 2)

        x = self.augment(x)
        y = self.augment(y)

        return x, y

