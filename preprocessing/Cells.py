from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from skimage import io, transform

class CellsSequence(keras.utils.Sequence):
    """
    Helper class to iterate over the preprocessing (from file paths to numpy arrays)
    """

    def __init__(self, x_paths, y_paths, batch_size, image_size):
        self.x_paths, self.y_paths = x_paths, y_paths
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
        pixels = (red == cell_label[0]) & (green == cell_label[1]) & (blue == cell_label[2]) #Get pixel indices that have that color
        #pixels is a (image_size, image_size) array with True and False values
        y[pixels] = 2

        #Replace background label with 0
        background_label = 0, 0, 0
        pixels = (red == background_label[0]) & (green == background_label[1]) & (
                    blue == background_label[2])
        y[pixels] = 0

        # Replace border label with 1
        #FIXME this misses some of the border pixels
        pixels = (red != background_label[0]) & (green != background_label[1]) & (
                blue != background_label[2]) & (red != cell_label[0]) & (green != cell_label[1]) & (blue != cell_label[2])  # Get pixel indices that have that color
        y[pixels] = 1

        return y


    def __getitem__(self, idx):
        """Return (input, target) numpy array corresponding to batch idx"""

        batch_x_paths = self.x_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y_paths = self.y_paths[idx * self.batch_size:(idx + 1) * self.batch_size]

        x = np.zeros((self.batch_size, self.image_size, self.image_size, 3), dtype="float32") #Input images are RGB
        y = np.zeros((self.batch_size, self.image_size, self.image_size, 1), dtype="uint8")

        for i, path in enumerate(batch_x_paths):
            img = load_img(path, target_size=(self.image_size, self.image_size))
            x[i] = np.array([img_to_array(img)])

        for i, path in enumerate(batch_y_paths):
            img = load_img(path, target_size=(self.image_size, self.image_size))
            data = np.array([img_to_array(img)], dtype="uint8")
            y[i] = np.expand_dims(self.convert_labels(data[0]), 2)

        return x, y

