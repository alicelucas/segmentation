from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from skimage import io, transform

class CellsSequence(keras.utils.Sequence):
    """
    Helper class to iterate over the data (from file paths to numpy arrays)
    """

    def __init__(self, x_paths, y_paths, batch_size, image_size):
        self.x_paths, self.y_paths = x_paths, y_paths
        self.batch_size = batch_size
        self.image_size = image_size

    def __len__(self):
        return len(self.x) // self.batch_size

    def __getitem__(self, idx):
        """Return (input, target) numpy array corresponding to batch idx"""

        batch_x_paths = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y_paths = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        x = np.zeros((self.batch_size,) + self.image_size + (3,), dtype="float32") #Input images are RGB
        y = np.zeros((self.batch_size,) + self.image_size + (1,), dtype="uint8") #GT masks are greyscales

        for i, path in enumerate(batch_x_paths):
            x[i] = load_img(path, target_size=self.image_size)
        for i, path in enumerate(batch_y_paths):
            img = load_img(path, target_size=self.image_size, color_mode="grayscale")
            y[i] = np.expand_dims(img, 2)
            # TODO See how ground truth labels are
            # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
            y[i] -= 1

        return x, y



        return np.array([transform.resize(io.imread(file_name), (self.image_size, self.image_size)) for file_name in batch_x]), np.array(batch_y)
