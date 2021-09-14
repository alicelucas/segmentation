import random

import numpy as np
from sklearn.preprocessing import LabelBinarizer
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from utils import preprocessing

from skimage import io, color


class CellsGenerator(keras.utils.Sequence):
    """
    Helper class to iterate over the input (from file paths to numpy arrays)
    """

    def __init__(self, x_paths, y_paths, batch_size, patch_size, crop_border, background_value, cell_value, draw_border, augment_count, should_augment):
        self.should_augment = should_augment
        self.augment_count = augment_count

        self.patch_size = patch_size

        self.crop_border = crop_border

        self.background_value = background_value
        self.cell_value = cell_value
        self.draw_border = draw_border

        self.x_paths = []
        self.y_paths = []

        self.x_patches, self.y_patches = self.create_patches(x_paths, y_paths)

        self.batch_size = min(batch_size, len(self.x_patches))

        # Randomize the dataset here
        random.Random(1337).shuffle(self.x_patches)
        random.Random(1337).shuffle(self.y_patches)
        random.Random(1337).shuffle(self.x_paths)
        random.Random(1337).shuffle(self.y_paths)

    def __len__(self):
        """
        Returns number of batches
        :return:
        """
        return len(self.x_patches) // self.batch_size

    def create_patches(self, x_paths, y_paths):
        """
        :param x_paths: list of paths of image
        :param y_paths: list of paths of label masks
        :return: list of patches of size 224 x 224
        """
        # Code below prepares patch extraction process for extracting patches from image
        x_patches = []
        y_patches = []

        for idx, x_path in enumerate(x_paths):

            if idx % 200 == 0:
                print(f"Loading and patching file #{idx} in {len(x_paths)} files")

            baz = load_img(x_path, color_mode="rgb")
            x_ori = np.array(img_to_array(baz), dtype="float32")

            foo = load_img(y_paths[idx], color_mode="rgb")
            data = np.array([img_to_array(foo)], dtype="uint8")
            mask = preprocessing.convert_labels(data[0], self.cell_value, self.background_value, self.draw_border)

            y_ori = np.expand_dims(mask, 2)


            count = 0  # The number of times we will augment the image

            while count < self.augment_count:

                if self.should_augment:
                    x, y = self.augment((x_ori, y_ori))  # Augment image
                else:
                    x, y = x_ori, y_ori

                # number of column directions
                n_row = x.shape[1] // self.patch_size
                n_col = x.shape[0] // self.patch_size

                # Extract patches over image
                for i in range(n_row):
                    for j in range(n_col):

                        # Extract y-patch
                        patch = y[self.patch_size * i:self.patch_size * (i + 1),
                                self.patch_size * j:self.patch_size * (j + 1), :]  # extract patch

                        if patch.shape[0] < self.patch_size or patch.shape[1] < self.patch_size:
                            continue #only go through this if we extracted a (patch size x patch size) patch

                        cropped_patch = patch[self.crop_border: patch.shape[0] - self.crop_border,
                                        self.crop_border: patch.shape[1] - self.crop_border, :]


                        # Only keep the patch if it has non-zero labels (i.e., not just black)
                        if 1 not in cropped_patch[:, :, 0]:
                            continue

                        # Convert y integer labels to one-hot labels
                        label_binarizer = LabelBinarizer()
                        label_binarizer.fit(range(np.amax(cropped_patch) + 1))
                        one_hot_patch = np.reshape(label_binarizer.transform(cropped_patch[:, :, 0].flatten()),
                                                   [cropped_patch.shape[0], cropped_patch.shape[1], -1])
                        #WARNING: if only two classes, this will stay as a sparse matrix and will not be one-hot encoded
                        #(from sklearn's documentation)
                        y_patches.append(one_hot_patch)

                        # Extract x-patch
                        patch = x[ self.patch_size * i:self.patch_size * (i + 1), self.patch_size * j:self.patch_size * (j + 1),
                                :]  # extract center patch
                        x_patches.append(patch)

                        # keep track of which patch belongs to which image
                        self.x_paths.append(x_paths[idx])
                        self.y_paths.append(y_paths[idx])

                        count += 1 #Increment augment count to keep track of number of augmented patches

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
        Given a batch of images and their gt mask, augment it by flipping it (horizontally or vertically),  doing a rotation,
        and scaling it
        :return: the augmented batch of data
        """

        flipped = preprocessing.flip(x_and_y)
        scaled = preprocessing.scale(flipped)
        rotated = preprocessing.rotate(scaled)

        return rotated

    def __getitem__(self, batch_idx):
        """Return (input, target) numpy array corresponding to batch idx"""

        x_patches = self.x_patches[batch_idx * self.batch_size: batch_idx * self.batch_size + self.batch_size]
        y_patches = self.y_patches[batch_idx * self.batch_size: batch_idx * self.batch_size + self.batch_size]

        num_classes = y_patches[0].shape[2]
        num_channels = x_patches[0].shape[2]
        x_shape = x_patches[0].shape[1]
        y_shape = y_patches[0].shape[1]
        x_batch = np.zeros((self.batch_size, x_shape, x_shape, num_channels),
                           dtype="float32")  # Input images are RGB
        y_batch = np.zeros((self.batch_size, y_shape, y_shape, num_classes),
                           dtype="float32")  # One hot encoded

        # Go through each patch in batch and augment it
        for i in range(len(x_patches)):
            x_batch[i], y_batch[i] = x_patches[i], y_patches[i]

        # io.imsave(f"TMP2_x.png", x_batch[0])
        # io.imsave(f"TMP2_y.png", color.label2rgb(y_batch[0, :, :, 0], bg_label=0))

        return x_batch, y_batch
