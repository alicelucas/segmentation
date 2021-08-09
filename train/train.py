from os import listdir
from os.path import join, isfile

import numpy
import tensorflow as tf

from input import Cells
from models import model


def train_unet():
    unet = model.unet_model()

    base_learning_rate = 0.0001


    validation_percentage = 0.2

    # Get list of input files and target masks
    image_dir = "data/maddox/images"
    target_dir = "data/maddox/masks"
    input_img_paths = [join(image_dir, f) for f in listdir(image_dir) if isfile(join(image_dir, f))]
    target_paths = [join(target_dir, f) for f in listdir(target_dir) if isfile(join(target_dir, f))]

    all_idx = numpy.arange(len(input_img_paths))
    numpy.random.shuffle(all_idx)
    val_train_idx = numpy.split(all_idx, [int(0.2 * len(input_img_paths)), int((1 - validation_percentage) * len(input_img_paths))])

    batch_size = 2

    patch_size = 224
    image_size = 1024

    training_generator = Cells.CellsGenerator(numpy.take(input_img_paths, val_train_idx[1]), numpy.take(target_paths, val_train_idx[1]), batch_size, patch_size, image_size)
    validation_generator = Cells.CellsGenerator(numpy.take(input_img_paths, val_train_idx[1]), numpy.take(target_paths, val_train_idx[1]),batch_size, patch_size, image_size )

    # ##TEMPORARY DEBUG
    # x_batch, y_batch = training_generator.__getitem__(4)
    # # Visualize input image and ground-truth output
    # io.imsave(f"./images/x_patch.png", x_batch[1])
    # io.imsave(f"./images/y_patch.png", color.label2rgb(y_batch[1][:, :, 0]))
    # ##END OF TEMPORARY DEBUG

    unet.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = unet.fit(training_generator,
                        validation_data=validation_generator)