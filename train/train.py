from os import listdir
from os.path import join, isfile

import numpy
import tensorflow as tf

from input import Cells
from models import model


def train_unet(config):
    """
    :param config: config file
    """
    unet = model.unet_model()

    base_learning_rate = config["lr"]
    epochs = config["epochs"]
    validation_percentage = config["validation_percentage"]

    # Get list of input files and target masks
    image_dir = config["train_image_dir"]
    target_dir = config["train_mask_dir"]
    input_img_paths = [join(image_dir, f) for f in listdir(image_dir) if isfile(join(image_dir, f))]
    target_paths = [join(target_dir, f) for f in listdir(target_dir) if isfile(join(target_dir, f))]

    all_idx = numpy.arange(len(input_img_paths))
    numpy.random.shuffle(all_idx)
    val_train_idx = numpy.split(all_idx, [int(0.2 * len(input_img_paths)), int((1 - validation_percentage) * len(input_img_paths))])

    batch_size = config["batch_size"]

    patch_size = config["input_size"]
    image_size = config["image_size"]

    training_generator = Cells.CellsGenerator(numpy.take(input_img_paths, val_train_idx[1]), numpy.take(target_paths, val_train_idx[1]), batch_size, patch_size, image_size)
    validation_generator = Cells.CellsGenerator(numpy.take(input_img_paths, val_train_idx[1]), numpy.take(target_paths, val_train_idx[1]),batch_size, patch_size, image_size)

    unet.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    unet.fit(training_generator, epochs=epochs,
                        validation_data=validation_generator)

    model_name = config["saved_model_name"]
    save_dir = config["save_dir"]
    unet.save(join(save_dir, model_name))