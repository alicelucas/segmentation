from os import listdir
from os.path import join, isfile

import numpy
import tensorflow as tf
from matplotlib import pyplot as plt

from input import Cells
from models import model

import math


def train_unet(config):
    """
    :param config: config file
    """

    input_size = config["input_size"]

    optimizer_name = config["optimizer_name"]

    dropout = config["dropout"]

    unet = model.unet_model([input_size, input_size, 3], dropout=dropout) #Assume training with 224 x 224 patches
    unet.summary()

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
    val_train_idx = numpy.split(all_idx, [int(math.floor(validation_percentage * len(input_img_paths)))])

    batch_size = config["batch_size"]

    patch_size = config["input_size"]
    pad_size = config["pad_size"]

    should_augment = config["augmentation"]

    training_generator = Cells.CellsGenerator(numpy.take(input_img_paths, val_train_idx[1]), numpy.take(target_paths, val_train_idx[1]), batch_size, patch_size, pad_size, should_augment)
    validation_generator = Cells.CellsGenerator(numpy.take(input_img_paths, val_train_idx[0]), numpy.take(target_paths, val_train_idx[0]), 8, patch_size, pad_size, should_augment=False)

    if optimizer_name == "Adam":
        optimizer = tf.keras.optimizers.Adam(lr=base_learning_rate)
    elif optimizer_name == "RMSProp":
        optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate)
    elif optimizer_name == "SGD":
        optimizer = tf.keras.optimizers.SGD(lr=base_learning_rate)
    else:
        optimizer = tf.keras.optimizers.Adam(lr=base_learning_rate)

    unet.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)

    history = unet.fit(training_generator, epochs=epochs,
                        validation_data=validation_generator, callbacks=[early_stopping])

    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Model loss")
    plt.ylabel("Sparse categorical crossentropy")
    plt.xlabel("Epoch")
    plt.legend(["train", "val"], loc="upper right")
    plt.savefig("loss.png")


    model_name = config["saved_model_name"]
    save_dir = config["save_dir"]
    unet.save(join(save_dir, model_name))