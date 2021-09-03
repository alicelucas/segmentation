from os import listdir
from os.path import join, isfile

import numpy
import tensorflow as tf
import tensorflow_addons as tfa
from matplotlib import pyplot as plt
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from input import Cells
from models import model

import math

from losses.weighted_categorical_cross_entropy import weighted_categorical_crossentropy



def train_unet(config):
    """
    :param config: config file
    """

    input_size = config["input_size"]

    optimizer_name = config["optimizer_name"]

    dropout = config["dropout"]

    crop_size = config["crop_border"]

    weight_class = config["weight_class"]

    background_value = config["background"]
    cell_value = config["cell"]
    draw_border = config["draw_border"]

    num_classes = config["num_classes"]

    unet = model.unet_model([input_size, input_size, 3], num_classes, crop_size, dropout=dropout) #Assume training with 224 x 224 patches
    unet.summary()

    base_learning_rate = config["lr"]
    epochs = config["epochs"]
    validation_percentage = config["validation_percentage"]

    # Get list of input files and target masks
    image_dir = config["train_image_dir"]
    target_dir = config["train_mask_dir"]
    input_img_paths = [join(image_dir, f) for f in listdir(image_dir) if isfile(join(image_dir, f)) and not f.startswith('.')]
    target_paths = [join(target_dir, f) for f in listdir(target_dir) if isfile(join(target_dir, f)) and not f.startswith('.')]

    all_idx = numpy.arange(len(input_img_paths))
    numpy.random.shuffle(all_idx)
    val_train_idx = numpy.split(all_idx, [int(math.floor(validation_percentage * len(input_img_paths)))])

    batch_size = config["batch_size"]

    patch_size = config["input_size"]

    should_augment = config["augmentation"]

    training_generator = Cells.CellsGenerator(numpy.take(input_img_paths, val_train_idx[1]), numpy.take(target_paths, val_train_idx[1]),
                                              batch_size, patch_size, crop_size, background_value, cell_value, draw_border, should_augment)
    validation_generator = Cells.CellsGenerator(numpy.take(input_img_paths, val_train_idx[0]), numpy.take(target_paths, val_train_idx[0]),
                                                8, patch_size, crop_size, background_value, cell_value, draw_border, should_augment=False)

    if optimizer_name == "Adam":
        optimizer = tf.keras.optimizers.Adam(lr=base_learning_rate)
    elif optimizer_name == "RMSProp":
        optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate)
    elif optimizer_name == "SGD":
        optimizer = tf.keras.optimizers.SGD(lr=base_learning_rate)
    else:
        optimizer = tf.keras.optimizers.Adam(lr=base_learning_rate)

    # loss = CategoricalCrossentropy()
    #From the y label, determine whether it is one hot encoded or not. If not, you shoudl use sparse categorical cross entropy.
    _, foo = training_generator.__getitem__(0)

    if foo.shape[3] > 1: #If one hot encoded, use your custom cateogorical cross entropy
        loss = weighted_categorical_crossentropy(weights=weight_class) #Custom categorical cross entropy where you can specify weights for each class
    else: #Otherwise use tensorflow's sparsecategorical cross entropy
        loss = SparseCategoricalCrossentropy()

    unet.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy'])

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)

    history = unet.fit(training_generator, epochs=epochs,
                        validation_data=validation_generator, callbacks=[early_stopping])

    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Model loss")
    plt.ylabel("Categorical crossentropy")
    plt.xlabel("Epoch")
    plt.legend(["train", "val"], loc="upper right")
    plt.savefig("loss.png")


    model_name = config["saved_model_name"]
    save_dir = config["save_dir"]
    unet.save(join(save_dir, model_name))