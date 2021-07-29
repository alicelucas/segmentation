from models import model
from input import Cells
import tensorflow as tf
from os.path import join, isfile
from os import listdir
import numpy

def train():
    unet = model.unet_model()

    base_learning_rate = 0.0001

    unet.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    validation_percentage = 0.2

    # Get list of input files and target masks
    image_dir = "data/maddox/images"
    target_dir = "data/maddox/masks"
    input_img_paths = [join(image_dir, f) for f in listdir(image_dir) if isfile(join(image_dir, f))]
    target_paths = [join(target_dir, f) for f in listdir(target_dir) if isfile(join(target_dir, f))]

    all_idx = numpy.arange(len(input_img_paths))
    sep_idx = numpy.split(all_idx, [int(0.2 * len(input_img_paths)), int((1 - validation_percentage) * len(input_img_paths))])

    batch_size = 64

    patch_size = 224
    image_size = 1024

    training_generator = Cells.CellsGenerator(input_img_paths[sep_idx[1]], target_paths[sep_idx[1]], batch_size, patch_size, image_size)
    validation_generator = Cells.CellsGenerator(input_img_paths[sep_idx[0]], target_paths[sep_idx[0]],batch_size, patch_size, image_size )

    history = unet.fit_generator(training_generator,
                        validation_data=validation_generator)