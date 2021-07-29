from models import model
from input import Cells

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
    val_idx, train_idx = numpy.split(all_idx, [int(0.2 * len(input_img_paths)), int((1 - validation_percentage) * len(input_img_paths))])

    training_generator = Cells.CellsGenerator(input_img_paths[train_idx], target_paths[train_idx], **params)
    validation_generator = Cells.CellsGenerator(input_img_paths[val_idx], target_paths[val_idx], **params)

    history = model.fit_generator(training_generator,
                        validation_data=validation_generator)