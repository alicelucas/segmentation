import os
from PIL import Image
import numpy as np
import skimage.io
import pandas as pd

def decode_DSB_test_set():
    """
    The DSB ground truth test set is provided as run-length-encoded data
    Go through that data and get the semantic png masks
    This code was taken from:
    https://github.com/carpenterlab/2019_caicedo_dsb/blob/7ce8dd12be9d5ca3fe719a3a2e62bba42b63e3bb/unet4nuclei/00-download-dataset.ipynb
    :return:
    """

    def decode(encoded, shape):
        r, c = shape

        if str(encoded) == 'nan':
            return None

        encoded = encoded.replace("[", "").replace("]", "").replace(",", " ")
        encoded = [int(instance) for instance in encoded.split(" ") if instance != '']

        image = np.zeros(r * c, dtype=np.uint8)

        for index, size in np.array(encoded).reshape(-1, 2):
            index -= 1

            image[index:index + size] = 255

        return image.reshape(c, r).transpose()

    def label_objects(mask_objects, shape):
        labels = np.zeros(shape, np.uint16)

        for index, mask_object in enumerate(mask_objects.itertuples()):
            decoded = decode(mask_object.EncodedPixels, shape)
            if decoded is not None:
                # labels[decoded == 255] = index + 1 #Instance segmentation
                labels[decoded == 255] = 255 #Semantic segmentation

        return labels

    masks_csv = './data/2018-DSB/instance/stage1_solution.csv'
    normalized_images_dir = "./data/2018-DSB/instance/stage1_test/"
    out = "./data/2018-DSB/semantic/test/"

    df = pd.read_csv(masks_csv)

    for imId in df.ImageId.unique():
        im = skimage.io.imread(normalized_images_dir + imId + "/images/" +  imId + ".png")
        objects = label_objects(df[df.ImageId == imId], im.shape[0:2])
        skimage.io.imsave(out + imId + ".png", objects)


    pass

def convert_DSB_to_semantic_masks(dataset_path, out_path):
    """
    The DSB dataset has multiple masks for a given image, since it provides the instance segmentation problem.
    Go through the dataset and make a new one such that it corresponds to the semantic format: for each image,
    we have a single mask. The mask will be binary in this case.
    Expects that the dataset_path folder has a folder of "images" with one image file
    And a folder of "masks" with multiple masks for each image (see DSB as example).

    e.g dataset_path = = "data/2018-DSB/stage1_train"
    e.g. out_path = "data/2018-DSB/semantic/train"
    """

    #Write output directories
    out_images_path = os.path.join(out_path, "images")
    out_masks_path = os.path.join(out_path, "masks")

    if not os.path.exists(out_images_path):
        os.makedirs(out_images_path)
    if not os.path.exists(out_masks_path):
        os.makedirs(out_masks_path)

    num_images = len(os.listdir(dataset_path)) #Used for logging

    for i, subdir in enumerate(os.listdir(dataset_path)):

        if (i % 50 == 0):
            print(f"Processing image {i} out of {num_images}")

        if not subdir.startswith('.') and os.path.isdir(os.path.join(dataset_path, subdir)): #Ignore hidden files, e.g. DSStore

            images_subdir = os.path.join(dataset_path, subdir, "images")
            masks_subdir = os.path.join(dataset_path, subdir, "masks")

            image_size = [0, 0]

            for file in os.listdir(images_subdir):
                #Read and save the image immediately
                im = Image.open(os.path.join(images_subdir, file))
                im.save(os.path.join(out_images_path, subdir+".png"))
                image_size = [im.size[1], im.size[0]] #PIL reads in column order, so inverting size

            semantic_mask = np.zeros(image_size, dtype="uint8")

            for file in os.listdir(masks_subdir):
                mask = Image.open(os.path.join(masks_subdir, file))
                mask_np = np.asarray(mask)
                semantic_mask[mask_np == 255] = 255

            Image.fromarray(semantic_mask).save(os.path.join(out_masks_path, subdir+".png"))

