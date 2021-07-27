import numpy
from skimage import color
from scipy import ndimage
import random

def rotate(x_and_y):
  """
  Given (x, y) tuple, rotate each image (used for data augmentation)
  :return: two rotated image as a tuple
  """
  x = x_and_y[0]
  y = x_and_y[1]
  angle = random.uniform(0, 180)
  output_x = numpy.zeros((x.shape[0], x.shape[1], 3), dtype="float32")
  output_y = numpy.zeros((x.shape[0], x.shape[1], 1), dtype="uint8")

  return ndimage.rotate(x, angle, reshape=False, output=output_x), ndimage.rotate(y, angle, reshape=False, output=output_y)


def flip(x_and_y):
  """
  Given (x, y) tuple, flip each image (used for data augmentation)
  Half of the time we will do a vertical flip, the other half will perform a horizontal flip.
  :return: the two flipped images as a tuple
  """
  x = x_and_y[0]
  y = x_and_y[1]
  if random.uniform(0, 0.5) > 0.5:
    return numpy.flip(x, 0), numpy.flip(y, 0)
  else:
    return numpy.flip(x, 1), numpy.flip(y, 1)


def prob_to_mask(pred_mask):
  """
  From array of probabilities to a single image with most likely category having its own color
  :param pred_mask: output of the neural network, size (B, N, M, n) where n is number of classes
  :return: (N, M) image color-coded by class where prob is max
  """
  label = numpy.argmax(pred_mask, axis=-1)
  return color.label2rgb(label)