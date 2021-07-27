import numpy
from skimage import color
from scipy import ndimage
import random

def rotate(x):
  """
  Rotate a batch of image (used for data augmentation) - size B x N x M x C
  :return: the batch of rotated images
  """
  angle = random.uniform(0, 180)
  return ndimage.rotate(x, angle, (1, 2), reshape=False)


def flip(x):
  """
  Flips images (used for data augmentation)
  Half of the time we will do a vertical flip, the other half will perform a horizontal flip.
  :return: the flipped images
  """
  if random.uniform(0, 0.5) > 0.5:
    return numpy.flip(x, 0)
  else:
    return numpy.flip(x, 1)


def prob_to_mask(pred_mask):
  """
  From array of probabilities to a single image with most likely category having its own color
  :param pred_mask: output of the neural network, size (B, N, M, n) where n is number of classes
  :return: (N, M) image color-coded by class where prob is max
  """
  label = numpy.argmax(pred_mask, axis=-1)
  return color.label2rgb(label)