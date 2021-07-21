import numpy
from skimage import color

def to_tensor(x):
  """
  From a (N, M) shape to a (B, N, M, C) shape
  :param x: (N, M) patch
  :return: (B, N, M, C)
  """
  x = numpy.expand_dims(x, axis=-1)
  x = numpy.repeat(x, 3, axis=-1)
  x = numpy.expand_dims(x, axis=0)

  return x


def create_mask(pred_mask):
  """
  From array of probabilities to a single image with most likely category having its own color
  :param pred_mask: output of the neural network, size (B, N, M, n) where n is number of classes
  :return: (N, M) image color-coded by class where prob is max
  """
  label = numpy.argmax(pred_mask, axis=-1)
  return color.label2rgb(label[0])