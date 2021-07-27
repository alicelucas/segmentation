import numpy
from skimage import color

def normalize(x):
  """
  Array is typically read in as int and from [0, 255]. Here we convert to float and normalize to range [0, 1].
  :param x: the array
  :return: the normalized array
  """
  return x.astype(float) / 255.0

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


def prob_to_mask(pred_mask):
  """
  From array of probabilities to a single image with most likely category having its own color
  :param pred_mask: output of the neural network, size (B, N, M, n) where n is number of classes
  :return: (N, M) image color-coded by class where prob is max
  """
  label = numpy.argmax(pred_mask, axis=-1)
  return color.label2rgb(label)