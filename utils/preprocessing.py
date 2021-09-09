import numpy
from skimage import color
from scipy import ndimage
import random
import numpy as np

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


def convert_labels(data, cell_value, background_value, draw_border):
  """
  From gt segmentation masks for instance segmentations to 3-class semantic segmentation
  :return: The three class segmentation mask with (0, 1, 2) labels for background, border, inside cell
  """

  y = numpy.zeros((data.shape[0], data.shape[1]), dtype="uint8") #Greyscale

  if len(y.shape) == 3:
    #Look at R, G, B channels in current mask
    red, green, blue = data[:, :, 0], data[:, :, 1], data[:, :, 2]

  elif len(y.shape) == 2:
    #Greyscale case
    red, green, blue = data[:, :], data[:, :], data[:, :]

  #Replace "inside cell labels with 2"
  cell_label = cell_value, cell_value, cell_value #178, 178, 178 (or whatever was specified by the user)
  cell_pixels = (red == cell_label[0]) & (green == cell_label[1]) & (blue == cell_label[2]) #Get pixel indices that have that color
  #pixels is a (image_size, image_size) array with True and False values
  if draw_border:
    y[cell_pixels] = 2
  else:
    y[cell_pixels] = 1

  #Replace background label with 0
  background_label = background_value, background_value, background_value
  background_pixels = (red == background_label[0]) & (green == background_label[1]) & (
              blue == background_label[2])
  y[background_pixels] = 0

  if draw_border:
    # Replace border label with 1
    #Here making the assumption that everything that is not background or cell was labeled as border
    border_pixels = numpy.logical_not(numpy.logical_or(background_pixels, cell_pixels))
    y[border_pixels] = 1

  return y


def prob_to_mask(pred_mask):
  """
  From array of probabilities to a single image with most likely category having its own color
  :param pred_mask: output of the neural network, size (B, N, M, n) where n is number of classes
  :return: (N, M) image color-coded by class where prob is max
  """
  label = numpy.argmax(pred_mask, axis=-1)
  return color.label2rgb(label, bg_label=0)


def pilToTensor(im):
  """
  From PIL image to a tensor that is suitable for forward pass
  :param im:
  :return: the tensor to be sent to the model
  """
  x = np.asarray(im, dtype="float32")

  if len(x.shape) == 2: #If greyscale input image, make RGB
      x = np.stack((x,) * 3, axis=-1)

  if len(x.shape) == 3 and x.shape[2] == 4: #if alpha channel present, don't feed alpha channel to model
      x = x[:, :, :3]


  return x[np.newaxis, :, :, :]