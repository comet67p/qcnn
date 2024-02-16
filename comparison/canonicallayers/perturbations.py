
# Taken from https://fsix.github.io/mnist/Perturbing.html

from scipy.ndimage import interpolation
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

from skimage.util import random_noise

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import math

seed = 42

def skew(image):
    """Skew the image provided.

    Taken from StackOverflow:
    http://stackoverflow.com/a/33088550/4855984
    """
    imageSize = image.shape[0]
    image = image.reshape(imageSize, imageSize)
    h, l = image.shape
    distortion = np.random.normal(loc=12, scale=1)

    def mapping(point):
        x, y = point
        dec = (distortion*(x-h))/h
        return x, y+dec+5
    return interpolation.geometric_transform(
        image, mapping, (h, l), order=5, mode='nearest')


def rotate(image, d):
    """Rotate the image by d/180 degrees."""
    center = 0.5*np.array(image.shape)
    rot = np.array([[np.cos(d), np.sin(d)],[-np.sin(d), np.cos(d)]])
    offset = (center-center.dot(rot)).dot(np.linalg.inv(rot))
    return interpolation.affine_transform(
        image,
        rot,
        order=2,
        offset=-offset,
        cval=0.0,
        output=np.float32)

# def noise(image, n=100):
#     """Add noise by randomly changing n pixels"""
  # imageSize = image.shape[0]
  # indices = np.random.random(size=(n, 2))*imageSize
  # image = image.copy()
  # for x, y in indices:
  #     x, y = int(x), int(y)
  #     image[x][y] = 0
  # return image

def noise(image, mode='gaussian', amount=0.05):
  """Add different kinds of noise - gaussian, s&p"""
  image = image.copy()
  if(mode == 'gaussian'):
    image = random_noise(image, mode, rng=seed)
  else:
    image = random_noise(image, mode, rng=seed, amount=amount)
  return image

def elastic_transform(image, alpha=36, sigma=5, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    
    :param image: a square image
    :param alpha: scale for filter
    :param sigma: the standard deviation for the gaussian
    :return: distorted square image
    """
    assert len(image.shape) == 2

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))
    
    return map_coordinates(image, indices, order=1).reshape(shape)