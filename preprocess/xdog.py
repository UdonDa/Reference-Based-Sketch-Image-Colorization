import sys

import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter

def hatch(image):
  """
  A naive hatching implementation that takes an image and returns the image in 
  the style of a drawing created using hatching.
  image: an n x m single channel matrix.
  
  returns: an n x m single channel matrix representing a hatching style image.
  """
  xdogImage = xdog(image, 0.1)

  hatchTexture = cv2.imread('./textures/hatch.jpg', cv2.CV_LOAD_IMAGE_GRAYSCALE)

  height = len(xdogImage)
  width = len(xdogImage[0])

  if height > 1080 or width > 1920:
    print "This method only supports images up to 1920x1080 pixels in size"
    sys.exit(1)

  croppedTexture = hatchTexture[0:height, 0:width]

  return xdogImage + croppedTexture

def xdog(image, epsilon=0.01):
  """
  Computes the eXtended Difference of Gaussians (XDoG) for a given image. This 
  is done by taking the regular Difference of Gaussians, thresholding it
  at some value, and applying the hypertangent function the the unthresholded
  values.
  image: an n x m single channel matrix.
  epsilon: the offset value when computing the hypertangent.
  returns: an n x m single channel matrix representing the XDoG.
  """
  phi = 10

  difference = dog(image, 200, 0.98)/255
  diff = difference*image

  for i in range(0, len(difference)):
    for j in range(0, len(difference[0])):
      if difference[i][j] >= epsilon:
        difference[i][j] = 1
      else:
        ht = np.tanh(phi*(difference[i][j] - epsilon))
        difference[i][j] = 1 + ht

  return difference*255

def dog(image, k=200, gamma=1):
  """
  Computes the Difference of Gaussians (DoG) for a given image. Returns an image 
  that results from computing the DoG. 
  image: an n x m array for which the DoG is computed.
  k: the multiplier the the second Gaussian sigma value.
  gamma: the multiplier for the second Gaussian result.
  
  return: an n x m array representing the DoG
  """

  s1 = 0.5
  s2 = s1*k

  gauss1 = gaussian_filter(image, s1)
  gauss2 = gamma*gaussian_filter(image, s2)

  differenceGauss = gauss1 - gauss2
  return differenceGauss

if __name__ == '__main__':
  try:
    in_fn, out_fn = sys.argv[1:3]
  except:
    print "Usage: python xdog.py FILE OUTPUT"
    exit(1)

  image = cv2.imread(in_fn, cv2.CV_LOAD_IMAGE_GRAYSCALE)
  
  # Main method defaults to computiong the eXtended Diference of Gaussians
  result = xdog(image)

  cv2.imwrite(out_fn, result)