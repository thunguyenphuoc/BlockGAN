"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import tarfile
import zlib
import io
from PIL import Image
import random
import pprint
import scipy.misc
import numpy as np

from tools.rotation_utils import *


import tensorflow as tf
import tensorflow.contrib.slim as slim
import glob
import os
import random
import scipy.misc

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

def show_all_variables():
  model_vars = tf.trainable_variables()
  slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def get_image(image_path, input_height, input_width,
              resize_height=64, resize_width=64,
              crop=True):
  image = load_webp(image_path)
  return transform(image, input_height, input_width,
                   resize_height, resize_width, crop)

def get_image_jitter(image_path, input_height, input_width,
              resize_height=64, resize_width=64, ratio=1.0,
              crop=True):
  image = load_webp(image_path)
  return transform_jitter(image, input_height, input_width,
                   resize_height, resize_width, ratio, crop)

def get_image_random_crop(image_path, resize_height=64, resize_width=64):
    image = load_webp(image_path)
    return transform_random_crop(image, resize_height, resize_width)

def save_images(images, size, image_path):
  return imsave(inverse_transform(images), size, image_path)

def load_webp(img_path):
    im = Image.open(img_path)
    return np.asarray(im)


def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  if (images.shape[3] in (3,4)):
    c = images.shape[3]
    img = np.zeros((h * size[0], w * size[1], c))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w, :] = image
    return img
  elif images.shape[3]==1:
    img = np.zeros((h * size[0], w * size[1]))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
    return img
  else:
    raise ValueError('in merge(images,size) images parameter '
                     'must have dimensions: HxW or HxWx3 or HxWx4')

def crop_square(im):

    h = im.shape[0]
    w = im.shape[1]

    crop_size = min(h, w)

    if h > w: #If image is vertical
        #Crop using center
        crop_size = min(h,w)
        mid_point = h // 2
        jitter = random.randint(0, (h- crop_size) // 2)
        mid_point += jitter #Move away from center crop to give some diversity
        try:
            cropped = im[(mid_point - crop_size // 2) : (mid_point + crop_size //2), :, :] #Crop using midpoint
            # cropped = im[top_left:top_left+crop_size, :, :] #Crop using top left point
        except:
            return None
    elif h == w: #If image is square
        cropped = im
    else: #If image is horizontal
        top_left = random.randint(0, w - crop_size)
        try:
            cropped = im[:, top_left:top_left+crop_size, :]
        except:
            return None

    return cropped

def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
  print(x.shape)
  h, w = x.shape[:2]
  min_dim = np.min([h, w])
  if min_dim < crop_h:
    print("MIN DIM {0}".format(min_dim))
    crop_h = min_dim
    print(crop_h)
  if h == w:
      print("EQUAL")
      return scipy.misc.imresize(x, [resize_h, resize_w])
  if crop_w is None:
    crop_w = crop_h
  j = int(round((h - crop_h)/2.))
  i = int(round((w - crop_w)/2.))
  return scipy.misc.imresize(
      x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def center_crop_jitter(x, crop_h, crop_w=None,
                resize_h=64, resize_w=64, ratio=1.0):
  print("Center crop jitter")
  h, w = x.shape[:2]
  min_dim = np.min([h, w])
  if min_dim < crop_h:
    crop_w = min_dim
  if h == w:
      return scipy.misc.imresize(x, [resize_h, resize_w])
  if h < w: #Only consider horizontal images
    mid_point = w // 2
    diff = w-min_dim
    rand = random.randint(0, int(ratio * diff // 2))
    if random.random() >= 0.5:
        mid_point += rand
    else:
        mid_point -= rand
    return scipy.misc.imresize(
          x[:, mid_point-crop_w//2:mid_point+crop_w//2], [resize_h, resize_w])
  if h > w:
      # Crop using center
      crop_size = min(h, w)
      mid_point = h // 2
      cropped = x[(mid_point - crop_size // 2): (mid_point + crop_size // 2), :, :]  # Crop using midpoint
      return scipy.misc.imresize(cropped, [resize_h, resize_w])

def random_crop(x,
                resize_h=64, resize_w=64):
  cropped = crop_square(x)
  return scipy.misc.imresize(
      cropped, [resize_h, resize_w])

def transform(image, input_height, input_width,
              resize_height=64, resize_width=64, crop=True):
  if crop:
    cropped_image = center_crop(
      image, input_height, input_width,
      resize_height, resize_width)
  else:
    cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
  if len(cropped_image.shape) != 3: #In case of binary mask with no channels:
    cropped_image = np.expand_dims(cropped_image, -1)
  return np.array(cropped_image)[:, :, :3]/127.5 - 1.

def transform_jitter(image, input_height, input_width,
              resize_height=64, resize_width=64, ratio=1.0, crop=True):
  if crop:
    cropped_image = center_crop_jitter(
      image, input_height, input_width,
      resize_height, resize_width, ratio)
  else:
    cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
  if len(cropped_image.shape) != 3: #In case of binary mask with no channels:
    cropped_image = np.expand_dims(cropped_image, -1)
  return np.array(cropped_image)[:, :, :3]/127.5 - 1.

def transform_random_crop(image, resize_height=64, resize_width=64):
  cropped_image = crop_square(image)
  cropped_image = scipy.misc.imresize(cropped_image, [resize_height, resize_width])
  if len(cropped_image.shape) != 3: #In case of binary mask with no channels:
    cropped_image = np.expand_dims(cropped_image, -1)
  return np.array(cropped_image)[:, :, :3]/127.5 - 1.

def inverse_transform(images):
  return (images+1.)/2.

def image_manifold_size(num_images):
  manifold_h = int(np.floor(np.sqrt(num_images)))
  manifold_w = int(np.ceil(np.sqrt(num_images)))

  assert manifold_h * manifold_w == num_images
  return manifold_h, manifold_w

def to_bool(value):
    """
       Converts 'something' to boolean. Raises exception for invalid formats
           Possible True  values: 1, True, "1", "TRue", "yes", "y", "t"
           Possible False values: 0, False, None, [], {}, "", "0", "faLse", "no", "n", "f", 0.0, ...
    """
    if str(value).lower() == "true": return True
    if str(value).lower() == "false": return False
    raise Exception('Invalid value for boolean conversion: ' + str(value))
