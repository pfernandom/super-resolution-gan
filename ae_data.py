#!/usr/bin/env python
# coding: utf-8

# In[31]:


# %load_ext filprofiler


# In[32]:

import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow.keras as keras
import math
from json import dumps
from tqdm.notebook import tqdm, trange
from IPython import display
import tensorflow_addons as tfa
from tensorflow.keras.layers import Conv2D, LeakyReLU, MaxPooling2D, Rescaling
from enum import Enum

from matplotlib import pyplot as plt
from IPython import display # If using IPython, Colab or Jupyter
import numpy as np
import tensorflow_addons as tfa
import datetime
import random
from sklearn.model_selection import train_test_split
import time

import os
import re
import pathlib


# In[35]:


def get_test_img():
    img = tf.keras.utils.load_img("upscale_imgs/1000_large.jpeg",  target_size=(480,720), keep_aspect_ratio=True)
    return tf.expand_dims(img, axis=0)

def imshow(img, title=None):
    plt.figure()
    if title:
        plt.title(title)
    plt.imshow(tf.cast(tf.squeeze(img), tf.uint8))


# ## Base augmentations

# In[36]:


def random_invert_img(x, p=0.5):
    if tf.random.uniform([]) < p:
        x = (255-x)
    else:
        x
    return x


def random_apply_saturation(x, p=0.5):
    if tf.random.uniform([]) < p:
        return tf.image.random_saturation(x, 5, 10)
    return x


# ## Base transformations

# In[37]:


def get_mask(size=(480, 720), invert=False, line_width=2, one_val=1., zero_val=0.):
    pattern = [one_val, zero_val] if invert else [zero_val, one_val]
    content = tf.constant(pattern, tf.float16)
    content = tf.repeat(content, line_width)

    p = tf.tile(
        content, [tf.cast(size[0]/len(content), tf.uint8)], name="mask_tile")
    p = tf.repeat([p], size[1], axis=0)
    p = tf.stack([p, p, p])
    p = tf.transpose(p, [2, 1, 0])
    return p


def get_masks(line_width=None):
    if line_width is None:
        line_width = tf.random.uniform(
            shape=(), minval=1, maxval=2, dtype=tf.int32)
#         tf.print(f"mask-width={line_width}")
#         print(f"mask-width={line_width}")
    return get_mask(line_width=line_width), get_mask(invert=True, line_width=line_width)


def lines_noise(img, roll_size, axis=1, line_width=None):
#     print(f"overlap_noise: axis={axis}, roll_size={roll_size} line_width={line_width}")
    m1, m2 = get_masks(line_width)
    img = tf.cast(img, tf.float16)
    rolled = tf.roll(img, roll_size, axis=axis)
    noisy1 = m1 * img
    noisy2 = m2 * rolled

    return tf.cast(noisy1 + noisy2, tf.uint8)


def overlap_noise(img, roll_size, noise_level=0.4, axis=1):
#     print(f"overlap_noise: axis={axis}, roll_size={roll_size}")
    img = tf.cast(img, tf.float32)
    rolled = tf.roll(img, roll_size, axis=axis)
    return tf.cast((img * (1-noise_level)) + (rolled * noise_level), tf.uint8)


def pixel_noise(img, size=(480, 720), factor=5):
    factor = 1/factor
    w = int(size[0]*factor)
    h = int(size[1]*factor)
    img = tf.image.resize(img, [w,h])
    return tf.image.resize(
        img,
        size,
        method="nearest",
        preserve_aspect_ratio=True,
        antialias=False,
        name=None
    )


# ## Function wrappers that only need an image as input

# In[38]:


def random_full_lines(x):
#     tf.print("(random full lines)")
    line_width = 10
#     line_width = 10
    full_lines_mask = tf.cast(get_mask(invert=True, line_width=line_width, one_val=0.5, zero_val=0.), tf.float32)

    noise_level = 0.1
    xn = tf.cast(x, tf.float32) * (1-noise_level)
    xm = tfa.image.mean_filter2d(full_lines_mask, filter_shape=6)
    xm = (xm * 255.) * noise_level
    return tf.cast(xn + xm, tf.uint8)

@tf.function(reduce_retracing=True)
def gauss(x):
        return tfa.image.mean_filter2d(x, filter_shape=random.choice([6,10]))
def pixels(x):
        x = tf.cast(x, tf.float32)
        y = x
        x = pixel_noise(x, factor=5)
        noise_level = random.choice([0.7,0.8,0.9])
        res = (((1-noise_level) * y) + (noise_level * x)) 
        return tf.cast(res, tf.uint8)

min_roll=3
max_roll=40
    
def overlapping(x):
    return overlap_noise(x, roll_size=random.randint(min_roll,max_roll), noise_level=random.choice([0.4, 0.5]), axis=2)

def lines(x):
    return lines_noise(x, axis=2, roll_size=random.randint(min_roll,max_roll), line_width=random.randint(1,5))

def overlapping2(x):
    return overlap_noise(x, roll_size=random.randint(max_roll*-1,min_roll*-1), noise_level=random.choice([0.4, 0.5]), axis=2)

def lines2(x):
    return lines_noise(x, axis=2, roll_size=random.randint(max_roll*-1,min_roll*-1), line_width=random.randint(1,5))


# ## Map name to transformation

# In[39]:


transfs = {
    "overlapping": (3, overlapping),
    "overlapping2": (2, overlapping2),
    "gaussian_filter": (3, gauss),
    "pixelation_noise": (3, pixels),
    "hlines": (3, lines),
    "hlines2": (2, lines2),
    "full_lines": (1, random_full_lines),
}


# ## Acceptable transformations:
# 
# - overlapping + full_lines
# - gaussian_filter + overlapping
# - gaussian_filter + full_lines
# - hlines + full_lines
# - full_lines + overlapping
# - full_lines + hlines

# In[50]:


def pipe(*ts):
    def fn(x):
        for w, t in ts:
            x = t(x)
        return x
    return fn


def compose(weight, acc, tmap, *tnames):
    acc.append((weight, " + ".join(tnames),
               pipe(*[tmap[tn] for tn in tnames])))


def create_list_of_transformations():
    # at = [(w, k, v) for k, (w, v) in filter(lambda x:x[0] in ["overlapping",
    #                                                           "overlapping2", "gaussian_filter", "pixelation_noise"], transfs.items())]
    at = []
    for k, (w, v) in transfs.items():
        at.append((w, k, v))
    # compose(at, transfs, "overlapping", "overlapping2")
    compose(2, at, transfs, "overlapping", "full_lines")
    compose(3, at, transfs, "gaussian_filter", "full_lines")
    compose(3, at, transfs, "gaussian_filter", "overlapping2")
    compose(2, at, transfs, "gaussian_filter", "overlapping2",  "hlines")
    compose(1, at, transfs, "hlines", "full_lines")
    compose(1, at, transfs, "full_lines", "overlapping")
    compose(1, at, transfs, "full_lines", "hlines")
    return at


# In[52]:


def apply_random_filters(x):
    all_filters = create_list_of_transformations()
    weights = list(map(lambda x: x[0], all_filters))

    w, filter_name, fn = random.choices(all_filters, weights=weights, k=1)[0]
    x = fn(x)
#     tf.print(filter_name)
#     print(f"transformation:{filter_name}")
    x = tf.cast(x, tf.uint8)
    return x

def augment_img(x):
    
    if tf.random.uniform(shape=[]) > 0.5:
        x = tf.image.random_brightness(x, 0.2)
        x = tf.image.random_contrast(x, 0.2, 0.5)
    x = random_apply_saturation(x, p=0.4)
    # x = tf.cast(x, tf.float16)
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    x = random_invert_img(x, p=0.4)
    x = tf.cast(x, tf.uint8)
    return x

def resize_data(h,w):
    def fn(x):
        x = tf.image.resize_with_crop_or_pad(x["hr"], h, w)
        x = tf.reshape(x, (h, w, 3))
        return x
    return fn

def clean(h,w):
    def fn(x,y):
        x = tf.reshape(x, (-1, 480, 720, 3))
        y = tf.reshape(y, (-1, h, w, 3))
        x = tf.cast(x, tf.float16)
        y = tf.cast(y, tf.float16)
        return x,y
    return fn
    

def random_noise_and_resize(y):   
    x = tf.image.resize(y, size=[480, 720])
    sh = x.shape
    x = tf.numpy_function(apply_random_filters, inp=[x,], Tout=tf.uint8)
    x = tf.reshape(x, (-1, 480, 720, 3))
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    assert x.shape == y.shape, f"random_noise_and_resize: expected shapes to match {x.shape} and {y.shape}"
    return x, y

def div2k_ds(split, output_shape=(720, 1080), batch_size=1):
    h,w = output_shape
    def augment(img):
        img = augment_img(img)
        img = random_noise_and_resize(img)
        return clean(h,w)(*img)
    
    return tfds.load('div2k', split=split, shuffle_files=True)\
            .map(resize_data(h,w), num_parallel_calls=tf.data.AUTOTUNE, deterministic=False, name="resize_data_map")\
            .batch(batch_size)\
            .map(augment)

