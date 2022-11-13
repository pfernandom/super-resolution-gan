import random
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow as tf

from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Input, Flatten,\
                                    Reshape, LeakyReLU as LR,\
                                    Activation, Dropout
from tensorflow.keras.models import Model, Sequential
from matplotlib import pyplot as plt
from IPython import display # If using IPython, Colab or Jupyter
import numpy as np
import tensorflow_addons as tfa
import datetime
import random
import time

class NoiseUtil:
    @staticmethod
    def filter_pixel(downsize_image_ratios = [1/16]):
        def fn(img):
            downsize_image_ratio = random.choice(downsize_image_ratios)
            resized_size_h = img.shape[1]
            resized_size_w = img.shape[2]
            
            # noisy = normalize(img) + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=(img.shape))
            # noisy = layers.MaxPooling2D(pool_size = (8, 8), padding='same')(tf.expand_dims(img, axis=0))
            down = tf.image.resize(
                img,
                [int(resized_size_h * downsize_image_ratio), int(resized_size_w * downsize_image_ratio)],
                preserve_aspect_ratio=True,
                antialias=False,
                name=None)

            up = tf.image.resize(
                down,
                [resized_size_h, resized_size_w],
                preserve_aspect_ratio=True,
                antialias=False,
                name=None)

            up = tf.cast((up), img.dtype)
            return up
        return fn

    @staticmethod
    def random_pixel_noise(img2, filter_size, patch_size, filter_fn, shape):
        
        if (len(tf.shape(img2)) < 4):
            img2 = tf.convert_to_tensor(img2[:,:,:])
            img2 = tf.expand_dims(img2, axis=0)
        else:
            img2 = tf.convert_to_tensor(img2[:,:,:,:])
  
        img2_filtered = filter_fn(img2)
        print(f"shape={shape}")
        shp = shape
        n = shp[0]
        h = shp[1]
        w = shp[2]
        c = shp[3]

        print(h)

        # mask = build_negative_filter(n, h, w, c, filter=patch_size, masks=filters)
        # x = tf.random.uniform(
        #     tf.shape(img2),
        #     minval=0,
        #     maxval=2,
        #     dtype=tf.int32,
        # )
        mask = tf.reshape(tf.range(h*w*c), (h,w,c))

        i = random.randint(0,h-patch_size)
        j = random.randint(0,w-patch_size)

        a = tf.math.logical_and(mask / h <= j+patch_size, mask % h <= i+patch_size)
        b = tf.math.logical_and(mask / w > j, mask % w > i)
        d = tf.math.logical_and(a,b)

        for i in range(filter_size):
            i = random.randint(0,h-patch_size)
            j = random.randint(0,w-patch_size)

            a = tf.math.logical_and(mask / h <= j+patch_size, mask % h <= i+patch_size)
            b = tf.math.logical_and(mask / w > j, mask % w > i)
            c = tf.math.logical_and(a,b)
            d = tf.math.logical_or(d,c)


        # im = tf.reshape(tf.constant(range(2*12*12*1)), (2,12,12,1))
        img3 = tf.where(d, tf.zeros_like(img2), img2)
        # img3 = tf.where(x == 0, img2, tf.zeros_like(img2))

        # img3 = tf.math.multiply(img2, mask)
        img4 = img2_filtered * 2
        img2 = img3 + img4 - img2_filtered


        # starts = random.sample(range(10000000), filters)
        
        # for i in range(filters):
        #     assert patch_size > 0
        #     assert patch_size % 2 == 0
        #     assert patch_size < img2.shape[1]-4
        #     assert patch_size < img2.shape[2]-4

        #     random.seed(time.time())

        #     s1 = random.randint(0,img2.shape[1]-patch_size-4)
        #     s2 = random.randint(0,img2.shape[2]-patch_size-4)
            
        #     img3 = tfa.image.cutout(img2, mask_size=[patch_size,patch_size], offset= [s1,s2])
        #     img4 = tfa.image.cutout(img2_filtered, mask_size=[patch_size,patch_size], offset= [s1,s2])

        #     img2 = img3 + img2_filtered - img4
        
        return img2

    @staticmethod
    def pixel_noise(img, filters, filter_size):
        return NoiseUtil.random_pixel_noise(img, filters, filter_size, filter_fn=NoiseUtil.filter_pixel(downsize_image_ratios = [1/4]), shape=img.shape)


class ImgUtils():
    @staticmethod
    def normalize(image, as_probs=False):
        a = (tf.cast(image, tf.float32) - 127.5) if not as_probs else tf.cast(image, tf.float32)
        return a / 127.5

    @staticmethod
    def denormalize(image, cast=False):
        # return image
        img = tf.cast(image, tf.float32) * 127.5 + 127.5
        img = tf.clip_by_value(img, 0.0, 255.0)
        if cast:
            return tf.cast(img, cast)
        return img

im = np.random.normal(loc=0.0, scale=1.0, size=(1,28,28,1))
im = tf.cast(im, tf.float32) * 127.5 + 127.5
im = tf.clip_by_value(im, 0.0, 255.0)

im_processed = ImgUtils.denormalize(ImgUtils.normalize(im), cast=im.dtype)

assert np.any(im == im_processed)


class DataLoader():

    @staticmethod
    def one_from_generator(fn, shape):
        output_signature=tf.TensorSpec(shape=shape, dtype=tf.float32)
        return tf.data.Dataset.from_generator(fn, output_signature=output_signature)

    @staticmethod
    def two_from_generator(fn, shape):
        output_signature=tf.TensorSpec(shape=shape, dtype=tf.float32)
        return tf.data.Dataset.from_generator(fn, output_signature=output_signature).map(lambda x: (x,x))

class DataManager():
    def __init__(self, train_ds, test_ds = None):
        self.train_ds = train_ds
        self.test_ds = test_ds

    def set_test_samples(self, test_samples=25):
        self.test_samples = test_samples

    def set_transform(self, fn):
        self.transform = fn

    @staticmethod
    def create_label_with_input_transform(train_generator, test_generator, input_shape, transform_fn=lambda x:x, test_samples=25):
        def normalize_both(x,y):
            return ImgUtils.normalize(x), ImgUtils.normalize(y)
        
        train_ds = DataLoader.two_from_generator(train_generator, input_shape).map(normalize_both).skip(test_samples)
        test_ds = None
        if test_generator:
            test_ds = DataLoader.two_from_generator(test_generator, input_shape).map(normalize_both).take(test_samples)

        dm = DataManager(train_ds, test_ds)
        dm.set_test_samples(test_samples)
        dm.set_transform(transform_fn)

        return dm

    def get_test_data(self, batch_size):
        return self.test_ds.take(batch_size).batch(batch_size).map(self.transform)

    def get_training_data(self, batch_size):
        return self.train_ds.take(batch_size).batch(batch_size).map(self.transform)

    def print_validation(self, model=lambda x:x, batch_size=5, save=False, path="./"):
        random.seed(10)
        rows = batch_size
        cols = 2

        results = [(model(x), y) for x,y in self.get_test_data(batch_size)]
        plt.figure(figsize=(rows * 2, cols * 2))
        for x,y in results:
            assert x.shape == y.shape
            for i in range(x.shape[0]):
                im = x[i,:,:,:]
                plt.subplot(cols, rows, i+1)
                plt.imshow(ImgUtils.denormalize(im, cast=tf.uint8))
                plt.axis('off')
           

            for i in range(y.shape[0]):
                im = y[i,:,:,:]
                plt.subplot(cols, rows, i+x.shape[0]+1)
                plt.imshow(ImgUtils.denormalize(im, cast=tf.uint8))
                plt.axis('off')
        plt.subplots_adjust(wspace = 0, hspace = 0.5)
        if save:
            plt.savefig(path)
        plt.show()
           
# def train_get():
#    for x, y in zip(x_train, y_train):
#         x = tf.expand_dims(x, axis=2)
#         yield x

# def test_get():
#    for x in x_test:
#         x = tf.expand_dims(x, axis=2)
#         yield x


# def add_noise(x,y):
#     n = NoiseUtil.pixel_noise(x, 30, 2)
#     return n,y
# dm = DataManager.create_label_with_input_transform(train_get, test_get, (28,28,1), add_noise)
# dm.print_validation() 