import tensorflow as tf
from matplotlib import pyplot as plt
from IPython import display  # If using IPython, Colab or Jupyter
import datetime
import random
import datetime


def now():
    now = datetime.datetime.now()
    return now.strftime('%Y_%m_%d_T%H_%M_%S') + ('_%02d' % (now.microsecond / 10000))


def minute():
    now = datetime.datetime.now()
    return now.strftime('%H_%M')


class TBUtilCallback(tf.keras.callbacks.Callback):
    def __init__(self, tb_util):
        self.tb_util = tb_util

    def on_epoch_end(self, epoch, logs):
        self.tb_util.log_scalars(logs.items(), step=epoch)


class TensorboardUtil():
    def __init__(self, log_dir):
        self.log_dir = log_dir
        t = now()
        self.train_dir = f"{log_dir}/{t}/train"
        self.val_dir = f"{log_dir}/{t}/validation"
        self.file_writer = tf.summary.create_file_writer(self.train_dir)
        self.val_file_writer = tf.summary.create_file_writer(self.val_dir)

    def log_text(self, data, name, step=0):
        with self.file_writer.as_default():
            tf.summary.text(name, data, step=step, description=None)

    def log_scalar(self, value, name, step=0):
        if "val_" in name:
            with self.val_file_writer.as_default():
                tf.summary.scalar(name, value, step=step)
        else:
            with self.file_writer.as_default():
                tf.summary.scalar(name.replace("val_", ""), value, step=step)

    def log_scalars(self, names_and_values, step):
        train_scalars = filter(lambda x: "val_" not in x[0], names_and_values)
        val_scalars = filter(lambda x: "val_" in x[0], names_and_values)

        with self.file_writer.as_default():
            for name, value in train_scalars:
                tf.summary.scalar(name, value, step=step)

        with self.val_file_writer.as_default():
            for name, value in val_scalars:
                tf.summary.scalar(name.replace("val_", ""), value, step=step)

        self.file_writer.flush()
        self.val_file_writer.flush()

#         hp.KerasCallback(logdir, hparams)

    def save_image(self, image, label, step=0):
        with self.file_writer.as_default():
            # print(f"Saved {label} to Tensorboard")
            tf.summary.image(label, image, step=step)

    def get_callback(self, profile_batch=0):
        return TBUtilCallback(self)
        # return tf.keras.callbacks.TensorBoard(log_dir = self.log_dir,
        #               write_graph=True,
        #               histogram_freq = 0,
        #               profile_batch=profile_batch)


class ImageRenderer():
    def __init__(self, imgs_to_render, max_size=20):
        self.imgs_to_render = imgs_to_render
        self.tb_util = None
        self.save = False
        self.max_size = max_size

    def withTensorboard(self, tensorboard_util, tb_sample=4, tb_batch=4):
        self.tb_util = tensorboard_util
        self.tb_batch = tb_batch
        self.tb_sample = tb_sample
        return self

    def saveImages(self, path):
        self.save = True
        self.save_path = path
        return self

    def render(self, model, dataset, batch=1):
        imgs_to_render = self.imgs_to_render

        img_path = "./prepa/2304.png"
        raw_png = tf.io.read_file(str(img_path), name=img_path)
        pimage = tf.expand_dims(tf.image.decode_png(
            raw_png, channels=3, name=img_path), axis=0)
        # imgs_to_render.append(pimage, pimage)
        y_pred = model(pimage, training=False)
        y_pred = tf.cast(y_pred, tf.uint8)

        fig = plt.figure(figsize=(10, 20))
        gs = fig.add_gridspec(2, hspace=0)
        axs = gs.subplots()

        p = axs[0]
        p.imshow(pimage[0])
        p.set_title("With noise")
        p.axis('off')

        p = axs[1]
        p.imshow(y_pred[0])
        p.set_title("Denoised")
        p.axis('off')

        if self.tb_util is not None:
            self.tb_util.save_image(y_pred, f"denoised_prepa", batch)
            self.tb_util.save_image(pimage, f"noisy_prepa", batch)

        plt.figure()

        rows = imgs_to_render
        cols = 3
        plt.ion()
        # plt.show()
        fig, axs = plt.subplots(rows, cols)
        fig.subplots_adjust(wspace=0, hspace=0.5)
        fig.set_size_inches((self.max_size+15) / cols, self.max_size / rows)

        for i, (x, y) in zip(range(imgs_to_render), dataset):
            y_pred = model(x, training=False)
            x = tf.cast(x, tf.uint8)
            y_pred = tf.cast(y_pred, tf.uint8)
            y = tf.cast(y, tf.uint8)

            if self.tb_util is not None:
                self.tb_util.save_image(y_pred, f"denoised", batch)
                self.tb_util.save_image(x, f"noisy", batch)
                self.tb_util.save_image(y, f"original", batch)

            plt.subplot(cols, rows, i+1)
            p = axs[i, 0]
            p.imshow(x[0])
            p.set_title("With noise")
            p.axis('off')

            p = axs[i, 1]
            p.imshow(y_pred[0])
            p.set_title("Denoised")
            p.axis('off')

            p = axs[i, 2]
            p.imshow(y[0])
            p.set_title("Original")
            p.axis('off')

        plt.draw()
        plt.pause(0.001)


'''
Gradient accumulator
'''


class CustomTrainStep(tf.keras.Model):
    def __init__(self, n_gradients, autoencoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.autoencoder = autoencoder
        print(f"CustomTrainStep: n_gradients = {n_gradients}")
        self.n_gradients = tf.Variable(
            n_gradients, dtype=tf.int32, trainable=False)
        self.n_acum_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.update_count = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.gradient_accumulation = [tf.Variable(tf.zeros_like(v, dtype=tf.float32),
                                                  trainable=False) for v in self.trainable_variables]

    def call(self, data):
        return self.autoencoder(data, training=False)

    def train_step(self, data):
        self.n_acum_step.assign_add(1)

        x, y = data
        # Gradient Tape
        with tf.GradientTape() as tape:
            y_pred = self.autoencoder(x, training=True)
            loss = self.compiled_loss(
                y, y_pred, regularization_losses=self.losses)
            scaled_loss = self.optimizer.get_scaled_loss(loss)

        # Calculate batch gradients
        scaled_grads = tape.gradient(
            scaled_loss, self.autoencoder.trainable_variables)
        gradients = self.optimizer.get_unscaled_gradients(scaled_grads)
        # Accumulate batch gradients
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign_add(gradients[i])

        # If n_acum_step reach the n_gradients then we apply accumulated gradients to update the variables otherwise do nothing
        tf.cond(tf.equal(self.n_acum_step, self.n_gradients),
                self.apply_accu_gradients, lambda: None)

        # update metrics
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def apply_accu_gradients(self):
        # apply accumulated gradients
        self.optimizer.apply_gradients(
            zip(self.gradient_accumulation, self.autoencoder.trainable_variables))
        self.update_count.assign_add(1)

        # reset
        self.n_acum_step.assign(0)
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign(tf.zeros_like(
                self.autoencoder.trainable_variables[i], dtype=tf.float32))
