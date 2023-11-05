import numpy as np

import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras import Model, Input

import matplotlib.pyplot as plt

import os
import datetime

print(tf.__version__)


class SRGAN():
    def __init__(self, lr_size, hr_size):
        super(SRGAN, self).__init__()
        self.c = 3
        self.lr_size = lr_size
        self.hr_size = hr_size
        self.residual_blocks_count = 16
        patch = int(self.hr_size[0] / 2 ** 4)
        self.disc_patch = (patch, patch, 1)
        self.gf = 64
        self.df = 64
        optimizer = Adam(.0002, .5)
        self.vgg_part = self.build_vgg()
        self.vgg_part.trainable = False
        self.vgg_part.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        self.generator = self.build_generator()
        img_hr = Input(shape=self.hr_size)
        img_lr = Input(shape=self.lr_size)
        fake_hr = self.generator(img_lr)
        fake_features = self.vgg_part(fake_hr)
        self.discriminator.trainable = False
        validity = self.discriminator(fake_hr)
        self.combined = Model([img_lr, img_hr], [validity, fake_features])
        self.combined.compile(loss=['binary_crossentropy', 'mse'], loss_weights=[1e-3, 1], optimizer=optimizer)

    def build_vgg(self):
        vgg = tf.keras.applications.VGG19(weights="imagenet")
        vgg.outputs = [vgg.layers[9].output]
        iut = Input(shape=self.hr_size)
        out = vgg(iut)
        return Model(iut, out)

    def build_generator(self):

        def residual_block(iut):
            out = layers.Conv2D(64, kernel_size=3, strides=1, padding='same')(iut)
            out = layers.Activation('relu')(out)
            out = layers.BatchNormalization(momentum=0.8)(out)
            out = layers.Conv2D(64, kernel_size=3, strides=1, padding='same')(out)
            out = layers.BatchNormalization(momentum=0.8)(out)
            out = layers.Add()([out, iut])
            return out

        def deconv2d(iut):
            out = layers.UpSampling2D(size=2)(iut)
            out = layers.Conv2D(256, kernel_size=3, strides=1, padding='same')(out)
            out = layers.Activation('relu')(out)
            return out

        iut = Input(shape=self.lr_size)
        out = layers.Conv2D(64, kernel_size=9, strides=1, padding='same')(iut)
        out = layers.Activation('relu')(out)
        out_1 = out
        for _ in range(self.residual_blocks_count):
            out = residual_block(out)
        out = layers.Conv2D(64, kernel_size=3, strides=1, padding='same')(out)
        out_2 = layers.BatchNormalization(momentum=0.8)(out)
        out = layers.Add()([out_2, out_1])
        out = deconv2d(out)
        out = deconv2d(out)
        out = layers.Conv2D(self.c, kernel_size=9, strides=1, padding='same', activation='tanh')(out)
        return Model(iut, out)

    def build_discriminator(self):

        def d_block(iut, filter, strides=1, bn=True):
            out = layers.Conv2D(filter, kernel_size=3, strides=strides, padding='same')(iut)
            out = LeakyReLU(.2)(out)
            if bn:
                out = layers.BatchNormalization(.8)(out)
            return out

        iut = Input(self.hr_size)
        out = d_block(iut, self.df, bn=False)
        out = d_block(out, self.df, strides=2)
        out = d_block(out, self.df * 2)
        out = d_block(out, self.df * 2, strides=2)
        out = d_block(out, self.df * 4)
        out = d_block(out, self.df * 4, strides=2)
        out = d_block(out, self.df * 8)
        out = d_block(out, self.df * 8, strides=2)
        out = layers.Dense(self.df * 16)(out)
        out = LeakyReLU(.2)(out)
        out = layers.Dense(1, activation='sigmoid')(out)
        return Model(iut, out)

    def train(self, epoch_count, batch_size=1, sample_interval=50):
        start_time = datetime.datetime.now()
        for epoch in range(epoch_count):
            hr_images, lr_images = self.data_loader.load_data(batch_size)
            fake_hr_images = self.generator.predict(lr_images)
            valid = np.ones((batch_size,) + self.disc_patch)
            fake = np.zeros((batch_size,) + self.disc_patch)
            d_loss_real = self.discriminator.train_on_batch(hr_images, valid)
            d_loss_fake = self.discriminator.train_on_batch(fake_hr_images, fake)
            hr_images, lr_images = self.data_loader.load_data(batch_size)
            valid = np.ones((batch_size,) + self.disc_patch)
            image_features = self.vgg_part.predict(hr_images)
            elapsed_time = datetime.datetime.now() - start_time
            print("%d time: %s" % (epoch, elapsed_time))
            if epoch % sample_interval == 0:
                self.sample_images(epoch)
            d_loss = .5 * np.add(d_loss_real, d_loss_fake)
            g_loss = self.combined.train_on_batch([lr_images, hr_images], [valid, image_features])

    def sample_images(self, epoch):
        os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
        hr_images, lr_images = self.data_loader.load_data(batch_size=2, is_testing=True)
        fake_hr_images = self.generator.predict(lr_images)
        lr_images = .5 * lr_images + .5
        hr_images = .5 * hr_images + .5
        fake_hr_images = .5 * fake_hr_images + .5
        titles = ['Generated', 'Original']
        fig, axs = plt.subplots(2, 2)
        count = 0
        for row in range(2):
            for col, image in enumerate([fake_hr_images, hr_images]):
                axs[row, col].imshow(image[row])
                axs[row, col].set_title(titles[col])
                axs[row, col].axis('off')
            count += 1
        fig.savefig("images/%s/%d.png" % (self.dataset_name, epoch))
        plt.close()
        for i in range(2):
            fig = plt.figure()
            plt.imshow(lr_images[i])
            fig.savefig('images/%s/%d_%d.png' % (self.dataset_name, epoch, i))
            plt.close()


if __name__ == '__main__':
    net = SRGAN((64, 64), (256, 256))
    net.train(epoch_count=30000, batch_size=1, sample_interval=50)
