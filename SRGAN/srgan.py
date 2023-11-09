import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

import os
import datetime

from SRGAN.data_loader import load_data


class SRGAN(keras.Model):
    def __init__(self):
        super(SRGAN, self).__init__()
        self.channels = 3
        self.lr_height = 120
        self.lr_width = 120
        self.lr_shape = (self.lr_height, self.lr_width, self.channels)
        self.hr_height = self.lr_height * 4
        self.hr_width = self.lr_width * 4
        self.hr_shape = (self.hr_height, self.hr_width, self.channels)
        self.n_residual_blocks = 16

        optimizer = tf.optimizers.Adam(0.0002, 0.5)
        self.vgg = self.build_vgg()
        self.vgg.trainable = False
        self.vgg.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        self.load_data = load_data

        patch = int(self.hr_height / 2 ** 4)
        self.disc_patch = (patch, patch, 1)

        self.gf = 64
        self.df = 64
        self.discriminator = self.build_discriminator()
        self.discriminator.trainable = False
        self.discriminator.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

        self.generator = self.build_generator()
        # self.generator.compile(loss='mse',optimizer=optimizer,metrics=['accuracy'])

        img_hr = keras.Input(shape=self.hr_shape)
        img_lr = keras.Input(shape=self.lr_shape)

        fake_hr = self.generator(img_lr)

        fake_features = self.vgg(fake_hr)

        validity = self.discriminator(fake_hr)

        self.combined = keras.Model([img_lr, img_hr], [validity, fake_features])
        self.combined.compile(loss=['binary_crossentropy', 'mse'], loss_weights=[1e-3, 1], optimizer=optimizer)

    def build_vgg(self):
        vgg = keras.applications.VGG19(weights="imagenet", include_top=False, input_shape=(self.hr_height, self.hr_width, self.channels))
        vgg = keras.Model(inputs=vgg.input, outputs=[vgg.layers[9].output])
        img = keras.Input(shape=self.hr_shape)
        img_features = vgg(img)
        return keras.Model(img, img_features)

    def build_generator(self):

        def residual_block(layer_input, filters):
            d = keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding='same')(layer_input)
            d = keras.layers.Activation('relu')(d)
            d = keras.layers.BatchNormalization(momentum=0.8)(d)
            d = keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding='same')(d)
            d = keras.layers.BatchNormalization(momentum=0.8)(d)
            d = keras.layers.Add()([d, layer_input])
            return d

        def deconv2d(layer_input):
            u = keras.layers.UpSampling2D(size=2)(layer_input)
            u = keras.layers.Conv2D(256, kernel_size=3, strides=1, padding='same')(u)
            u = keras.layers.Activation('relu')(u)
            return u

        img_lr = keras.Input(shape=self.lr_shape)

        c1 = keras.layers.Conv2D(64, kernel_size=9, strides=1, padding='same')(img_lr)
        c1 = keras.layers.Activation('relu')(c1)

        r = residual_block(c1, self.gf)
        for _ in range(self.n_residual_blocks - 1):
            r = residual_block(r, self.gf)

        c2 = keras.layers.Conv2D(64, kernel_size=3, strides=1, padding='same')(r)
        c2 = keras.layers.BatchNormalization(momentum=0.9)(c2)
        c2 = keras.layers.Add()([c2, c1])

        u1 = deconv2d(c2)
        u2 = deconv2d(u1)
        gen_hr = keras.layers.Conv2D(self.channels, kernel_size=9, strides=1, padding='same', activation='tanh')(u2)

        return keras.Model(img_lr, gen_hr)

    def build_discriminator(self):

        def d_block(layer_input, filters, strides=1, bn=True):
            d = keras.layers.Conv2D(filters, kernel_size=3, strides=strides, padding='same')(layer_input)
            d = keras.layers.LeakyReLU(alpha=0.2)(d)
            if bn:
                d = keras.layers.BatchNormalization(momentum=0.8)(d)
            return d

        d0 = keras.Input(shape=self.hr_shape)

        d1 = d_block(d0, self.df, bn=False)
        d2 = d_block(d1, self.df, strides=2)
        d3 = d_block(d2, self.df * 2)
        d4 = d_block(d3, self.df * 2, strides=2)
        d5 = d_block(d4, self.df * 4)
        d6 = d_block(d5, self.df * 4, strides=2)
        d7 = d_block(d6, self.df * 8)
        d8 = d_block(d7, self.df * 8, strides=2)

        d9 = keras.layers.Dense(self.df * 16)(d8)
        d10 = keras.layers.LeakyReLU(alpha=0.2)(d9)
        validity = keras.layers.Dense(1, activation='sigmoid')(d10)

        return keras.Model(d0, validity)

    def train(self, epochs, batch_size=1, sample_interval=50):
        start_time = datetime.datetime.now()
        for epoch in range(epochs):

            imgs_hr, imgs_lr = self.load_data(batch_size, epoch)
            fake_hr = self.generator.predict(imgs_lr)
            valid = np.ones((batch_size,) + self.disc_patch)
            fake = np.zeros((batch_size,) + self.disc_patch)
            d_loss_real = self.discriminator.train_on_batch(imgs_hr, valid)
            d_loss_fake = self.discriminator.train_on_batch(fake_hr, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            imgs_hr, imgs_lr = self.load_data(batch_size, epoch)
            valid = np.ones((batch_size,) + self.disc_patch)
            image_features = self.vgg.predict(imgs_hr)
            g_loss = self.combined.train_on_batch([imgs_lr, imgs_hr], [valid, image_features])
            elapsed_time = datetime.datetime.now() - start_time
            print("%d time: %s" % (epoch, elapsed_time))
            if epoch % sample_interval == 0:
                self.sample_images(epoch)
                save_dir = ('../saves/' + str(epoch))
                os.makedirs(save_dir)
                model_save_dir = (save_dir + "/model.h5")
                self.generator.save(model_save_dir)

    def sample_images(self, epoch):
        os.makedirs('images/div2k/', exist_ok=True)
        r, c = 2, 2

        imgs_hr, imgs_lr = self.load_data(2, epoch)
        fake_hr = self.generator.predict(imgs_lr)

        imgs_lr = 0.5 * imgs_lr + 0.5
        fake_hr = 0.5 * fake_hr + 0.5
        imgs_hr = 0.5 * imgs_hr + 0.5
        titles = ['Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for row in range(r):
            for col, image in enumerate([fake_hr, imgs_hr]):
                axs[row, col].imshow(image[row])
                axs[row, col].set_title(titles[col])
                axs[row, col].axis('off')
            cnt += 1
        fig.savefig("./images/div2k/%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    gan = SRGAN()
    gan.train(epochs=100000, batch_size=1, sample_interval=1000)
