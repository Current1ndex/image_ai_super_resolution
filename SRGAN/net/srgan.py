import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras import Model, Input

print(tf.__version__)


class SRGAN():
    def __init__(self, lr_size, hr_size):
        super(SRGAN, self).__init__()
        self.c = 3
        self.lr_size = lr_size
        self.hr_size = hr_size
        self.residual_blocks_count = 16
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


srgan = SRGAN((1, 1), (4, 4))
