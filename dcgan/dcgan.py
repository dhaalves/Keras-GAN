from __future__ import print_function, division

import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from keras.datasets import mnist
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tqdm import trange


class DCGAN():
    def __init__(self, images_folder):
        # Input shape
        self.images_folder = images_folder
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 3
        self.gen_first_dense = int(self.img_rows / 4)
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(
            Dense(128 * self.gen_first_dense * self.gen_first_dense, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((self.gen_first_dense, self.gen_first_dense, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        # model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, save_interval=50):

        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            validation_split=0.2)

        for epoch in trange(epochs):

            train_generator = train_datagen.flow_from_directory(
                self.images_folder,
                target_size=(self.img_rows, self.img_cols),
                batch_size=batch_size,
                subset='training',
                class_mode='categorical')

            # val_generator = train_datagen.flow_from_directory(
            #     '/mnt/sdb1/datasets/leaves/leaves1_200x200',
            #     target_size=(self.img_rows, self.img_cols),
            #     batch_size=batch_size,
            #     subset='validation',
            #     class_mode='categorical')

            batches = 0
            for i, (imgs, labels) in enumerate(train_generator):

                batches += 1
                if batches >= train_generator.samples / batch_size:
                    break

                valid = np.ones((len(imgs), 1))
                fake = np.zeros((len(imgs), 1))

                # Sample noise and generate a batch of new images
                noise = np.random.normal(0, 1, (len(imgs), self.latent_dim))
                gen_imgs = self.generator.predict(noise)

                # Train the discriminator (real classified as ones and generated as zeros)
                d_loss_real = self.discriminator.train_on_batch(imgs, valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # Train the generator (wants discriminator to mistake images as real)
                g_loss = self.combined.train_on_batch(noise, valid)

                # Plot the progress
                print("%d - %d/%d - [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (
                    epoch, (i + 1) * len(imgs), train_generator.samples, d_loss[0], 100 * d_loss[1], g_loss))

            self.save_imgs(epoch)
            self.save_model()

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        # gen_imgs = 0.5 * gen_imgs + 0.5
        # gen_imgs *= 255

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :])  # , 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("images/leaves_%d.png" % epoch)
        plt.close()

    def save_model(self):

        def save(model, model_name):
            model_path = "saved_model/%s.json" % model_name
            weights_path = "saved_model/%s_weights.hdf5" % model_name
            options = {"file_arch": model_path,
                       "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.generator, "generator")
        save(self.discriminator, "discriminator")


def get_gen_img(model):
    model.generator.load_weights('saved_model/generator_weights.hdf5')
    noise = np.random.normal(0, 1, (1, 100))
    img = dcgan.generator.predict(noise)
    plt.imshow(img[0])
    plt.axis('off')
    plt.savefig('test.png')


if __name__ == '__main__':

    dataset_path = 'C:\\Users\\daniel\\Downloads\\leaves1\\'
    dcgan = DCGAN(dataset_path)
    dcgan.train(epochs=300, batch_size=32, save_interval=50)
