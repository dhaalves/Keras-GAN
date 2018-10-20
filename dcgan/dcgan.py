from __future__ import print_function, division

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from tqdm import trange

from utils import normalize


class DCGAN():
    def __init__(self, images_folder, epochs, batch_size):
        # Input shape
        self.epochs = epochs
        self.images_folder = images_folder
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100
        self.batch_size = batch_size
        self.train_generator = self.load_dataset(images_folder)
        self.num_classes = self.train_generator.num_classes
        self.gen_first_dense = int(self.img_rows / 4)

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


    def load_dataset(self, images_folder):
        def normalize(img):
            return (img.astype(np.float32) - 127.5) / 127.5

        train_datagen = ImageDataGenerator(
            preprocessing_function=normalize)
        train_generator = train_datagen.flow_from_directory(
            images_folder,
            shuffle=True,
            target_size=(self.img_rows, self.img_cols),
            batch_size=self.batch_size,
            class_mode='categorical')
        return train_generator

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

    def train(self, save_interval=50):


        for epoch in trange(self.epochs):

            batches = 0
            for i, (imgs, labels) in enumerate(self.train_generator):

                batches += 1
                if batches >= self.train_generator.samples / self.batch_size:
                    break

                step_num_imgs = len(imgs)
                valid = np.ones((step_num_imgs, 1))
                fake = np.zeros((step_num_imgs, 1))

                # Sample noise and generate a batch of new images
                noise = np.random.normal(0, 1, (step_num_imgs, self.latent_dim))
                gen_imgs = self.generator.predict(noise)

                # Train the discriminator (real classified as ones and generated as zeros)
                d_loss_real = self.discriminator.train_on_batch(imgs, valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # Train the generator (wants discriminator to mistake images as real)
                g_loss = self.combined.train_on_batch(noise, valid)

                # Plot the progress
                # print("%d - %d/%d - [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (
                #     epoch, (i + 1) * len(imgs), train_generator.samples, d_loss[0], 100 * d_loss[1], g_loss))

            self.save_imgs(epoch)
            self.save_model()

    def generate_img(self, epoch, size=10):
        noises = np.random.normal(0, 1, (size, 100))
        gen_imgs = self.generator.predict(noises)
        for i, img in enumerate(gen_imgs):
            img = normalize(img)
            # img_norm = cv2.normalize(img_gen, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            # img_norm = img_norm.astype(np.uint8)
            # img_norm_rgb = cv2.cvtColor(img_norm, cv2.COLOR_BGR2RGB)
            # cv2.imwrite('color_img_norm_cv.jpg', img_norm_rgb)
            Image.fromarray(img, 'RGB').save('images/%s_%d.png' % (epoch, i))

    def save_imgs(self, epoch):
        for i in self.train_generator.class_indices:
            self.generate_img(i, epoch)

    def save_model(self):
        def save(model, model_name):
            model_path = "saved_model/%s" % model_name
            model.save(model_path)
        save(self.generator, "generator")
        save(self.discriminator, "discriminator")

if __name__ == '__main__':

    dataset_path = '/mnt/sdb1/datasets/parasites/parasites_eggs_8'
    dcgan = DCGAN(images_folder=dataset_path, epochs=300, batch_size=32)
    dcgan.train(save_interval=50)
    # generate_img('saved_model/generator_leaves.hdf5', size=10)
