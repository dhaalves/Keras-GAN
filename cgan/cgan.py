from __future__ import print_function, division

from PIL import Image
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from tqdm import trange

from utils import normalize


class CGAN():
    def __init__(self, images_folder, epochs, batch_size):
        # Input shape
        self.epochs = epochs
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
        self.discriminator.compile(loss=['binary_crossentropy'],
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        img = self.generator([noise, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid = self.discriminator([img, label])

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model([noise, label], valid)
        self.combined.compile(loss=['binary_crossentropy'],
                              optimizer=optimizer)

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

    # def build_generator(self):
    #
    #     model = Sequential()
    #
    #     model.add(
    #         Dense(128 * self.gen_first_dense * self.gen_first_dense, activation="relu", input_dim=self.latent_dim))
    #     model.add(Reshape((self.gen_first_dense, self.gen_first_dense, 128)))
    #     model.add(UpSampling2D())
    #     model.add(Conv2D(128, kernel_size=3, padding="same"))
    #     model.add(BatchNormalization(momentum=0.8))
    #     model.add(Activation("relu"))
    #     model.add(UpSampling2D())
    #     model.add(Conv2D(64, kernel_size=3, padding="same"))
    #     model.add(BatchNormalization(momentum=0.8))
    #     model.add(Activation("relu"))
    #     model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
    #     model.add(Activation("tanh"))
    #
    #     model.summary()
    #
    #     # noise = Input(shape=(self.latent_dim,))
    #     # img = model(noise)
    #     #
    #     # return Model(noise, img)
    #
    #     noise = Input(shape=(self.latent_dim,))
    #     label = Input(shape=(1,), dtype='int32')
    #     label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))
    #
    #     model_input = multiply([noise, label_embedding])
    #     img = model(model_input)
    #
    #     return Model([noise, label], img)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))

        model_input = multiply([noise, label_embedding])
        img = model(model_input)

        return Model([noise, label], img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Dense(512, input_dim=np.prod(self.img_shape)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.img_shape)
        label = Input(shape=(1,), dtype='int32')

        label_embedding = Flatten()(Embedding(self.num_classes, np.prod(self.img_shape))(label))
        flat_img = Flatten()(img)

        model_input = multiply([flat_img, label_embedding])

        validity = model(model_input)

        return Model([img, label], validity)

    # def build_discriminator(self):
    #
    #     model = Sequential()
    #
    #     model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
    #     model.add(LeakyReLU(alpha=0.2))
    #     model.add(Dropout(0.25))
    #     model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    #     model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
    #     model.add(BatchNormalization(momentum=0.8))
    #     model.add(LeakyReLU(alpha=0.2))
    #     model.add(Dropout(0.25))
    #     model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    #     model.add(BatchNormalization(momentum=0.8))
    #     model.add(LeakyReLU(alpha=0.2))
    #     model.add(Dropout(0.25))
    #     model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    #     model.add(BatchNormalization(momentum=0.8))
    #     model.add(LeakyReLU(alpha=0.2))
    #     model.add(Dropout(0.25))
    #     model.add(Flatten())
    #     model.add(Dense(1, activation='sigmoid'))
    #
    #     # model.summary()
    #
    #     img = Input(shape=self.img_shape)
    #     validity = model(img)
    #
    #     return Model(img, validity)

    def train(self, sample_interval=1):

        for epoch in range(self.epochs):

            print('Epoch %d/%d' % (epoch, self.epochs))
            batches = 0
            for i, (imgs, labels) in enumerate(self.train_generator):

                batches += 1
                if batches >= self.train_generator.samples / self.batch_size:
                    break

                step_num_imgs = len(imgs)
                valid = np.ones((step_num_imgs, 1))
                fake = np.zeros((step_num_imgs, 1))

                # Sample noise as generator input
                noise = np.random.normal(0, 1, (step_num_imgs, 100))

                # Generate a half batch of new images
                labels = np.argmax(labels, axis=1)
                gen_imgs = self.generator.predict([noise, labels])

                # Train the discriminator
                d_loss_real = self.discriminator.train_on_batch([imgs, labels], valid)
                d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # Condition on labels
                sampled_labels = np.random.randint(0, 10, step_num_imgs).reshape(-1, 1)

                # Train the generator
                g_loss = self.combined.train_on_batch([noise, sampled_labels], valid)

                # Plot the progress
                # print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.save_imgs(epoch)
                self.save_model()



    def generate_img(self, class_name, epoch, size=1):
        noises = np.random.normal(0, 1, (size, 100))
        sampled_labels = np.array([self.train_generator.class_indices[class_name]] * size)
        gen_imgs = self.generator.predict([noises, sampled_labels])
        for i, img in enumerate(gen_imgs):
            img = normalize(img)
            # img_norm = cv2.normalize(img_gen, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            # img_norm = img_norm.astype(np.uint8)
            # img_norm_rgb = cv2.cvtColor(img_norm, cv2.COLOR_BGR2RGB)
            # cv2.imwrite('color_img_norm_cv.jpg', img_norm_rgb)
            Image.fromarray(img, 'RGB').save('images/%s_%d_%d.png' % (class_name, epoch, i))

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
    cgan = CGAN(images_folder='/mnt/sdb1/datasets/leaves/leaves1_200x200', epochs=100, batch_size=32)
    cgan.train()
