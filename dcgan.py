"""
Based on https://github.com/jacobgil/keras-dcgan

Changes:
- Adds a third agent which classifies generated digits
- Generator given an input class as well as random IV
- Gave the generator some more neurons

Agents:

- A generator which takes a random signal + a digit class
- A discriminator which is given images and has to decide if they are fake or not
- A classifier which is given generated images and classifies them with a digit class

"""
import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Dropout
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD, Adam
from keras.datasets import mnist
from keras import backend as K
import numpy as np
from PIL import Image
import argparse
import math

# input image dimensions
from keras.utils import np_utils

img_rows, img_cols = 28, 28


def generator_model():
    iv = Input(shape=(100,))
    class_input = Input(shape=(10,), name='class_input')

    # embedding layer to encode the random IV sequence
    iv_embedding = Dense(units=1024, activation='tanh')(iv)

    x = keras.layers.concatenate([iv_embedding, class_input])

    x = Dense(128*7*7, activation='relu')(x)
    x = Dense(128*7*7, activation='relu')(x)

    x = BatchNormalization()(x)
    x = Activation('tanh')(x)
    x = Reshape((7, 7, 128), input_shape=(128*7*7,))(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(64, kernel_size=5, padding='same', activation='tanh')(x)
    x = UpSampling2D(size=(2, 2))(x)
    out = Conv2D(1, kernel_size=5, padding='same', activation='tanh')(x)
    model = Model(inputs=[iv, class_input], outputs=out)

    return model


def classifier_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32,
                     kernel_size=3,
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(Conv2D(32, kernel_size=5, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    return model


def discriminator_model(input_shape):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=5, padding='same', input_shape=input_shape))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=5))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model


def generator_containing_discriminator(generator, discriminator):
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    return model


def generator_containing_classifier(generator, classifier):
    return combine_model(classifier, generator)


def combine_model(uses_generated, generator):
    model = Sequential()
    model.add(generator)
    uses_generated.trainable = False
    model.add(uses_generated)
    return model


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = (28, 28)
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        if K.image_data_format() == 'channels_first':
            image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
                img[0, :, :]
        else:
            image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
                img[:, :, 0]
    return image


def train(batch_size, epochs, save_epoch_weights, path=''):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape
    if K.image_data_format() == 'channels_first':
        print("Backend prefers channels first")
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        print("Channels last")
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = (x_train.astype(np.float32) - 127.5)/127.5

    discriminator = discriminator_model(input_shape)
    classifier = classifier_model(input_shape)

    generator = generator_model()

    discriminator_on_generator = generator_containing_discriminator(generator, discriminator)
    classifier_on_generator = generator_containing_classifier(generator, classifier)

    #d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    #g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)

    g_optim = Adam(lr=0.0001)
    g_c_optim = Adam(lr=0.0001)
    g_d_optim = Adam(lr=0.0001)
    c_optim = Adam()
    d_optim = Adam(lr=0.0001)
    generator.compile(loss='binary_crossentropy', optimizer=g_optim)
    discriminator_on_generator.compile(loss='binary_crossentropy', optimizer=g_d_optim)
    classifier_on_generator.compile(loss='categorical_crossentropy', optimizer=g_c_optim)

    discriminator.trainable = True
    discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)

    classifier.trainable = True
    classifier.compile(loss='categorical_crossentropy', optimizer=c_optim)

    noise = np.zeros((batch_size, 100))

    print("{} batches per epoch".format(x_train.shape[0] // batch_size))
    for epoch in range(epochs):
        for index in range(int(x_train.shape[0]/batch_size)):
            for i in range(batch_size):
                noise[i, :] = np.random.uniform(-1, 1, 100)
            image_batch = x_train[index * batch_size:(index + 1) * batch_size]

            # sample classes here and get the generator to make fake images
            classes = np_utils.to_categorical(np.random.randint(0, 10, batch_size), 10)
            generated_images = generator.predict([noise, classes], verbose=0)

            if index % 50 == 0:
                image = combine_images(generated_images[:25])
                image = image*127.5+127.5
                filename = '{}/{}_{}.png'.format(path, epoch, index)
                Image.fromarray(image.astype(np.uint8)).save(filename)

            # Compute the classifier's loss on the generated images
            # The classifier doesn't see the raw images.
            X = generated_images
            y = classes
            c_loss = classifier.train_on_batch(X, y)
            classifier.trainable = False

            # Compute the discriminator's loss using the generated images:
            X = np.concatenate((image_batch, generated_images))
            y = [1] * batch_size + [0] * batch_size
            d_loss = discriminator.train_on_batch(X, y)
            discriminator.trainable = False

            # Update random vector
            for i in range(batch_size):
                noise[i, :] = np.random.uniform(-1, 1, 100)


            # Compute the generator's loss on if it could correctly convince the classifier
            g_c_loss = classifier_on_generator.train_on_batch(
                [noise, classes],
                classes
            )

            # Compute the generator's loss on if it could fool the discriminator
            g_d_loss = discriminator_on_generator.train_on_batch(
                [noise, classes],
                [1] * batch_size)

            g_loss = g_d_loss + g_c_loss
            discriminator.trainable = True
            if index % 10 == 0:
                print("Epoch {} batch:{:3} d_loss: {:8.6f} c_loss: {:8.6f} g_loss: {:8.6f}".format(
                    epoch, index, d_loss, c_loss, g_loss)
                )

        if save_epoch_weights:
            generator.save_weights('generator_{}.h5'.format(epoch), True)
            discriminator.save_weights('discriminator_{}.h5'.format(epoch), True)

        generator.save_weights('generator.h5'.format(epoch), True)
        discriminator.save_weights('discriminator.h5'.format(epoch), True)


def generate(BATCH_SIZE, nice=False):
    generator = generator_model()
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    generator.load_weights('generator.h5')
    if nice:

        if K.image_data_format() == 'channels_first':
            input_shape = (1, img_rows, img_cols)
        else:
            input_shape = (img_rows, img_cols, 1)

        discriminator = discriminator_model(input_shape)
        discriminator.compile(loss='binary_crossentropy', optimizer="SGD")
        discriminator.load_weights('discriminator.h5')
        noise = np.zeros((BATCH_SIZE*20, 100))
        for i in range(BATCH_SIZE*20):
            noise[i, :] = np.random.uniform(-1, 1, 100)
        generated_images = generator.predict(noise, verbose=1)
        d_pret = discriminator.predict(generated_images, verbose=1)
        index = np.arange(0, BATCH_SIZE*20)
        index.resize((BATCH_SIZE*20, 1))
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        nice_images = np.zeros((BATCH_SIZE,) + input_shape, dtype=np.float32)
        for i in range(int(BATCH_SIZE)):
            idx = int(pre_with_index[i][1])
            if K.image_data_format() == 'channels_first':
                nice_images[i, :, :] = generated_images[idx, 0, :, :].reshape(input_shape)
            else:
                nice_images[i, :, :] = generated_images[idx, :, :, 0].reshape(input_shape)
        image = combine_images(nice_images)
    else:
        noise = np.zeros((BATCH_SIZE, 100))
        for i in range(BATCH_SIZE):
            noise[i, :] = np.random.uniform(-1, 1, 100)
        generated_images = generator.predict(noise, verbose=1)
        image = combine_images(generated_images)
    image = image*127.5+127.5
    Image.fromarray(image.astype(np.uint8)).save(
        "generated_image.png")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--path", type=str, help="Path to save generated images")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--save-weights", dest="save_epochs", action='store_true',
                        help="Save weights for each epoch")
    parser.add_argument("--nice", dest="nice", action="store_true")
    parser.set_defaults(nice=False, save_epochs=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    if args.mode == "train":
        import os.path
        path = os.path.abspath(args.path)
        train(batch_size=args.batch_size, epochs=args.epochs, save_epoch_weights=args.save_epochs, path=path)
    elif args.mode == "generate":
        generate(BATCH_SIZE=args.batch_size, nice=args.nice)
