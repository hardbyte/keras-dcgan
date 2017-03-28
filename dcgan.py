"""
Based on https://github.com/jacobgil/keras-dcgan

"""

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD
from keras.datasets import mnist
from keras import backend as K
import numpy as np
from PIL import Image
import argparse
import math

# input image dimensions
img_rows, img_cols = 28, 28


def generator_model():
    model = Sequential()
    model.add(Dense(units=1024, input_dim=100))
    model.add(Activation('tanh'))
    model.add(Dense(128*7*7))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((7, 7, 128), input_shape=(128*7*7,)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(64, kernel_size=5, padding='same', activation='tanh'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(1, kernel_size=5, padding='same', activation='tanh'))
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
    generator = generator_model()
    discriminator_on_generator = \
        generator_containing_discriminator(generator, discriminator)
    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    discriminator_on_generator.compile(
        loss='binary_crossentropy', optimizer=g_optim)
    discriminator.trainable = True
    discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)
    noise = np.zeros((batch_size, 100))
    print("{} batches per epoch".format(x_train.shape[0] // batch_size))

    for epoch in range(epochs):
        for index in range(int(x_train.shape[0]/batch_size)):
            for i in range(batch_size):
                noise[i, :] = np.random.uniform(-1, 1, 100)
            image_batch = x_train[index * batch_size:(index + 1) * batch_size]
            generated_images = generator.predict(noise, verbose=0)
            if index % 50 == 0:
                image = combine_images(generated_images[:25])
                image = image*127.5+127.5
                filename = '{}/{}_{}.png'.format(path, epoch, index)
                Image.fromarray(image.astype(np.uint8)).save(filename)

            # Compute the discriminator's loss using the generated images:
            X = np.concatenate((image_batch, generated_images))
            y = [1] * batch_size + [0] * batch_size
            d_loss = discriminator.train_on_batch(X, y)
            for i in range(batch_size):
                noise[i, :] = np.random.uniform(-1, 1, 100)
            discriminator.trainable = False

            # Compute the generator's loss on if it could fool the discriminator
            g_loss = discriminator_on_generator.train_on_batch(
                noise, [1] * batch_size)
            discriminator.trainable = True
            if index % 10 == 0:
                print("Epoch {} batch:{:3} d_loss: {:8.6f} g_loss: {:8.6f}".format(epoch, index, d_loss, g_loss))

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
