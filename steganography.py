"""
Agents:

- *Alice*: a generator which takes a cover image + a message to hide
- *Eve*: A discriminator which is given images and has to decide if they contain a hidden message or not
- *Bob*: Given generated images and reconstructs the secret message

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

img_shape = img_rows, img_cols, img_depth = 28, 28, 1
msg_len = 64


def generator_model(input_shape):
    img = Input(shape=input_shape, name='CoverImage')
    msg = Input(shape=(msg_len,), name='msg_input')

    # Fully connected embedding layer to encode the message
    msg_embedding = Dense(units=msg_len*2, activation='tanh')(msg)

    # Conv2D layers to encode the image
    img_l1 = Conv2D(32,
           kernel_size=3,
           activation='relu',
           input_shape=img_shape)(img)
    img_l2 = Conv2D(32, kernel_size=3, activation='relu')(img_l1)
    img_l3 = Conv2D(32, kernel_size=5, activation='relu')(img_l2)
    img_l4 = MaxPooling2D(pool_size=(2, 2))(img_l3)
    img_l5 = Dropout(0.25)(img_l4)
    img_l6 = Flatten()(img_l5)

    x = keras.layers.concatenate([msg_embedding, img_l6])

    fc1 = Dense(units=1024, activation='tanh')(x)

    # Maybe more layers here...
    fc2 = Dense(128*7*7, activation='relu')(fc1)
    #fc3 = Dense(128*7*7, activation='relu')(fc2)

    x1 = BatchNormalization()(fc2)
    x2 = Activation('tanh')(x1)
    x3 = Reshape((7, 7, 128), input_shape=(128*7*7,))(x2)
    x4 = UpSampling2D(size=(2, 2))(x3)
    x5 = Conv2D(64, kernel_size=5, padding='same', activation='tanh')(x4)
    x6 = UpSampling2D(size=(2, 2))(x5)

    out = Conv2D(1, kernel_size=5, padding='same', activation='tanh')(x6)

    model = Model(inputs=[img, msg], outputs=out)

    return model


def classifier_model(input_shape):
    # Bob: images and reconstructs the secret message
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
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='tanh'))
    return model


def discriminator_model(input_shape):
    # Eve: given images and has to decide if they contain a hidden message or not
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


def gen_bit_data(samples, msg_len=msg_len):
    return (np.random.randint(0, 2, size=(samples, msg_len)) * 2 - 1).astype(float)


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
        input_shape = (img_rows, img_cols, img_depth)

    x_train = (x_train.astype(np.float32) - 127.5)/127.5

    alice = generator_model(input_shape)
    bob = classifier_model(input_shape)
    discriminator = discriminator_model(input_shape)


    # Create the Alice Bob GAN System
    ab_msg_input = Input(shape=(msg_len, ), name='SecretMsg')
    ab_img_input = Input(shape=input_shape, name='OriginalImage')
    ab_inputs = [ab_img_input, ab_msg_input]
    stamped_img = alice(ab_inputs)
    reconstructed_msg_out = bob(inputs=[stamped_img])
    alice_and_bob_gan = Model(ab_inputs, reconstructed_msg_out, name='Alice Bob GAN')

    # Create the Alice Eve GAN System
    ae_msg_input = Input(shape=(msg_len, ), name='SecretMsg')
    ae_img_input = Input(shape=input_shape, name='OriginalImage')
    ae_inputs = [ae_img_input, ae_msg_input]
    stamped_img = alice(ae_inputs)
    contains_secret_msg = discriminator(inputs=[stamped_img])
    alice_and_eve_gan = Model(ae_inputs, contains_secret_msg, name='Alice Eve GAN')


    # Setup the optimizers and compile the models
    g_optim = Adam(lr=0.0001)
    g_c_optim = Adam(lr=0.0001)
    g_d_optim = Adam(lr=0.0001)
    c_optim = Adam()
    d_optim = Adam(lr=0.0001)


    alice.compile(loss='binary_crossentropy', optimizer=g_optim)
    alice_and_eve_gan.compile(loss='binary_crossentropy', optimizer=g_d_optim)
    alice_and_bob_gan.compile(loss='mean_absolute_error', optimizer=g_c_optim)


    discriminator.compile(loss='mean_absolute_error', optimizer=d_optim)

    bob.trainable = True
    bob.compile(loss='mean_absolute_error', optimizer=c_optim)


    print("{} batches per epoch".format(x_train.shape[0] // batch_size))
    for epoch in range(epochs):
        for index in range(int(x_train.shape[0]/batch_size)):

            image_batch = x_train[index * batch_size:(index + 1) * batch_size]
            msg_batch = gen_bit_data(batch_size)

            # sample classes here and get the generator to make fake images
            generated_images = alice.predict([image_batch, msg_batch], verbose=0)

            if index % 50 == 0:
                image = combine_images(generated_images[:25])
                image = image*127.5+127.5
                filename = '{}/{}_{}.png'.format(path, epoch, index)
                Image.fromarray(image.astype(np.uint8)).save(filename)

            # Compute Bob's loss - tasked with reconstruction of the secret.
            # loss on the generated images
            bob.trainable = True
            c_loss = bob.train_on_batch(generated_images, msg_batch)
            bob.trainable = False

            # Compute the discriminator's loss using both stamped and
            # unstamped images:
            X_img = np.concatenate((image_batch, generated_images))
            y = [0] * batch_size + [1] * batch_size
            discriminator.trainable = True
            d_loss = discriminator.train_on_batch(X_img, y)
            discriminator.trainable = False

            # Compute Alice's loss on if it could correctly communicate to Bob
            alice.trainable = True
            g_c_loss = alice_and_bob_gan.train_on_batch(
                [image_batch, msg_batch],
                msg_batch
            )

            # TODO Should Alice be locked here?
            alice.trainable = False
            # Compute Alice's loss on if it could fool Eve
            g_d_loss = alice_and_eve_gan.train_on_batch(
                [image_batch, msg_batch],
                [0] * batch_size
            )

            discriminator.trainable = True
            if index % 10 == 0:
                print("Epoch {} batch:{:3} Eve: {:8.6f} Bob: {:8.6f} AliceBob: {:8.6f} AliceEve: {:8.6f}".format(
                    epoch, index, d_loss, c_loss, g_c_loss, g_d_loss)
                )

        if save_epoch_weights:
            alice.save_weights('generator_{}.h5'.format(epoch), True)
            bob.save_weights('classifier_{}.h5'.format(epoch), True)
            discriminator.save_weights('discriminator_{}.h5'.format(epoch), True)

        alice.save_weights('generator.h5'.format(epoch), True)
        discriminator.save_weights('discriminator.h5'.format(epoch), True)


def generate(batch_size, nice=False):
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
        noise = np.zeros((batch_size * 20, 100))
        desired_digit = np.zeros((batch_size * 20, 10), dtype=int)
        for i in range(batch_size*20):
            noise[i, :] = np.random.uniform(-1, 1, 100)
            desired_digit[i, i % 10] = 1
        generated_images = generator.predict([noise, desired_digit], verbose=1)
        d_pret = discriminator.predict(generated_images, verbose=1)
        index = np.arange(0, batch_size * 20)
        index.resize((batch_size * 20, 1))
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        nice_images = np.zeros((batch_size,) + input_shape, dtype=np.float32)
        for i in range(int(batch_size)):
            idx = int(pre_with_index[i][1])
            if K.image_data_format() == 'channels_first':
                nice_images[i, :, :] = generated_images[idx, 0, :, :].reshape(input_shape)
            else:
                nice_images[i, :, :] = generated_images[idx, :, :, 0].reshape(input_shape)
        image = combine_images(nice_images)
    else:
        noise = np.zeros((batch_size, 100))
        desired_digit = np.zeros((batch_size, 10), dtype=int)
        for i in range(batch_size):
            noise[i, :] = np.random.uniform(-1, 1, 100)
            desired_digit[i, i % 10] = 1
        generated_images = generator.predict([noise, desired_digit], verbose=1)
        image = combine_images(generated_images)
    image = image*127.5+127.5
    Image.fromarray(image.astype(np.uint8)).save(
        "generated_image.png")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--path", type=str, default='.', help="Path to save generated images")
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
        train(batch_size=args.batch_size, epochs=args.epochs, save_epoch_weights=args.save_epochs, path=args.path)
    elif args.mode == "generate":
        generate(batch_size=args.batch_size, nice=args.nice)
