"""
Agents:

- *Alice*: a generator which takes a cover image + a message to hide
- *Eve*: A discriminator which is given images and has to decide if they contain a hidden message or not
- *Bob*: Given generated images and reconstructs the secret message

"""
import os
import os.path
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


def generator_model(input_shape, msg_len):
    img = Input(shape=input_shape, name='CoverImage')
    msg = Input(shape=(msg_len,), name='msg_input')

    # Fully connected embedding layer to encode the message
    msg_embedding = Dense(units=msg_len, activation='tanh')(msg)
    msg_2 = Dense(units=img_rows*img_cols, activation='tanh')(msg_embedding)
    msg_3 = Reshape((img_rows, img_cols, 1), input_shape=(img_rows*img_cols,))(msg_2)


    # Conv2D layers to pass the image through
    img_l1 = Conv2D(img_depth*5,
                    kernel_size=1,
                    padding='same',
                    activation='tanh',
                    input_shape=img_shape)(img)

    img_l2 = Conv2D(3,
           kernel_size=1,
           padding='same',
           #activation='tanh',# Seems to be ok to leave off
           input_shape=img_shape)(img_l1)

    # Mix the image and message

    x = keras.layers.concatenate([msg_3, img_l2])
    img_l3 = Conv2D(3,
           kernel_size=1,
           padding='same',
           #activation='tanh',# Leave off
           input_shape=img_shape)(x)

    out = Conv2D(1, kernel_size=1, padding='same', activation='tanh')(img_l3)

    model = Model(inputs=[img, msg], outputs=out)

    return model


def classifier_model(input_shape, msg_len):
    # Bob: Given images reconstructs the secret message
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
    model.add(Dense(msg_len, activation='tanh'))
    return model


def discriminator_model(input_shape):
    # Eve: given images and has to decide if they contain a hidden message or not
    # Optionally could consider ACGAN - output a probability of the class
    # Could use something designed for this https://github.com/daniellerch/stego_cnn/blob/master/stego_cnn_xu.py#L98
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


def gen_bit_data(samples, msg_len):
    return (np.random.randint(0, 2, size=(samples, msg_len)) * 2 - 1).astype(float)


def train(msg_len, batch_size, epochs, save_epoch_weights, path=''):
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

    alice = generator_model(input_shape, msg_len)
    bob = classifier_model(input_shape, msg_len)
    discriminator = discriminator_model(input_shape)


    # Create the Alice Bob GAN System
    ab_msg_input = Input(shape=(msg_len, ), name='SecretMsg')
    ab_img_input = Input(shape=input_shape, name='OriginalImage')
    ab_inputs = [ab_img_input, ab_msg_input]
    stamped_img = alice(ab_inputs)
    reconstructed_msg_out = bob(inputs=[stamped_img])
    alice_and_bob_gan = Model(ab_inputs, reconstructed_msg_out, name='Alice Bob GAN')

    # Create the Alice Discriminator GAN System
    ae_msg_input = Input(shape=(msg_len, ), name='SecretMsg')
    ae_img_input = Input(shape=input_shape, name='OriginalImage')
    ae_inputs = [ae_img_input, ae_msg_input]
    stamped_img = alice(ae_inputs)
    contains_secret_msg = discriminator(inputs=[stamped_img])
    alice_and_eve_gan = Model(ae_inputs, contains_secret_msg, name='Alice Eve GAN')


    # Setup the optimizers and compile the models
    g_c_optim = Adam(lr=0.0001)
    g_d_optim = Adam(lr=0.0005)
    c_optim = Adam()
    d_optim = Adam()

    alice_and_eve_gan.compile(loss='binary_crossentropy', optimizer=g_d_optim)
    alice_and_bob_gan.compile(loss='mean_absolute_error', optimizer=g_c_optim)

    discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)

    alice.trainable = True
    bob.trainable = True
    bob.compile(loss='mean_absolute_error', optimizer=c_optim)
    c_loss, g_c_loss = 9, 9
    print("{} batches per epoch".format(x_train.shape[0] // batch_size))
    for epoch in range(epochs):
        num_batches = int(x_train.shape[0]/batch_size)

        for batch in range(num_batches):

            image_batch = x_train[batch * batch_size:(batch + 1) * batch_size]
            msg_batch = gen_bit_data(batch_size, msg_len)

            for k in range(1):
                # Get Alice to generate stamped images
                generated_images = alice.predict([image_batch, msg_batch], verbose=0)

                # Compute the discriminator's loss using both stamped and
                # unstamped images:
                X_img = np.concatenate((image_batch, generated_images))
                y = [0] * batch_size + [1] * batch_size
                discriminator.trainable = True
                d_loss = discriminator.train_on_batch(X_img, y)
                discriminator.trainable = False


            # This helps give Alice an advantage to learn to correctly generate images
            # before trying to hide information in them.
            if epoch > 0:
                # Compute Bob's loss - tasked with reconstruction of the secret.
                # loss on the generated images
                bob.trainable = True
                c_loss = bob.train_on_batch(generated_images, msg_batch)

                # Compute Alice's loss on if it could correctly communicate to Bob
                g_c_loss = alice_and_bob_gan.train_on_batch(
                    [image_batch, msg_batch],
                    msg_batch
                )

                bob.trainable = False

            if batch % 50 == 0:
                image = combine_images(np.concatenate((generated_images[:2], image_batch[:2])))
                image = image*127.5+127.5
                filename = '{}/{}bits_e{}-s{}.png'.format(path, msg_len, epoch, batch)
                Image.fromarray(image.astype(np.uint8)).save(filename)


            # Compute Alice's loss on if it could fool the discriminator
            msg_batch = gen_bit_data(batch_size, msg_len)
            discriminator.trainable = False
            g_d_loss = alice_and_eve_gan.train_on_batch(
                [image_batch, msg_batch],
                [0] * batch_size
            )

            discriminator.trainable = True
            if batch % 10 == 0:
                print("{} Step {}.{:3} Discriminator: {:8.6f} Bob: {:8.6f} Alice2Bob: {:8.6f} Alice fooling Discriminator: {:8.6f}".format(
                    msg_len, epoch, batch, d_loss, c_loss, g_c_loss, g_d_loss)
                )

        if save_epoch_weights:
            alice.save_weights('generator_{}.h5'.format(epoch), True)
            bob.save_weights('classifier_{}.h5'.format(epoch), True)
            discriminator.save_weights('discriminator_{}.h5'.format(epoch), True)

        alice.save_weights('generator_{}_bits.h5'.format(msg_len), True)
        discriminator.save_weights('discriminator_{}_bits.h5'.format(msg_len), True)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ex-name", type=str, help='experiment-name')
    parser.add_argument("--path", type=str, default='.', help="Path to save results")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--save-weights", dest="save_epochs", action='store_true',
                        help="Save weights for each epoch")
    parser.set_defaults(nice=False, save_epochs=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    base_path = os.path.abspath(args.path)

    for bits in [4, 8, 16, 32, 64, 80, 100]:
        path = os.path.join(base_path, 'ex3-{}-bit-results'.format(bits))
        if not os.path.exists(path): os.mkdir(path)

        train(msg_len=bits, batch_size=args.batch_size, epochs=args.epochs, save_epoch_weights=args.save_epochs, path=path)
