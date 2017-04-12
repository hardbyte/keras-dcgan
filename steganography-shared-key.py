"""
Agents:

- *Alice*: a generator which takes a cover image + a message to hide + key
- *Eve*: A discriminator which is given images and has to decide if they contain a hidden message or not
- *Bob*: Given generated images + key and reconstructs the secret message

"""
import os
import os.path
import keras
import time

import sklearn
from keras.losses import mean_absolute_error, mean_squared_error
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Dropout, Lambda, AveragePooling2D, GlobalAveragePooling2D, LSTM
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

from matplotlib import pylab

from tbcallback import TensorBoardGAN
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

tensorboard_callback = TensorBoardGAN(
    log_dir='/tmp/ganlogs',
    histogram_freq=0,
    write_graph=False,
    write_images=False
)

img_shape = img_rows, img_cols, img_depth = 28, 28, 1
number_classes = 10

np.random.seed(42)  # for reproducibility


def generator_model(input_shape, msg_len, key_len):
    img = Input(shape=input_shape, name='CoverImage')
    msg = Input(shape=(msg_len,), name='msg_input')
    key = Input(shape=(key_len,), name='key_input')

    # Fully connected embedding layer to encode the message
    msg_embedding = Dense(units=msg_len, activation='tanh')(msg)
    msg_2 = Dense(units=img_rows*img_cols, activation='tanh')(msg_embedding)
    msg_3 = Reshape((img_rows, img_cols, 1), input_shape=(img_rows*img_cols,))(msg_2)

    # Fully connected embedding layer to encode the key
    key_embedding = Dense(units=key_len, activation='tanh')(key)
    key_2 = Dense(units=img_rows*img_cols, activation='tanh')(key_embedding)
    key_3 = Reshape((img_rows, img_cols, 1), input_shape=(img_rows*img_cols,))(key_2)

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

    x = keras.layers.concatenate([key_3, msg_3, img_l2])
    mix_l1 = Conv2D(256,  # TODO ?
           kernel_size=1,
           padding='same',
           #activation='tanh',# Leave off
           input_shape=img_shape)(x)

    # Seems like there could do with some layers here... ?

    out = Conv2D(1, kernel_size=1, padding='same', activation='tanh')(mix_l1)

    model = Model(inputs=[img, msg, key], outputs=out)

    return model


def classifier_model(input_shape, msg_len, key_len):
    # Bob: Given images and a key reconstructs the secret message
    # And outputs what the image was supposed to be

    img = Input(shape=input_shape, name='StampedImage')
    img_embedding = Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=input_shape)(img)
    key = Input(shape=(key_len,), name='key_input')

    # Fully connected embedding layer to encode the key
    key_embedding = Dense(units=key_len, activation='tanh')(key)
    key_2 = Dense(units=img_rows * img_cols, activation='tanh')(key_embedding)
    key_3 = Reshape((img_rows, img_cols, 1), input_shape=(img_rows * img_cols,))(key_2)

    # Merge inputs layer
    x = keras.layers.concatenate([img_embedding, key_3])

    # Todo filters usually get smaller not larger...?
    x = Conv2D(32, kernel_size=3, activation='relu')(x)
    x = Conv2D(32, kernel_size=5, activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.25)(x)

    x_mesg_layer = Dense(128, activation='relu')(x)
    x_class_layer = Dense(128, activation='relu')(x)

    merged_layer = keras.layers.concatenate([x_mesg_layer, x_class_layer])

    x_mesg_out = Dense(msg_len, activation='tanh')(merged_layer)
    #x_class_out = Dense(number_classes, activation='softmax')(merged_layer)

    model = Model(inputs=[img, key], outputs=[x_mesg_out])
    return model


def discriminator_model(input_shape):
    # Eve: given images and has to decide if they contain a hidden message or not
    # Optionally could consider ACGAN - output a probability of the class
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, padding='same', input_shape=input_shape, activation='tanh'))
    model.add(Conv2D(64, (3, 3), activation='tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, activation='tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='tanh'))

    # Output
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model


def output_sample_images(alice, validation_data, filename):
    batch_size = 10
    x, y = validation_data
    img_batch = np.array([
        x[y == i][0] for i in range(10)
    ], dtype=x.dtype)

    msg_batch = gen_bit_data(batch_size, alice.inputs[1].shape[-1].value)
    key_batch = gen_bit_data(batch_size, alice.inputs[2].shape[-1].value)
    generated_images = alice.predict([img_batch, msg_batch, key_batch])

    all_images = np.concatenate((
        img_batch * 127.5 + 127.5,
        generated_images * 127.5 + 127.5,
        ((img_batch - generated_images) * 127.5 + 127.5)/2,
        ))

    num = all_images.shape[0]
    width = 10
    height = int(math.ceil(float(num)/width))
    shape = all_images.shape[1:]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=np.uint8)
    for index, img in enumerate(all_images):
        i = int(index/width)
        j = index % width
        if K.image_data_format() == 'channels_first':
            image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
                img[0, :, :]
        else:
            image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
                img[:, :, 0]

    Image.fromarray(image).save(filename)


def gen_bit_data(samples, msg_len):
    return (np.random.randint(0, 2, size=(samples, msg_len)) * 2 - 1).astype(float)


def create_model(input_shape, msg_len, batch_size, save_epoch_weights, path='',
                 alice_weight_eve=1.0, alice_weight_bob=1.0, alice_cover_diff_weight=1.0):

    # The key could be a mask
    key_len = img_rows * img_cols * img_depth

    alice = generator_model(input_shape, msg_len, key_len)
    bob = classifier_model(input_shape, msg_len, key_len)
    discriminator = discriminator_model(input_shape)

    # Create the Alice Bob GAN System
    ab_msg_input = Input(shape=(msg_len, ), name='SecretMsg')
    ab_img_input = Input(shape=input_shape, name='OriginalImage')
    ab_key_input = Input(shape=(key_len,), name='SharedKey')

    ab_inputs = [ab_img_input, ab_msg_input, ab_key_input]
    stamped_img = alice(ab_inputs)
    reconstructed_msg_out = bob(inputs=[stamped_img, ab_key_input])
    alice_and_bob_gan = Model(ab_inputs, [reconstructed_msg_out], name='Alice Bob GAN')

    # Create the Alice Discriminator GAN System
    ae_msg_input = Input(shape=(msg_len, ), name='SecretMsg')
    ae_img_input = Input(shape=input_shape, name='OriginalImage')
    ae_key_input = Input(shape=(key_len,), name='Key')
    ae_inputs = [ae_img_input, ae_msg_input, ae_key_input]
    stamped_img = alice(ae_inputs)
    contains_secret_msg = discriminator(inputs=[stamped_img])
    alice_and_eve_gan = Model(ae_inputs, contains_secret_msg, name='Alice Eve GAN')

    # Setup the optimizers and compile the models
    g_c_optim = Adam(lr=0.0010)
    g_d_optim = Adam(lr=0.0002)
    g_optim =   Adam(lr=0.0005)
    c_optim =   Adam(lr=0.0005)
    d_optim =   Adam(lr=0.0001)

    def alice_loss(y_true, y_pred):
        return alice_cover_diff_weight * mean_absolute_error(y_true, y_pred)

    def alice_eve_loss(y_true, y_pred):
        """
        A loss function where the objective is to actually guess half 
        the predictions correctly. If a bit is wrong the absolute error 
        will be 2, thus we aim for a mean abs error of 1
        """
        error = mean_absolute_error(y_pred, y_true)
        return alice_weight_eve * K.abs(1 - error)

    def alice_bob_loss(y_true, y_pred):
        # weighted distance of bob's recovery of the message
        return alice_weight_bob * mean_absolute_error(y_true, y_pred)


    alice_and_eve_gan.compile(loss=alice_eve_loss, optimizer=g_d_optim)
    alice_and_bob_gan.compile(loss=alice_bob_loss, optimizer=g_c_optim)

    discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)

    alice.trainable = True
    alice.compile(loss=mean_absolute_error, optimizer=g_optim)

    bob.trainable = True
    bob.compile(loss=mean_absolute_error, optimizer=c_optim)

    # Set callbacks
    tensorboard_callback.set_model(alice_and_bob_gan)
    #alice_eve_tb.set_model(alice_and_eve_gan)

    class GanModel:

        metrics_names = ['acc']

        def loss(self, *args, **kwargs):
            print('lost?')

        def fit(self, x=None,
                y=None,
                batch_size=32,
                epochs=1,
                verbose=1,
                callbacks=None,
                validation_split=0.,
                validation_data=None,
                shuffle=True,
                class_weight=None,
                sample_weight=None,
                initial_epoch=0):
            epoch_acc = 0
            print("{} batches per epoch".format(x_train.shape[0] // batch_size))
            for epoch in range(epochs):
                num_batches = int(x.shape[0]/batch_size)

                for batch in range(num_batches):

                    image_batch = x[batch * batch_size:(batch + 1) * batch_size]

                    msg_batch = gen_bit_data(batch_size, msg_len)
                    key_batch = gen_bit_data(batch_size, key_len)

                    # Get Alice to generate stamped images
                    generated_images = alice.predict([image_batch, msg_batch, key_batch],
                                                     verbose=0)

                    # Compute the discriminator's loss using both stamped and
                    # unstamped images:
                    X_img = np.concatenate((image_batch, generated_images))
                    y = [0] * batch_size + [1] * batch_size
                    discriminator.trainable = True
                    d_loss = discriminator.train_on_batch(X_img, y)
                    discriminator.trainable = False

                    # This may (?) help give Alice an advantage to learn to correctly generate images
                    # before trying to hide information in them.
                    if True:    #epoch > 0:
                        # Compute Bob's loss - tasked with reconstruction of the secret.
                        # loss on the generated images
                        bob.trainable = True
                        # c_loss = bob.train_on_batch(
                        #     generated_images,
                        #     [msg_batch]
                        # )
                        # bob.trainable = False

                        # Compute Alice's loss on if it could correctly communicate to Bob
                        g_c_loss = alice_and_bob_gan.train_on_batch(
                            [image_batch, msg_batch, key_batch],
                            [msg_batch]
                        )

                        bob.trainable = False

                        c_loss = bob.evaluate([generated_images, key_batch], [msg_batch], verbose=0)

                    # Compute Alice's loss on how similar the image is to the cover
                    a_diff_loss = alice.train_on_batch(
                            [image_batch, msg_batch, key_batch],
                            [image_batch]
                        )

                    # Compute Alice's loss on if it could fool the discriminator
                    msg_batch = gen_bit_data(batch_size, msg_len)
                    discriminator.trainable = False
                    g_d_loss = alice_and_eve_gan.train_on_batch(
                        [image_batch, msg_batch, key_batch],
                        [0] * batch_size
                    )

                    discriminator.trainable = True
                    if batch % 500 == 0:
                        self.evaluate(validation_data[0][:min(100, len(validation_data[0]))])
                        with open('{}/logs.txt'.format(path), 'at') as f:
                            print("{} Step {}.{:3} Discriminator: {:8.6f} Bob: {:8.6f} Alice2Bob: {:8.6f} Alice fooling Discriminator: {:8.6f} Alice Cover Diff: {:8.6f}".format(
                                msg_len, epoch, batch, d_loss, c_loss, g_c_loss, g_d_loss, a_diff_loss),
                                file=f
                            )

                        tensorboard_callback.on_epoch_end(batch + epoch * num_batches, {
                            'gc-loss': g_c_loss,
                            'gd-loss': g_d_loss,
                            'd-loss': d_loss,
                            'c-loss': c_loss,
                            'gen vs cover loss': a_diff_loss,
                            'accuracy': epoch_acc

                        })

                #self.loss = [d_loss, c_loss, g_c_loss, g_d_loss, a_diff_loss]
                if save_epoch_weights:
                    alice.save_weights('{}/generator_{}.h5'.format(path, epoch), True)
                    bob.save_weights('{}/classifier_{}.h5'.format(path, epoch), True)
                    discriminator.save_weights('{}/discriminator_{}.h5'.format(path, epoch), True)

                alice.save_weights('{}/generator_{}_bits.h5'.format(path, msg_len), True)
                bob.save_weights('{}/classifier_{}_bits.h5'.format(path, msg_len), True)
                discriminator.save_weights('{}/discriminator_{}_bits.h5'.format(path, msg_len), True)
                print("Epoch {} -".format(epoch), end=' ')
                epoch_acc = self.evaluate(validation_data[0][:min(1000, len(validation_data[0]))])

                filename = '{}/epoch-{}-transfer-accuracy-{}.png'.format(path, epoch, int(epoch_acc))
                output_sample_images(alice, validation_data, filename)

        def evaluate(self, x, y=None, batch_size=32, verbose=0):
            image_batch = x
            msg_batch = gen_bit_data(len(x), msg_len)
            key_batch = gen_bit_data(len(x), key_len)
            generated_images = alice.predict([image_batch, msg_batch, key_batch], verbose=0)

            X_img = np.concatenate((image_batch, generated_images))
            y = [0] * len(image_batch) + [1] * len(generated_images)
            d_loss = discriminator.evaluate(X_img, y, verbose=verbose)

            if verbose > 0:
                print("Discriminator loss: {}".format(d_loss))

            received_messages = alice_and_bob_gan.predict(
                [image_batch, msg_batch, key_batch],
                verbose=verbose
            )
            msgs_transfered = 0
            for msg_received, msg_sent in zip(received_messages, msg_batch):
                if all(np.sign(msg_received) == np.sign(msg_sent)):
                    msgs_transfered += 1

            print("Correctly transfered {} messages out of {}".format(msgs_transfered, len(msg_batch)))
            return 100.0 * msgs_transfered / len(msg_batch)

    return GanModel()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ex-name", type=str, help='experiment-name')
    parser.add_argument("--path", type=str, default='.', help="Path to save results")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--save-weights", dest="save_epochs", action='store_true',
                        help="Save weights for each epoch")
    parser.set_defaults(nice=False, save_epochs=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    base_path = os.path.abspath(args.path)

    bits = 10

    path = os.path.join(base_path, 'steg-key-ex5-{}-bit-results'.format(bits))
    if not os.path.exists(path): os.mkdir(path)

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
    x_test = (x_test.astype(np.float32) - 127.5)/127.5

    def make_model(alice_weight_bob, alice_weight_eve, alice_cover_diff_weight):
        print('New model:', alice_weight_bob)
        model = create_model(
            input_shape,
            msg_len=bits,
            batch_size=args.batch_size,
            save_epoch_weights=args.save_epochs,
            path=path,
            alice_weight_bob=alice_weight_bob,
            alice_weight_eve=alice_weight_eve,
            alice_cover_diff_weight=alice_cover_diff_weight
        )
        return model

    model = make_model(1.0, 0.5, 0.5)
    start = time.time()
    model.fit(x_train, y=y_train, epochs=args.epochs, verbose=0, validation_data=(x_test, y_test))
    print(time.time() - start)

    print(model.evaluate(x_test, verbose=0))

    # classifier = KerasClassifier(make_model, batch_size=32)
    #
    # validator = GridSearchCV(
    #     classifier,
    #     param_grid={
    #         #'alice_weight_eve': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,],
    #         'alice_weight_bob': [ 1.0,],
    #         #'alice_cover_diff_weight': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    #     },
    #     #scoring='',
    #     n_jobs=1)
    #
    # # validator = make_model(1.0)
    #
    # start = time.time()
    # validator.fit(x_train, y=y_train)
    # print(time.time() - start)
    #
    # print('The parameters of the best model are: ')
    # print(validator.best_params_)
    # # validator.best_estimator_.model returns the (unwrapped) keras model
    # best_model = validator.best_estimator_.model
    # print(best_model.evaluate(x_test))
