"""

An attempt at implementing
[Learning to Protect Communications with Adversarial Neural Cryptography](https://arxiv.org/abs/1610.06918)
using keras.

Inspiration also from https://nlml.github.io/neural-networks/adversarial-neural-cryptography/

Alice
    Inputs: a plaintext message P, and a random key K
    Outputs: a ciphertext C

Bob
    Inputs: ciphertext C, key K
    Outputs: plaintext Pb

Eve
    Inputs: ciphertext C
    Outputs: plaintext Pe

"""
import pickle
import numpy as np
import time
import keras
from keras.losses import mean_absolute_error
from keras.models import Model, Sequential
from keras.layers import Dense, Input
from keras.layers import Reshape, Conv1D
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Flatten

from keras import backend as K

# parameters
from keras.optimizers import Adam

key_size = 16
msg_size = 16
cipher_size = 16
epochs = 100
batch_size = 1024
learning_rate = 0.0008
extra_eve_training_batches = 3

callback = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=False, write_images=False)


def common_layers(first_fully_connected_layer, name='unknown'):
    """
    Mix and transform model.

    Layers:

    - Fully Connected (key_size + msg_size) -> (key_size + msg_size)
    - 4 x Conv1D layers
    - Last Conv2D layer has output of correct size

    Note input values are -1, +1
    """

    # The Convolution layers described in section 2.5
    x1 = Conv1D(filters=2,
                kernel_size=4,
                strides=1,
                activation='sigmoid',
                padding='same',
                name='{}_conv1'.format(name))(first_fully_connected_layer)
    x2 = Conv1D(filters=4,
                kernel_size=2,
                strides=2,
                activation='sigmoid',
                padding='same',
                name='{}_conv2'.format(name))(x1)
    x3 = Conv1D(filters=4,
                kernel_size=2,
                strides=1,
                activation='sigmoid',
                padding='same',
                name='{}_conv3'.format(name))(x2)
    x4 = Conv1D(filters=1,
                kernel_size=1,
                strides=1,
                activation='tanh',
                padding='same',
                name='{}_conv4'.format(name))(x3)

    return x4


def alice_bob_input_fc(name, msg_name='plaintext'):
    key_vector = Input(shape=(key_size, 1), name='key')
    msg_input = Input(shape=(msg_size, 1), name=msg_name)

    input_size = key_size + msg_size
    inputs = keras.layers.concatenate([key_vector, msg_input],
                                      name='{}_input'.format(name))

    # 2N * 2N fully connected embedding layer
    # allows mixing between key and plaintext
    fc = Dense(units=input_size,
               activation='sigmoid',
               input_shape=(input_size,),
               name='{}_fc'.format(name))(inputs)
    # TODO Is this a bit hacky? Have I got it correct?
    # (batch_size, steps, input_dim)
    fc = Reshape((input_size, -1))(fc)

    return key_vector, msg_input, fc


def eve_input_fc(name='eve'):
    inputs = Input(shape=(msg_size, 1), name='ciphertext')

    N = msg_size

    # N * 2N fully connected embedding layer
    # allows mixing between key and plaintext
    fc = Dense(units=2 * N,
               activation='sigmoid',
               input_shape=(N,),
               name='{}_fc'.format(name))(inputs)

    # TODO Is this a bit hacky? Have I got it correct?
    fc = Reshape((2 * N, -1))(fc)

    return inputs, fc


def gen_data(n=batch_size, msg_len=msg_size, key_len=key_size):
    return (np.random.randint(0, 2, size=(n, msg_len, 1)) * 2 - 1).astype(float), \
           (np.random.randint(0, 2, size=(n, key_len, 1)) * 2 - 1).astype(float)


key_vector, msg_input, fc_layer = alice_bob_input_fc('alice')

alice_layers = common_layers(fc_layer, 'alice')
alice = Model(inputs=[key_vector, msg_input], outputs=alice_layers, name='Alice')

key_vector, ciphertext_input, fc_layer = alice_bob_input_fc('bob', msg_name='ciphertext')
bob_layers = common_layers(fc_layer, 'bob')
bob = Model(inputs=[key_vector, ciphertext_input], outputs=bob_layers, name='Bob')

eve_ciphertext_input, fc = eve_input_fc()
eve_layers = common_layers(fc, 'eve')
eve = Model(inputs=[eve_ciphertext_input], outputs=eve_layers, name='Eve')


def gan_models():
    key_vector = Input(shape=(key_size, 1), name='key')
    msg_input = Input(shape=(msg_size, 1), name='plaintext')
    alice_bob_gan_inputs = [key_vector, msg_input]
    C = alice(alice_bob_gan_inputs)
    p_bob = bob(inputs=[key_vector, C])
    alice_bob_gan = Model(alice_bob_gan_inputs, p_bob, name='AliceBobGAN')
    p_eve = eve(inputs=[C])
    alice_eve_gan = Model(alice_bob_gan_inputs, p_eve, name='AliceEveGAN')
    return alice_bob_gan, alice_eve_gan


def train(batch_size, epochs, path=''):
    """
    Goal is to train "Alice + Bob" for one minibatch then "Eve" for 2 minibatches.
    """

    alice_and_bob_gan, eve_gan = gan_models()

    def alice_eve_loss(y_true, y_pred):
        """
        Alice's second loss component - seeing if it could
        *prevent* Eve from learning the correct message.

        If y_pred is close to y_true, or close to the opposite of y_true
        then the loss should be high.

        The optimal loss is half the bits match.
        """

        return K.abs(K.mean(K.abs(y_pred - y_true), axis=-1) - 1)

    def update_alice_and_bob_gan(last_eve_loss=1):
        def alice_loss(y_true, y_pred):
            return 1 + mean_absolute_error(y_true, y_pred) - last_eve_loss

        alice_and_bob_gan.compile(loss=alice_loss, optimizer=Adam(lr=learning_rate))

    eve.compile(loss='mean_absolute_error', optimizer=Adam(lr=learning_rate))
    eve_gan.compile(loss=alice_eve_loss, optimizer=Adam(lr=learning_rate))
    update_alice_and_bob_gan()

    alice_losses = []
    combined_losses = []
    eve_losses = []

    for epoch in range(epochs):
        print('epoch {}'.format(epoch))
        for index in range(1000):
            x_train_keys, x_train_msgs = gen_data(batch_size)

            # Compute Alice's loss on if it could communicate with Bob
            # While trying not to communicate with eve

            alice.trainable = True
            bob.trainable = True
            alice_and_bob_gan.fit(
                [x_train_keys, x_train_msgs],
                x_train_msgs,
                callbacks=[callback],
                verbose=0, initial_epoch=epoch
            )

            a_loss = alice_and_bob_gan.evaluate([x_train_keys, x_train_msgs], x_train_msgs, verbose=0)

            alice_losses.append((time.time(), a_loss))

            # Not too much point training eve until alice and bob are communicating
            if a_loss < 0.1:
                generated_coms = alice.predict([x_train_keys, x_train_msgs])

                # Train eve by trying to reconstruct the comms
                eve.trainable = True
                e_loss = eve.train_on_batch([generated_coms], x_train_msgs)

                eve.trainable = False

                # # Train bob to try reconstruct the comms
                # # alice.trainable = False
                # # bob.trainable = True
                # # b_loss = bob.train_on_batch([x_train_keys, generated_coms], x_train_msgs)
                # # bob.trainable = False


                # The important adversarial part!
                alice.trainable = True
                eve.trainable = False

                a_loss_against_eve = eve_gan.train_on_batch([x_train_keys, x_train_msgs], x_train_msgs)
                combined_losses.append((time.time(), a_loss_against_eve))

                # # Give Eve an advantage by allowing extra training against the model
                # # that Alice and Bob have at the moment.
                eve.trainable = True
                for extra_eve_training_batch in range(extra_eve_training_batches):
                    x_train_keys, x_train_msgs = gen_data(batch_size)
                    generated_coms = alice.predict([x_train_keys, x_train_msgs])
                    e_loss = eve.train_on_batch([generated_coms], x_train_msgs)
                eve_losses.append((time.time(), e_loss))

                update_alice_and_bob_gan(e_loss)

                if index % 50 == 0:
                    print(a_loss, a_loss_against_eve, e_loss)
            else:
                print(a_loss)

        # x_test_keys, x_test_msgs = gen_data(100)
        # print("AB communication loss: ", alice_and_bob_gan.evaluate([x_test_keys, x_test_msgs], x_test_msgs, verbose=0))
        pickle.dump({
            'alice': alice_losses,
            'eve': eve_losses,
            'both': combined_losses
        }, open('data/out-{}.dat'.format(epoch), 'wb'))


if __name__ == '__main__':
    train(batch_size, epochs)


