import keras
import random
import pylab
import seaborn  # noqa
import numpy as np

from keras.models import Model
from keras.layers import Dense, Activation, Input, add
import keras.activations
import keras.backend as K

# Residual CPPNs
# inspired by http://blog.otoro.net/2016/03/25/generating-abstract-patterns-with-tensorflow/
# see also https://pair-code.github.io/deeplearnjs/demos/nn-art/nn-art-demo.html
# residual networks should make nets more robust to mutations: https://arxiv.org/abs/1603.05027

# input is X, Y, R, angle (scaled by 0.1 / pi), Z_1 ... Z_(N-4)
#   (currently, angle is zero because it leads to a branch cut in the same place)
#   Zs are traininable parameters that make the first layer NxN dense just like all the others.
#
# each layer is a dense network with N inputs, N outputs.
# each layer activation can be tanh, sigmoid, relu, abs, identity, negative relu.
# final activation is sigmoid of third (R) channel - (just to choose one)

# operations to create new nets via mutation:
#    random: create a new network from scratch
#    crossover: randomly select two candidates, chop each at some midpoint, stitch together opposite halves.
#       - how to mix Zs?
#    mutate_weights: choose a random layer, choose a small fraction of weights, increase/decrease randomly
#    mutate_activation: choose a random layer, randomly switch its activation
#    mutate_Z: small fraction of initial Zs are increased/decreased randomly
#      - or perhaps these should all just be zero?  Or the same values for all nets?
#    duplicate: choose a random subsequence of layers, repeat them
#    harmonic?  not sure best way to implement that.  Maybe insert a scaling dense layer at the bottom.
#    ablation: remove a random layer

# Ideas for making Angle useful as an input:
#   Intermediate rotation (or transformation) layers that interpolate values from previous layer?
#   Random offset at first layer (one of the Zs?)

# number of input channels
N = 10

activations = ["tanh", "sigmoid", "relu", "abs", "linear", "nrelu", "sin"]  # others: exp/nexp?
image_size = 128

# create the missing activations
keras.activations.abs = K.abs
keras.activations.nrelu = lambda x: - K.relu(-x)
keras.activations.sin = K.sin


def weight_init(shape, dtype=None):
    return K.random_normal(shape, stddev=1 / np.sqrt(N), dtype=dtype)


def random_cppn(min_layers=2, max_layers=10):
    num_layers = random.randint(min_layers, max_layers)

    input = x = Input(shape=(N,))
    for n in range(num_layers):
        x2 = Dense(N, kernel_initializer=weight_init, input_dim=N)(x)
        x3 = Activation(random.choice(activations))(x2)
        x = add([x, x3])
    x = Activation('sigmoid')(x)  # final activation
    return Model(inputs=[input], outputs=[x]), np.random.uniform(0, 1, N - 4)


if __name__ == '__main__':
    X, Y = np.meshgrid(np.linspace(-1, 1, image_size),
                       np.linspace(-1, 1, image_size))
    R = np.sqrt(X ** 2 + Y ** 2)

    # Angle seems uninteresting, currenrl
    # angle might need a random offset/shift to be interesting.
    A = np.arctan2(Y, X) * 0.05 / np.pi * 0

    XYRA = np.stack([X, Y, R, A], axis=0)

    for x in range(5):
        for y in range(5):
            model, z = random_cppn()
            Z = np.ones((N - 4, image_size, image_size)) * z[..., np.newaxis, np.newaxis]
            input = np.concatenate((XYRA, Z), axis=0)
            # reshape to make each row a batch
            input = input.reshape((N, -1)).T

            output = model.predict_on_batch(input)[:, 2]  # R channel

            # reshape back to image_size
            output = output.reshape((image_size, image_size))

            pylab.subplot(5, 5, x * 5 + y + 1)
            pylab.imshow(output)
            pylab.gca().grid(False)
            pylab.axis('off')
    pylab.show()
