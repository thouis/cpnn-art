import keras
import random
import pylab
import seaborn  # noqa
import numpy as np

from keras.models import Model
from keras.layers import Dense, Activation, Input
from keras.layers.merge import Add
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
#       - how to mix Zs? - probably keep them with bottom layer
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
def nrelu(x):
    return - K.relu(-x)

keras.activations.abs = K.abs
keras.activations.nrelu = nrelu
keras.activations.sin = K.sin


def weight_init(shape, dtype=None):
    return K.random_normal(shape, stddev=0.7 / np.sqrt(N), dtype=dtype)


def random_cppn(min_layers=2, max_layers=10):
    num_layers = random.randint(min_layers, max_layers)

    input = x = Input(shape=(N,))
    for n in range(num_layers):
        x2 = Dense(N, kernel_initializer=weight_init, input_dim=N)(x)
        x3 = Activation(random.choice(activations))(x2)
        x = Add()([x, x3])
    x = Activation('sigmoid')(x)  # final activation
    return Model(inputs=[input], outputs=[x]), np.random.uniform(0, 1, N - 4)


def crossover(nz1, nz2):
    net1, z1 = nz1
    net2, z2 = nz2

    # This is very painful, but I think might be the simplest way to achieve this.

    # choose a random dense layer in net1
    split1 = net1.layers.index(np.random.choice([l for l in net1.layers if isinstance(l, Dense)]))
    # choose a random Dense layer in net2
    split2 = net2.layers.index(random.choice([l for l in net2.layers if isinstance(l, Dense)]))

    # temporary list of new layers
    pre_layers = net1.layers[:split1]  # don't include dense activation that was chosen from net1
    post_layers = net2.layers[split2:]  # do include the Dense layer chosen from net2 (to
                                        # guarantee the new net has >= 1, otherwise future
                                        # crosses will fail).
    new_layers = pre_layers + post_layers

    # create a new model, copying weights & activations
    input = x = Input(shape=(N,))

    # copy layers and weights from new layers
    for l in new_layers[1:-1]:  # ignore input & final activation
        if isinstance(l, Dense):
            d = Dense(N, input_dim=N)
            d.build((None, N))
            d.set_weights(l.get_weights())
            x2 = d(x)
        elif isinstance(l, Activation):
            a = Activation(l.activation)
            x3 = a(x2)
        else:
            assert isinstance(l, Add)
            x = Add()([x, x3])
    x = Activation('sigmoid')(x)  # final activation
    return Model(inputs=[input], outputs=[x]), z1


if __name__ == '__main__':
    X, Y = np.meshgrid(np.linspace(-1, 1, image_size),
                       np.linspace(-1, 1, image_size))
    R = np.sqrt(X ** 2 + Y ** 2)

    # angle might need a random offset/shift to be interesting.
    A = np.arctan2(Y, X) * 0.05 / np.pi * 0

    XYRA = np.stack([X, Y, R, A], axis=0)

    for x in range(15):
        model1, z1 = random_cppn()
        model2, z2 = random_cppn()
        modelcross, zc = crossover((model1, z1), (model2, z2))
        for y, (model, z) in enumerate([(model1, z1),
                                        (model2, z2),
                                        (modelcross, zc)]):

            Z = np.ones((N - 4, image_size, image_size)) * z[..., np.newaxis, np.newaxis]
            input = np.concatenate((XYRA, Z), axis=0)
            # reshape to make each row a batch
            input = input.reshape((N, -1)).T

            output = model.predict_on_batch(input)[:, 2]  # R channel

            # reshape back to image_size
            output = output.reshape((image_size, image_size))

            pylab.subplot(3, 15, y * 15 + x + 1)
            pylab.imshow(output)
            pylab.gca().grid(False)
            pylab.axis('off')
    pylab.show()
