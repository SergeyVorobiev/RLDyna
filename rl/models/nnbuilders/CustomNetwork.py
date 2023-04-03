from keras import Input, Model
from keras.api._v2.keras import initializers
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K


class CustomNetwork:

    # kernel_init - by default GlorotUniform (Xavier)
    # opt - by default Adam
    # size - [10, 10]
    # act - ['relu, 'relu']
    # loss - None by default
    @staticmethod
    def build_quadratic(input_shape, output_n, alpha, kernel_init=None, bias_init=None, loss=None, optimizer=None,
                        act=None, out='softmax', size=None, custom_model=None):
        if act is None:
            act = ['relu', 'relu']
        if size is None:
            size = [10, 10]
        if kernel_init is None:
            kernel_init = initializers.GlorotUniform
        inputs = Input(shape=input_shape)
        x = Dense(size[0], activation=act[0], kernel_initializer=kernel_init, bias_initializer=bias_init)(inputs)
        x = Dense(size[1], activation=act[1], kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
        x = Dense(output_n, activation=out, kernel_initializer=kernel_init)(x)
        if custom_model is None:
            model = Model(inputs, x)
        else:
            model = custom_model(inputs, x)
        if optimizer is None:
            optimizer = Adam(learning_rate=alpha)
        model.compile(loss=loss, optimizer=optimizer)
        return model

    # kernel_init - by default GlorotUniform (Xavier)
    # opt - by default Adam
    # size - 10
    # act - 'relu'
    # loss - None by default
    @staticmethod
    def build_linear(input_shape, output_n, alpha, kernel_init=None, bias_init=None, loss=None, optimizer=None,
                     act=None, out='softmax', size=None, custom_model=None):
        if act is None:
            act = 'relu'
        if size is None:
            size = 10
        if kernel_init is None:
            kernel_init = initializers.GlorotUniform
        inputs = Input(shape=input_shape)
        x = Dense(size, activation=act, kernel_initializer=kernel_init, bias_initializer=bias_init)(inputs)
        x = Dense(output_n, activation=out, kernel_initializer=kernel_init)(x)
        if custom_model is None:
            model = Model(inputs, x)
        else:
            model = custom_model(inputs, x)
        if optimizer is None:
            optimizer = Adam(learning_rate=alpha)
        model.compile(loss=loss, optimizer=optimizer)
        return model

    @staticmethod
    def tanh2_activation(x):
        return K.tanh(x) * 2.0

