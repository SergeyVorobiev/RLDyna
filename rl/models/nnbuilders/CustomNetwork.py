from keras import Input, Model, regularizers
from keras.api._v2.keras import initializers
from keras.layers import Dense, BatchNormalization, Dropout, concatenate
from keras.optimizers import Adam
import tensorflow as tf


class CustomNetwork:

    @staticmethod
    def u_critic(input_shape, alpha, out, act, size, kernel_init=None,
                 bias_init=None,
                 loss=None,
                 optimizer=None, custom_model_build_func=None, run_eagerly=False, dropout=0,
                 batch_normalization=False, l1=0.001, l2=0.001):
        return CustomNetwork.two_hidden(1, input_shape, alpha, out, act, size, kernel_init, bias_init, loss, optimizer,
                                        custom_model_build_func,
                                        run_eagerly, dropout, batch_normalization, l1, l2)

    @staticmethod
    def two_hidden(output_n, input_shape, alpha, out, act, size, kernel_init=None,
                   bias_init=None,
                   loss=None,
                   optimizer=None, custom_model_build_func=None, run_eagerly=False, dropout=0,
                   batch_normalization=False, l1=0.001, l2=0.001):
        if kernel_init is None:
            kernel_init = initializers.GlorotUniform
        inputs = Input(shape=input_shape, name="input")
        if batch_normalization:
            x = BatchNormalization()(inputs, name="norm")
        else:
            x = inputs
        x = Dense(size, activation=act[0], kernel_initializer=kernel_init, bias_initializer=bias_init,
                  name="hidden1",
                  kernel_regularizer=regularizers.L1L2(l1, l2))(x)
        if dropout > 0:
            x = Dropout(dropout, name="drop")(x)
        x = Dense(size, activation=act[1], kernel_initializer=kernel_init, bias_initializer=bias_init,
                  name="hidden2",
                  kernel_regularizer=regularizers.L1L2(l1, l2))(x)
        out = Dense(output_n, activation=out, kernel_initializer=kernel_init, name="output")(x)

        if custom_model_build_func is None:
            model = Model(inputs, out)
        else:
            model = custom_model_build_func(inputs, out)
        if optimizer is None:
            optimizer = Adam(learning_rate=alpha)
        model.compile(loss=loss, optimizer=optimizer, run_eagerly=run_eagerly)
        return model

    @staticmethod
    def discrete_separate(input_shape, output_n, alpha, out, act_common, act_action, size, kernel_init=None,
                          bias_init=None,
                          loss=None,
                          optimizer=None, custom_model_build_func=None, run_eagerly=False, dropout=0,
                          batch_normalization=False, l1=0.001, l2=0.001):
        if kernel_init is None:
            kernel_init = initializers.GlorotUniform
        inputs = Input(shape=input_shape, name="input")
        if batch_normalization:
            x = BatchNormalization()(inputs, name="norm")
        else:
            x = inputs
        x = Dense(size, activation=act_common, kernel_initializer=kernel_init, bias_initializer=bias_init,
                  name="hidden1")(x)
        if dropout > 0:
            x = Dropout(dropout, name="drop")(x)

        action_denses = []
        a_size = int(size / output_n)
        for i in range(output_n):
            name = "hidden2" + str(i)
            action_denses.append(Dense(a_size, activation=act_action, kernel_initializer=kernel_init,
                                       bias_initializer=bias_init,
                                       kernel_regularizer=regularizers.L1L2(l1, l2), name=name)(x))
        outputs = []
        i = 0
        for dense in action_denses:
            name = "output" + str(i)
            i += 1
            outputs.append(Dense(1, activation=out, kernel_initializer=kernel_init, name=name)(dense))

        if custom_model_build_func is None:
            model = Model(inputs, concatenate(outputs))
        else:
            model = custom_model_build_func(inputs, concatenate(outputs))
        if optimizer is None:
            optimizer = Adam(learning_rate=alpha)
        model.compile(loss=loss, optimizer=optimizer, run_eagerly=run_eagerly)
        return model

    @staticmethod
    def build_distribution(input_shape, output_n, act_mean, act_deviation, alpha, loss, size, dropout=0.0,
                           kernel_init=None, bias_init=None, l1=0.0, l2=0.0, batch_normalization=False,
                           optimizer=None, custom_model_build_func=None, run_eagerly=False):
        i = Input(shape=input_shape)
        if batch_normalization:
            i = BatchNormalization()(i)
        means = []
        deviations = []
        for k in range(output_n):
            mu = Dense(size, activation=act_mean[0], kernel_initializer=kernel_init, bias_initializer=bias_init,
                       kernel_regularizer=regularizers.L1L2(l1, l2), name="mean1" + str(k))(i)
            if dropout > 0:
                mu = Dropout(dropout, name="mudrop" + str(k))(mu)
            mu = Dense(size, activation=act_mean[1], kernel_initializer=kernel_init, bias_initializer=bias_init,
                       kernel_regularizer=regularizers.L1L2(l1, l2), name="mean2" + str(k))(mu)
            means.append(Dense(1, name="mean3" + str(k))(mu))

            de = Dense(size, activation=act_deviation[0], kernel_initializer=kernel_init, bias_initializer=bias_init,
                       kernel_regularizer=regularizers.L1L2(l1, l2), name="deviation1" + str(k))(i)
            if dropout > 0:
                de = Dropout(dropout, name="dedrop" + str(k))(de)
            de = Dense(size, activation=act_deviation[1], kernel_initializer=kernel_init, bias_initializer=bias_init,
                       kernel_regularizer=regularizers.L1L2(l1, l2), name="deviation2" + str(k))(de)
            deviations.append(Dense(1, activation=lambda x: tf.nn.elu(x) + 1, name="deviation3" + str(k))(de))

        mean = concatenate(means)
        deviation = concatenate(deviations)

        if optimizer is None:
            optimizer = Adam(learning_rate=alpha)
        if custom_model_build_func is None:
            model = Model(inputs=i, outputs=[mean, deviation])
        else:
            model = custom_model_build_func(inputs=i, outputs=[mean, deviation])
        model.compile(loss=loss, optimizer=optimizer, run_eagerly=run_eagerly)
        return model
