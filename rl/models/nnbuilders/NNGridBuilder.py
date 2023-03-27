import tensorflow as tf
from keras import Input, Model
from keras.api._v2.keras import initializers
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense, Dropout, AveragePooling2D
from keras.optimizers import Adam


from rl.helpers.ImageHelper import save_input_as_image
from rl.models.nnlayers.InterceptorLayer import InterceptorLayer

#gpus = tf.config.list_physical_devices('GPU')
#tf.config.set_visible_devices(gpus[0], 'GPU')

#tf.config.set_visible_devices([], 'GPU')


def interceptor_input_func(inputs):
    pixels = inputs[0, :, :, :]
    for i in range(pixels.shape[2]):
        f_image = pixels[:, :, i]
        save_input_as_image(f_image, str(i))


class NNGridBuilder:

    def __init__(self):
        pass

    @staticmethod
    def build_frozen_lake_around_supporter():
        bias_initializer = initializers.Constant(1)
        kernel_initializer = initializers.GlorotNormal()
        inputs = Input(shape=(2, 2, 1))
        x = inputs
        x = Conv2D(filters=1, kernel_size=1, strides=1, padding='valid')(inputs)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        #x = InterceptorLayer(interceptor_input_func)(x)
        x = Flatten()(x)
        print(x.shape)
        x = Dense(16, activation='relu', bias_initializer=bias_initializer,
                  kernel_initializer=kernel_initializer)(x)
        x = Dense(4, activation='linear', kernel_initializer=kernel_initializer)(x)
        model = Model(inputs, x)
        model.compile(loss='mse', optimizer=Adam())
        return model

    @staticmethod
    def build_simple_frozen_lake_cnn(input_shape, n_actions, kernel_initializer):
        bias_initializer = initializers.Constant(1)
        inputs = Input(shape=input_shape)
        x = inputs
        # x = Rescaling(scale=1./255)(x) # We already did this in Prepare class

        x = Conv2D(filters=4, kernel_size=(4, 4), strides=(1, 1), padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(x)
        # x = InterceptorLayer(interceptor_input_func)(x)

        x = Conv2D(filters=8, kernel_size=2, strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=3, strides=1, padding="same")(x)
        # x = InterceptorLayer(interceptor_input_func)(x)

        x = Flatten()(x)
        print(x.shape)
        x = Dense(x.shape[1], activation='relu', bias_initializer=bias_initializer,
                  kernel_initializer=kernel_initializer)(x)
        # x = Dense(x.shape[1], activation='relu', bias_initializer=bias_initializer,
        #          kernel_initializer=kernel_initializer)(x)
        x = Dense(n_actions, activation='linear', kernel_initializer=kernel_initializer)(x)

        model = Model(inputs, x)
        model.compile(loss='mse', optimizer=Adam())
        return model
