from keras import Input
from keras.api._v2.keras import initializers
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam

from rl.models.nns.MCPGNNDiscrete import MCPGNNDiscrete


class ShelterNetwork():

    @staticmethod
    def build_model(input_shape, alpha):
        bias_initializer = initializers.Constant(1)
        bias_initializer = None
        kernel_initializer = initializers.GlorotNormal()
        inputs = Input(shape=input_shape)
        x = Conv2D(filters=8, kernel_size=(4, 4), strides=(1, 1), padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(x)
        # x = InterceptorLayer(interceptor_input_func)(x)

        x = Conv2D(filters=16, kernel_size=2, strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=3, strides=1, padding="same")(x)
        # x = InterceptorLayer(interceptor_input_func)(x)

        x = Flatten()(x)
        print(x.shape)
        x = Dense(x.shape[1], activation='relu', bias_initializer=bias_initializer,
                  kernel_initializer=kernel_initializer)(x)
        x = Dense(4, activation='softmax')(x)
        model = MCPGNNDiscrete(inputs, x)

        # run_eagerly - in case you want to debug
        model.compile(optimizer=Adam(learning_rate=alpha), run_eagerly=True)
        return model
