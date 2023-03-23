from keras.engine.base_layer import Layer


class InterceptorLayer(Layer):

    def __init__(self, inputs_interceptor=None, **kwargs):
        super(InterceptorLayer, self).__init__(**kwargs)
        if inputs_interceptor is None:
            self.__inputs_interceptor = lambda inputs: print(inputs.shape)
        else:
            self.__inputs_interceptor = inputs_interceptor

    def call(self, inputs):
        self.__inputs_interceptor(inputs)
        return inputs