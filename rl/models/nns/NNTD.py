import tensorflow as tf
from keras import Model


# The basic algorithm for continuous TD(N)
# noinspection PyAbstractClass
class NNTD(Model):

    def __init__(self, inputs, outputs, **kwargs):
        super(NNTD, self).__init__(inputs, outputs, **kwargs)
        self._sigma = tf.Variable(0.0, trainable=False)

    # xs = [[state]]
    # ys = [[g, ...]] only first feature is important
    @tf.function  # AutoGraph
    def train_step(self, data):
        xs = data[0]
        ys = data[1]
        gs = ys[:, 0]
        size = tf.shape(xs)[0]
        i = 0

        # train_step is expected to have only one row of features to correctly gather sigma
        while i < size:
            x = tf.convert_to_tensor([xs[i]])
            g = tf.convert_to_tensor([gs[i]])
            with tf.GradientTape() as tape:
                us = self(x, training=True)[0]

                # G - U(S, w)
                error = self.compiled_loss(
                    g,
                    us,
                )
                self._sigma.assign(error)

                # sigma * gradient(u)
                # gradient(u * sigma)
                # minus to convert DSC to ASC
                error = -error * us
            grads = tape.gradient(error, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
            i += 1

        # Train step should be invoked after each step we collect sigma in callback
        return {"loss": self._sigma.value()}  # {m.name: m.result() for m in self.metrics}
