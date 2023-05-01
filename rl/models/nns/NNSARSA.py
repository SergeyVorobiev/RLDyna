import tensorflow as tf
from keras import Model


# noinspection PyAbstractClass
class NNSARSA(Model):

    def __init__(self, inputs, outputs, **kwargs):
        super(NNSARSA, self).__init__(inputs, outputs, **kwargs)

    # (x = state, y = [g, action, done])
    @tf.function  # AutoGraph
    def train_step(self, data):
        xs = data[0]
        ys = data[1]
        actions = ys[:, 1]
        g = ys[:, 0]
        actions = tf.cast(actions, tf.int32)

        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            qs = self(xs, training=True)
            q = tf.gather(qs, actions, batch_dims=1)
            error = (g - q) * q
            grads = tape.gradient(-error, self.trainable_variables)

        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {m.name: m.result() for m in self.metrics}
