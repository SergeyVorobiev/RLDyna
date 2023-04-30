import tensorflow as tf
from keras import Model


# noinspection PyAbstractClass
# Use this model to debug your loss function
class NNLossTest(Model):

    # Specify run_eagerly=True to debug the model
    # @tf.function() Auto-graph
    def train_step(self, data):
        x = data[0]
        y = data[1]
        with tf.GradientTape() as tape:
            predicted = self(x, training=True)

            loss_value = self.custom_loss(y, predicted)
            loss = self.compiled_loss(
                y,
                predicted,
            )
        grads = tape.gradient(loss_value, self.trainable_variables)

        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {m.name: m.result() for m in self.metrics}

    # @tf.function()
    def custom_loss(self, y, predicted):
        return 0

