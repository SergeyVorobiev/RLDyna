import keras.metrics
import tensorflow as tf
from keras import Model
from tensorflow_probability.python.distributions import Normal


# noinspection PyAbstractClass
# Policy Gradient Continuous
class NNPGC(Model):

    def __init__(self, inputs, outputs, **kwargs):
        super(NNPGC, self).__init__(inputs, outputs, **kwargs)
        self.loss_tracker = keras.metrics.Mean(name="loss")

    # (x = state, y = [sigma, discount, done, actions]
    @tf.function()
    def train_step(self, data):
        xs = data[0]
        ys = data[1]
        sigmas = ys[:, 0]
        discounts = ys[:, 1]
        sigma = tf.expand_dims(sigmas, axis=1)
        discount = tf.expand_dims(discounts, axis=1)
        size = ys.shape[1]
        actions = ys[:, 3:size]  # to get number of actions for 2 actions it will be [3:5)
        with tf.GradientTape() as tape:
            mean_deviation = self(xs, training=True)
            mean = mean_deviation[0]
            deviation = mean_deviation[1]

            dist = Normal(loc=mean, scale=deviation)

            # discount^t * sigma * ln(policy), minus for ASC, note that discount is already precalculated
            # sigma also contains discount
            loss = -dist.log_prob(actions) * sigma * discount
            self.loss_tracker.update_state(tf.reduce_mean(loss))
            grads = tape.gradient(loss, self.trainable_variables)

        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {"loss": self.loss_tracker.result()}
