import tensorflow as tf
from keras import Model


# noinspection PyAbstractClass
class NNSARSALambda(Model):

    def __init__(self, inputs, outputs, lambda_v, **kwargs):
        super(NNSARSALambda, self).__init__(inputs, outputs, **kwargs)
        self._z = []
        self._lambda = lambda_v

    # One step only (x = state, y = [next_q, reward, action, done, discount]
    def train_step(self, data):
        x = data[0]
        y = data[1][0]
        action = y[2]
        done = y[3]
        discount = y[4]
        with tf.GradientTape() as tape:
            qs = self(x, training=True)[0]
            action = tf.cast(action, tf.int32)
            q = qs[action]
        grads = tape.gradient(q, self.trainable_variables)  # Get the gradients from q

        # Initialize Z = 0 (Eligibility traces) and reset them every episode
        if self._z.__len__() == 0:
            for grad in grads:
                v = tf.Variable(tf.zeros(grad.shape), trainable=False)
                self._z.append(v)

        error = self.compiled_loss(
            y,
            qs,
        )

        total_grads = []
        for i in range(len(grads)):

            # z = y * lambda * z + gradient(q(A, S, w)
            self._z[i].assign(discount * self._lambda * self._z[i] + grads[i])

            # total_grad = alpha * error * self._z[i]
            total_grad = -error * self._z[i]  # Convert desc to asc, alpha in opt.
            total_grads.append(total_grad)

        self.optimizer.apply_gradients(zip(total_grads, self.trainable_variables))

        # self.trainable_variables[i].assign_add(total_grad)

        # Reset Z-traces after episode ends
        tf.cond(done > 0.5, lambda: list(map(lambda z, grad: z.assign(0 * grad), self._z, grads)), lambda: self._z)
        return 0

    @staticmethod
    def get_signatures(input_shape, model):

        # To correctly save custom model
        return model.train.get_concrete_function(tf.data.DatasetSpec(element_spec=(
            tf.TensorSpec(shape=input_shape, dtype=tf.float32, name=None),  # states
            tf.TensorSpec(shape=(), dtype=tf.float32, name=None),  # q
            tf.TensorSpec(shape=(), dtype=tf.float32, name=None),  # rewards
            tf.TensorSpec(shape=(), dtype=tf.float32, name=None),  # actions
            tf.TensorSpec(shape=(), dtype=tf.float32, name=None),  # done
            tf.TensorSpec(shape=(), dtype=tf.float32, name=None),  # discount
            # tf.TensorSpec(shape=(), dtype=tf.float32, name=None),  # lambda
            # tf.TensorSpec(shape=(), dtype=tf.float32, name=None),  # alpha
        )))
