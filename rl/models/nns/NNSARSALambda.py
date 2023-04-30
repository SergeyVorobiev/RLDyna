import tensorflow as tf
from keras import Model


# noinspection PyAbstractClass
class NNSARSALambda(Model):

    def __init__(self, inputs, outputs, lambda_v, discount, **kwargs):
        super(NNSARSALambda, self).__init__(inputs, outputs, **kwargs)
        self._z = []
        self._lambda = lambda_v
        self._discount = discount

        self._total_grads = []
        for t_vars in self.trainable_variables:
            v = tf.Variable(tf.zeros(t_vars.shape), trainable=False)
            self._total_grads.append(v)

    # (x = state, y = [g, action, done])
    @tf.function  # AutoGraph
    def train_step(self, data):
        xs = data[0]
        ys = data[1]
        size = tf.shape(xs)[0]
        gs = ys[:, 1:ys.shape[1]]

        # Clear total grads collector
        k = 0
        for t_vars in self.trainable_variables:
            self._total_grads[k].assign(0 * t_vars)
            k += 1

        # In theory, it should always be 1 step calc before we take next action, size > 1 possibly will not converge
        i = 0
        while i < size:
            x = tf.convert_to_tensor([xs[i]])
            y = ys[i]
            action = tf.cast(y[1], tf.int32)
            done = y[0]

            with tf.GradientTape() as tape:
                qs = self(x, training=True)[0]
                q = qs[action]
                error = self.compiled_loss(
                    y,
                    qs,
                )
            grads = tape.gradient(q, self.trainable_variables)  # Get the gradients from q

            # Initialize Z = 0 (Eligibility traces) and reset them every episode
            if self._z.__len__() == 0:
                for grad in grads:
                    v = tf.Variable(tf.zeros(grad.shape), trainable=False)
                    self._z.append(v)

            for j in range(len(grads)):

                # z = y * lambda * z + gradient(q(A, S, w))
                self._z[j].assign(self._discount * self._lambda * self._z[j] + grads[j])

                # total_grad = alpha * error * self._z[j]
                total_grad = error * self._z[j]
                self._total_grads[j].assign_add(total_grad)
            # self.trainable_variables[i].assign_add(total_grad)

            # Reset Z-traces after episode ends
            tf.cond(done > 0.5, lambda: list(map(lambda z, grad: z.assign(0 * grad), self._z, grads)), lambda: self._z)
            i += 1

        self.optimizer.apply_gradients(zip(self._total_grads, self.trainable_variables))
        return {m.name: m.result() for m in self.metrics}

    # is not used
    @staticmethod
    def get_signatures(input_shape, model):

        # To correctly save custom model
        return model.train.get_concrete_function(tf.data.DatasetSpec(element_spec=(
            tf.TensorSpec(shape=input_shape, dtype=tf.float32, name=None),  # states
            tf.TensorSpec(shape=(), dtype=tf.float32, name=None),  # g
            tf.TensorSpec(shape=(), dtype=tf.float32, name=None),  # actions
            tf.TensorSpec(shape=(), dtype=tf.float32, name=None),  # done
        )))
