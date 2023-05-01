import tensorflow as tf
from keras import Model


# noinspection PyAbstractClass
# Policy Gradient Discrete
class NNPGDLambda(Model):

    def __init__(self, inputs, outputs, discount, lambda_v, **kwargs):
        super(NNPGDLambda, self).__init__(inputs, outputs, **kwargs)
        self._z = []
        self._lambda = lambda_v
        self._discount = discount

        self._total_grads = []
        for t_vars in self.trainable_variables:
            v = tf.Variable(tf.zeros(t_vars.shape), trainable=False)
            self._total_grads.append(v)

    # We expect that sigma is our baseline that comes from another model without conversion (minus for asc)
    # (x = state, y = [sigma, action, discount, done])
    @tf.function  # AutoGraph
    def train_step(self, data):
        xs = data[0]
        ys = data[1]
        size = tf.shape(xs)[0]

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
            sigma = y[0]
            action = tf.dtypes.cast(y[1], tf.int64)
            pre_discount = y[2]
            done = y[3]
            with tf.GradientTape() as tape:
                probs = self(x, training=True)[0]
                policy = probs[action]

                # Only log(policy)
                loss = self.compiled_loss(
                    y,
                    policy,
                )
            grads = tape.gradient(loss, self.trainable_variables)  # Get the gradients from pi(s, a)

            # Initialize Z = 0 (Eligibility traces) and reset them every episode
            if self._z.__len__() == 0:
                for grad in grads:
                    v = tf.Variable(tf.zeros(grad.shape), trainable=False)
                    self._z.append(v)

            # total_grads = []
            for j in range(len(grads)):
                # in first case we use const discount in second - precalculated i.e. 0.9 * 0.9 * 0.9  etc.
                # z = discount * lambda * z + pre_discount * gradient(ln(policy))
                self._z[j].assign(self._discount * self._lambda * self._z[j] + pre_discount * grads[j])

                # total_grad = alpha * error * self._z[j]
                total_grad = -sigma * self._z[j]  # convert here to ASC
                # total_grads.append(total_grad)
                self._total_grads[j].assign_add(total_grad)
            # self.optimizer.apply_gradients(zip(total_grads, self.trainable_variables))

            # Reset Z-traces after episode ends
            tf.cond(done > 0.5, lambda: list(map(lambda z, grd: z.assign(0 * grd), self._z, grads)), lambda: self._z)
            i += 1
        self.optimizer.apply_gradients(zip(self._total_grads, self.trainable_variables))
        return {m.name: m.result() for m in self.metrics}
