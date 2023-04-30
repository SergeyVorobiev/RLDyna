import tensorflow as tf
from keras import Model


# The basic algorithm for continuous TD(N) or MC with eligibility traces according to the Sutton & Barto book:

# sigma = R - R' + U(S', w) - U(S, w)
# R' = R' - alpha_r * sigma
# z = lambda * z + gradient(U(S, w))
# w = w + alpha * sigma * z

# In case if we have alpha_r = 0
# sigma = R + U(S' w) - U(S, w)
# z = lambda * z + gradient(U(S, w))
# w = w + alpha * sigma * z

# In other words:
# sigma = G - U(S, w)
# z = lambda * z + gradient(U(S, w))
# w = w + alpha * sigma * z

# Without traces:
# sigma = G - U(S, w)
# w = w + alpha * sigma * gradient(U(S, w))

# noinspection PyAbstractClass
class NNTDLambda(Model):

    def __init__(self, inputs, outputs, discount, lambda_v, cont_alpha=0.0, **kwargs):
        super(NNTDLambda, self).__init__(inputs, outputs, **kwargs)
        self._z = []
        self._lambda = lambda_v
        self._discount = discount
        self._cont_alpha = cont_alpha  # For continuous problems, if 0 it does not affect and becomes just episodic
        self._r = tf.Variable(0.0, trainable=False)
        self._sigma = tf.Variable(0.0, trainable=False)

        self._total_grads = []
        for t_vars in self.trainable_variables:
            v = tf.Variable(tf.zeros(t_vars.shape), trainable=False)
            self._total_grads.append(v)

    @tf.function  # AutoGraph
    def train_step(self, data):
        xs = data[0]
        ys = data[1]
        gs = ys[:, 0]
        dones = ys[:, 1]
        size = tf.shape(xs)[0]
        i = 0
        self._sigma.assign(0.0)

        # Clear total grads collector
        k = 0
        for t_vars in self.trainable_variables:
            self._total_grads[k].assign(0 * t_vars)
            k += 1

        # In theory, it should always be 1 step calc before we take next action, size > 1 possibly will not converge
        while i < size:
            x = tf.convert_to_tensor([xs[i]])
            g = tf.convert_to_tensor([gs[i]])
            done = dones[i]
            with tf.GradientTape() as tape:
                us = self(x, training=True)[0]

                # G - U(S, w)
                error = self.compiled_loss(
                    g,
                    us,
                )
            grads = tape.gradient(us, self.trainable_variables)  # Get the gradients from u

            # Initialize Z = 0 (Eligibility traces) and reset them every episode
            if self._z.__len__() == 0:
                for grad in grads:
                    v = tf.Variable(tf.zeros(grad.shape), trainable=False)
                    self._z.append(v)

            # This part is for continuous, if cont_alpha = 0 then it works as common episodic without any changes
            # to sigma
            # The basic formula for continuous task is R - R' + U(S', w) - U(S, w) in other words:
            # G - U(S, w) - R'
            self._sigma.assign(error - self._r)

            # According to the formula:
            # R' = R' + alpha * sigma
            self._r.assign_add(self._cont_alpha * self._sigma.value())

            # total_grads = []
            for j in range(len(grads)):
                # z = y * lambda * z + gradient(u(s, w))
                self._z[j].assign(self._discount * self._lambda * self._z[j] + grads[j])
                total_grad = -self._sigma.value() * self._z[j]
                # total_grads.append(total_grad)
                self._total_grads[j].assign_add(total_grad)
            # self.optimizer.apply_gradients(zip(total_grads, self.trainable_variables))

            # Reset Z-traces after episode ends
            tf.cond(done > 0.5, lambda: list(map(lambda z, grad: z.assign(0 * grad), self._z, grads)), lambda: self._z)
            tf.cond(done > 0.5, lambda: self._r.assign(0.0), lambda: self._r)
            i += 1
        self.optimizer.apply_gradients(zip(self._total_grads, self.trainable_variables))

        # Train step should be invoked after each step we collect sigma in callback
        return {"loss": self._sigma.value()}  # {m.name: m.result() for m in self.metrics}
