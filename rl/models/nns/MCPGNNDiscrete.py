import tensorflow as tf
from keras import Model


# noinspection PyAbstractClass
class MCPGNNDiscrete(Model):

    @tf.function()
    def train(self, data):
        for state, action, reward, baseline in data:
            with tf.GradientTape() as tape:

                # So here we use Policy-Gradient Control formula:
                # w = w + alpha + gamma * (G - b) * gradient(ln(policy))
                # alpha - is a learning rate, we specify it in Adam (SGD) opt
                # b - is a baseline (can depend on state, like U(S), reward average, random
                # or computed by any tricky function) it reduces the variance
                # gamma & G is contained in reward
                # policy - we got them from the model according to the action we did
                policy = self(state)[0][action]
                loss = -tf.math.log(policy) * (reward - baseline)  # Minus to convert desc to asc
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return 0

    @staticmethod
    def get_signatures(input_shape, model):

        # To correctly save custom model
        # to know signatures invoke print(data) into MCPGModel.update()
        return model.train.get_concrete_function(tf.data.DatasetSpec(element_spec=(
            tf.TensorSpec(shape=input_shape, dtype=tf.float32, name=None),  # states
            tf.TensorSpec(shape=(), dtype=tf.int32, name=None),  # actions
            tf.TensorSpec(shape=(), dtype=tf.float32, name=None),  # rewards
            tf.TensorSpec(shape=(), dtype=tf.float32, name=None))))  # baselines
