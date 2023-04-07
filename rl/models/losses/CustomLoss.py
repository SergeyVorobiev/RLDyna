import tensorflow as tf


class CustomLoss:

    # Monte Carlo Policy Gradient Loss Function.
    # data should contain all actions that was taken due to episode and rewards accordingly.
    # policy is what the model give automatically as predictions after invoking model.fit function.
    # So according to formula w = w + alpha * G * gradient(ln(Pi(A | S, w))) we can consider it as
    # w = w + alpha * gradient(G * ln(Pi(A | S, w)))
    # w = w + alpha * gradient(mc_policy_gradient), we add minus sign to convert desc to asc. Applying gradient with
    # alpha to the weights happens automatically by using optimizer inside.
    @staticmethod
    def mc_policy_gradient(y, policy):
        actions = y[:, 0]
        actions = tf.dtypes.cast(actions, tf.int64)  # Take chosen actions
        rewards = y[:, 1]
        discount = y[:, 2]
        policy = tf.gather(policy, actions, batch_dims=1)  # Take only policies according to actions
        return -tf.math.log(policy) * rewards * discount

    @staticmethod
    def one_step_sarsa_lambda(y, qs):
        q_next = y[0]
        reward = y[1]
        action = y[2]
        discount = y[4]

        # Calculates the error = R + y * Qn(S, A, w) - Q(S, A, w)
        return reward + discount * q_next - qs[tf.cast(action, tf.int32)]

    # sigma = g - U(S, w)
    # w = w + alpha * sigma * gradient(U(S, w))
    @staticmethod
    def td_loss(g, us):
        sigma = g - us
        return -sigma * us  # Convert desc to asc by -
