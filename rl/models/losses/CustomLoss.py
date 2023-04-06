import tensorflow as tf


class CustomLoss:

    # Monte Carlo Policy Gradient Loss Function.
    # data should contain all actions that was taken due to episode and rewards accordingly.
    # policy is what the model give automatically as predictions after invoking model.fit function.
    @staticmethod
    def mc_policy_gradient(y, policy):
        actions = y[:, 0]
        actions = tf.dtypes.cast(actions, tf.int64)  # Take chosen actions
        rewards = y[:, 1]
        baselines = y[:, 2]
        policy = tf.gather(policy, actions, batch_dims=1)  # Take only policies according to actions
        return -tf.math.log(policy) * (rewards - baselines)  # Apply formula

    @staticmethod
    def one_step_sarsa_lambda(y, qs):
        q_next = y[0]
        reward = y[1]
        action = y[2]
        discount = y[4]

        # Calculates the error = R + y * Qn(S, A, w) - Q(S, A, w)
        return reward + discount * q_next - qs[tf.cast(action, tf.int32)]
