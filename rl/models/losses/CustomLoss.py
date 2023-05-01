import tensorflow as tf


# To generalize the approach all losses should be converted to ASC by adding minus (in case if we do not use custom
# model all will work) but you need to take care if you use traces or some additional calculations in custom model
# that the loss is already converted
class CustomLoss:

    # Monte Carlo Policy Gradient Loss Function.
    # data should contain all actions that was taken due to episode and rewards accordingly.
    # policy is what the model give automatically as predictions after invoking model.fit function.
    # So according to formula w = w + alpha * G * gradient(ln(Pi(A | S, w))) we can consider it as
    # w = w + alpha * gradient(y^tG * ln(Pi(A | S, w))) y^t is a discount, important to note that G is also discounted
    # w = w + alpha * gradient(mcpgd), we add minus sign to convert desc to asc. Applying gradient with
    # alpha to the weights happens automatically by using optimizer inside.
    @staticmethod
    @tf.function()
    def mcpgd(y, policy):
        gs = y[:, 0]
        actions = y[:, 1]
        actions = tf.dtypes.cast(actions, tf.int64)  # Take chosen actions
        discount = y[:, 2]
        policy = tf.gather(policy, actions, batch_dims=1)  # Take only policies according to actions
        return -tf.math.log(policy) * gs * discount

    # Only for usage in NNPGDLambda
    @staticmethod
    @tf.function()
    def mcpgd_traces(y, policy):
        return tf.math.log(policy)

    # N-step q learning, we suppose that G = R + yQ'(S, A, w)
    # We expect that gradient(Q(S, A, w)) is calculated inside custom train_step or else you also need to multiply
    # error by Q - -error * q
    @staticmethod
    @tf.function()
    def q_loss(y, qs):
        g = y[0]
        action = y[1]
        q = qs[tf.cast(action, tf.int32)]

        # N - step, for one step G = R + yQ'(S, A, w)
        # Calculates the error = G - Q(S, A, w)
        error = (g - q)
        return -error  # Convert desc to asc, alpha in opt.

    # sigma = g - U(S, w)
    # w = w + alpha * sigma * gradient(U(S, w))
    # w = w + alpha * gradient(sigma * U(S, w))
    # w = w + alpha * gradient(mc_loss) # we add minus to convert w - alpha to w + alpha
    # y = [g, done]
    @staticmethod
    @tf.function()
    def mc_loss(y, us):
        g = y[:, 0]
        u = us[:, 0]
        sigma = g - u
        return -sigma * u  # Convert desc to asc by -

    # The gradient and minus should be added inside custom train_step, used in NNTDLambda, NNTD
    # Do not use it with MCPGAlgorithms and default model, because by default g will have shape 2 [g, done] instead of 1
    # use mc_loss instead
    @staticmethod
    @tf.function()
    def mc_loss_custom(g, us):
        return g - us

    @staticmethod
    @tf.function()
    def custom_loss_gaussian(y, x):
        mean = x[:, 0]
        deviation = x[:, 1]
        action = y[:, 0]
        sigma = y[:, 1]
        discount = y[:, 2]

        # This is Gaussian PDF formula, see Wiki
        # You can also just use:
        # tensorflow_probability.python.distributions
        # -Normal(loc=mean, scale=deviation).log_prob(action)) * sigma * discount
        degree = ((action - mean) / deviation) ** 2
        numerator = tf.exp(-0.5 * degree)
        deviation += 0.000001
        denominator = deviation * 2.50662827463  # sqrt 2 * pi
        pdf_value = numerator / denominator

        # Convert pdf value to log probability
        return -tf.math.log(pdf_value + 0.000001) * sigma * discount
