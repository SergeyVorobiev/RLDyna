import gym
import matplotlib
import tensorflow as tf
import keras
from rl.tasks.EnvRenderer import EnvRenderer
from rl.environments.EnvBuilder import EnvBuilder


class EnvRendererInst:

    def __init__(self, env_builder: EnvBuilder):
        self._env_builder: EnvBuilder = env_builder
        self._env, self._agent = self._env_builder.build_env_and_agent()

    def render(self):
        print("Tensorflow version: " + tf.__version__)
        print("Keras version: " + keras.__version__)
        print("Gym version: " + gym.__version__)
        print("Matplot version: " + matplotlib.__version__)
        list_gpu = tf.config.list_physical_devices('GPU')
        print("GPU Devices: ", list_gpu)
        if len(list_gpu) == 0:
            print("Running on CPU")
        EnvRenderer.render(self._env, self._agent, self._env_builder.get_iterations(),
                           episode_done_listener=self._env_builder.episode_done,
                           iteration_complete_listener=self._env_builder.iteration_complete,
                           stop_render_listener=self._env_builder.stop_render)
