from rl.tasks.EnvRenderer import EnvRenderer
from rl.environments.EnvBuilder import EnvBuilder


class EnvRendererInst:

    def __init__(self, env_builder: EnvBuilder):
        self._env_builder: EnvBuilder = env_builder
        self._env, self._agent = self._env_builder.build_env_and_agent()

    def render(self):
        EnvRenderer.render(self._env, self._agent, self._env_builder.get_iterations(),
                           episode_done_listener=self._env_builder.episode_done,
                           iteration_complete_listener=self._env_builder.iteration_complete,
                           stop_render_listener=self._env_builder.stop_render)
