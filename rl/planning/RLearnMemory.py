from rl.planning.RBaseMemory import RBaseMemory


class RLearnMemory(RBaseMemory):

    def __init__(self, memory_capacity: int):
        super().__init__(memory_capacity)
        self._need_clear = False

    # We clean memory after episode is done
    def memorize_step(self, state, action, reward, next_state, done, truncated, env_props):
        if self._need_clear:
            self.clear_memory()
            self._need_clear = False
        self.memorize((state, action, reward, next_state, done, truncated, env_props))
        if done:
            self._need_clear = True
