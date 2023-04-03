from gym import Env

from rl.dyna.Dyna import Dyna


class EnvRenderer:

    @staticmethod
    def render(env: Env, agent: Dyna, iterations: int, episode_done_listener=None, stop_render_listener=None,
               iteration_complete_listener=None):

        # get first state of environment.
        state = env.reset()

        for i in range(iterations):

            env.render()

            # get action the agent decided to use according to the current state.
            action = agent.act(state)

            # get next state and reward from the environment according to the action.
            next_state, reward, done, player_prop = env.step(action)

            # make the agent learn depending on the state it was, action it applied, reward it got,
            # next state it ended up.
            agent.learn(state, action, reward, next_state, done, player_prop)

            if iteration_complete_listener is not None:
                iteration_complete_listener(state, action, reward, next_state, done, player_prop)

            state = next_state

            # if we've achieved the goal, print some information, reset state and repeat.
            if done:
                state = env.reset()
                agent.improve_policy()
                agent.clear_memory()
                if episode_done_listener is not None:
                    episode_done_listener(player_prop)
            if stop_render_listener is not None and stop_render_listener():
                break
