from gym import Env

from rl.dyna.Dyna import Dyna


class EnvRenderer:

    @staticmethod
    def render(env: Env, agent: Dyna, iterations: int):

        # get first state of environment
        state = env.reset()

        for i in range(iterations):

            # q_supplier - just to draw q values in the cells.
            env.render()

            # get action the agent decided to use according to the current state.
            action = agent.act(state)

            # get next state and reward from the environment according to the action.
            next_state, reward, done, player_prop = env.step(action)

            # make the agent learn depending on the state it was, action it applied, reward it got,
            # next state it ended up.
            agent.learn(state, action, reward, next_state, done, player_prop)

            # assign the new state
            state = next_state

            # if we've achieved the goal, print some information, reset state and repeat.
            if done:
                state = env.reset()
                agent.improve_policy()
                agent.clear_memory()