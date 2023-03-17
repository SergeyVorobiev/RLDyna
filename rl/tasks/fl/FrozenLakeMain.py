from rl.dyna.Dyna import Dyna
from rl.env.BasicGridEnv import StateType
from rl.env.FrozenLakeEnv import FrozenLakeEnv
from rl.models.TableSingle import TableSingle
from rl.planning.NoPlanning import NoPlanning
from rl.policy.EGreedyRPolicy import EGreedyRPolicy
from rl.algorithms.Q import Q
from rl.planning.SimplePlanning import SimplePlanning

from rl.tasks.fl.FrozenLakeStatePreparator import FrozenLakeStatePreparator


# =====================================================================================================================
grid_map = [[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 3],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
            [4, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]]

# move, fall in hole, hit a wall, finished
rewards = [-1, -50, -30, 30]

color_map = [[0, 255, 255], [0, 0, 255], [0, 0, 0], [255, 255, 0], [0, 255, 0], [255, 0, 0]]

env = FrozenLakeEnv(StateType.blind, rewards=rewards, grid_map=grid_map, screen_width=1800, screen_height=1000,
                    color_map=color_map, begin_from_start_if_get_in_hole=False)

iterations = 1000000

need_colorize_q_map = True

skip_frames = 2  # to speed up computations

colorize_q_map_frames_skip = 40  # to speed up computations

slow_render = 0  # to see what is going on more carefully
# =====================================================================================================================


def build_brain():
    n_states = env.get_x() * env.get_y()
    discount = 1

    # Policy, 0.02 means that in 2% it will choose the random action to explore
    e_greedy = EGreedyRPolicy(0.02)

    # How many happened situations we can memorize and how often we should use this information for pretraining.
    planning = SimplePlanning(plan_batch_size=300, memory_size=n_states * env.action_space.n)
    # planning = NoPlanning()

    # Iterative algorithm
    algorithm = Q(e_greedy, discount=discount)
    # sarsa = SARSA(e_greedy, alpha=alpha, discount=discount)
    # expected_sarsa = ExpectedSARSA(e_greedy, alpha=alpha, discount=discount)

    # Model keeps the previously learned information and get the data back when needed.
    # It is usually can be tabular or neural network.
    models = [TableSingle(n_states=n_states, n_actions=env.action_space.n)]

    brain = Dyna(models=models, algorithm=algorithm, planning=planning,
                 state_preparator=FrozenLakeStatePreparator(env.get_y()))
    return brain


if __name__ == '__main__':

    # get first state of environment
    state = env.reset()
    key = True

    # build specific Dyna brain
    used_brain: Dyna = build_brain()

    # function to get current q values depending on the state (uses to view the values in cell)
    q_values_getter = lambda raw_state: used_brain.get_q_values(raw_state)

    # Skip some part of screen updating to speed up the process and visualize it at the same time.
    env.set_skip_frame(skip_frames)
    env.draw_map_frame_skip(colorize_q_map_frames_skip)
    env.set_slow_render(slow_render)

    for i in range(iterations):

        # q_supplier - just to draw q values in the cells.
        env.render(q_supplier=q_values_getter, draw_map=need_colorize_q_map)

        # get action the brain decided to use according to the current state.
        action = used_brain.act(state)

        # get next state and reward according to the action from the environment.
        next_state, reward, done, player_prop = env.step(action)

        # make the brain learn depending on state in which it was, action it applied, reward it got,
        # next state it ended up.
        used_brain.learn(state, action, reward, next_state, done, player_prop)

        # assign the new state
        state = next_state

        # if we achieve the goal, print some information, reset state and repeat.
        if done:
            score = player_prop['max_score']
            if key and score == 2:
                key = False
                print(i)
            # print("Score: " + str(player_prop['score']) + " Max score: " + str(player_prop['max_score']))
            state = env.reset()
