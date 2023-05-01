from graphics import GraphWin

from rl.bonus.cartransfer.TransitionCars import TransitionCars
from rl.helpers.Timer import Timer
from rl.mdp.MDP import MDP
from rl.mdp.State import State
from rl.visual import VisualGrid
from rl.visual.Cell import Cell
from rl.visual.Colorizer import Colorizer
from rl.visual.DrawGrid import DrawGrid


def each_calc_callback(state: State):
    cell: Cell = state.visual
    cell.name = state.name
    cell.exp_return = state.v
    cell.set_state_v(state.v)
    cell.update_v_text()
    cell.win.getMouse()


def sweep_calc_callback(states: [State]):
    cells = []
    for state in states:
        cell: Cell = state.visual
        cell.name = state.name
        cell.exp_return = state.v
        cell.optimal_q = state.optimal_actions[0].key
        cell.opt_action_name = str(cell.optimal_q)
        cell.set_state_v(state.v)
        cell.update_v_text()
        cells.append(cell)
    Colorizer.update_value_colors_cells(cells, lambda c: c.optimal_q)
    # win.getMouse()


def evaluate_calc_callback(states: [State]):
    sweep_calc_callback(states)


def run():
    win = GraphWin('CarTransfer', 1300, 1000)
    Colorizer.setup_colors(150, 50, -50)
    win.yview()
    visual_grid = VisualGrid.build_grid(win, 21, 21, 20, 20, 60, 45)
    draw_grid = DrawGrid(visual_grid)

    mdp = MDP(discount=0.9)
    max_cars1 = 20
    max_cars2 = 20

    # Build all actions
    actions = []
    for a in range(-5, 6):
        actions.append(a)

    # Build states
    for s_location in range(max_cars1 + 1):
        for f_location in range(max_cars2 + 1):

            # invert y
            first_cars_count = 20 - f_location
            second_cars_count = s_location
            name = str(first_cars_count) + " - " + str(second_cars_count)
            visual_grid[s_location][f_location].name = name

            # Build possible actions
            possible_actions = []
            action: int
            for action in actions:
                cars_first = first_cars_count - action
                cars_second = second_cars_count + action
                if 0 <= cars_first <= max_cars1 and 0 <= cars_second <= max_cars2:
                    possible_actions.append(action)
            mdp.add_state(name, possible_actions, info=[first_cars_count, second_cars_count],
                          visual=visual_grid[s_location][f_location])

    draw_grid.draw()

    # Build transitions
    mdp.set_transitions(TransitionCars())

    # Build policy
    mdp.build_constant_policy(0)

    # Add a callback
    # mdp.register_callback_for_each_expected_value_calc(each_calc_callback)
    mdp.register_callback_for_sweep_expected_values_calc(sweep_expected_calc=sweep_calc_callback)
    mdp.register_callback_for_improvement(sweep_calc_callback)
    print("Click left mouse button to start")
    win.getMouse()
    print("Calculating...")
    timer = Timer()
    mdp.gpi(evaluation_epsilon=10)
    print(timer.stop(2))
    print(mdp.get_expected_values())
    print("End")
    win.getMouse()


if __name__ == '__main__':
    run()
