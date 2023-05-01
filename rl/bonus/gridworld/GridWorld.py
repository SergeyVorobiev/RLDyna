import random

from graphics import Rectangle, Point, color_rgb

from rl.helpers.Timer import Timer
from rl.mdp.MDP import MDP
from rl.mdp.State import State
from rl.mdp.TransitionMap import TransitionMapBuilder
from rl.visual.Cell import Cell


class GridWorld(object):

    def __init__(self, win, width, height, epsilon, spots_count, draw_text, spot_reward, transition_reward,
                 out_bound_reward, out_bound_value, start_grid_point, cell_size):
        self.transition_reward = transition_reward
        self.width = width
        self.height = height
        self.border_x_down = 0
        self.border_y_down = 0
        self.draw_text = draw_text
        self.spots_count = spots_count
        self.border_x_up = self.width - 0
        self.border_y_up = self.height - 0
        self.spot_x = self.border_x_down
        self.spot_y = self.border_y_down
        self.win = win
        self.epsilon = epsilon
        self.spot_reward = spot_reward
        self.out_bound_value = out_bound_value
        self.out_bound_reward = out_bound_reward
        self.grid = []
        self.target_spots: [Cell] = []
        rect_start_y = start_grid_point.y
        for y in range(height):
            rect_start_x = start_grid_point.x
            self.grid.append([])
            for x in range(width):
                sp = Point(rect_start_x, rect_start_y)
                ep = Point(rect_start_x + cell_size, rect_start_y + cell_size)
                rect = Rectangle(sp, ep)
                self.grid[y].append(Cell(0, color_rgb(0, 240, 0), self.draw_text, rect, win, x, y))
                rect_start_x += cell_size
            rect_start_y += cell_size
        self.mdp = None

    def __create_mdp(self):
        width = self.width
        height = self.height
        mdp: MDP = MDP()
        actions = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        action_names = ["l", "r", "u", "d"]
        state_ids = []
        for y in range(height):
            state_ids.append([])
            for x in range(width):
                name = str(x) + "-" + str(y)
                state_id = mdp.add_state(name, action_names, visual=self.grid[y][x])
                state_ids[y].append(state_id)
        loop_state_id = mdp.add_state("termination", ["stay"])
        transition_builder: TransitionMapBuilder = MDP.transition_map_builder()
        transition_builder.add_transition(loop_state_id, "stay", loop_state_id, 1, 0)
        for y in range(height):
            for x in range(width):
                for action, action_name in zip(actions, action_names):
                    dx = action[0]
                    dy = action[1]
                    pos_x = x + dx
                    pos_y = y + dy
                    if pos_x < 0 or pos_y < 0 or pos_x >= width or pos_y >= height:
                        next_state_id = state_ids[y][x]
                        reward = self.out_bound_reward
                    else:
                        reward = self.transition_reward
                        next_state_id = state_ids[pos_y][pos_x]
                    for termination in self.target_spots:
                        term_x = termination.pos_x
                        term_y = termination.pos_y
                        if x == term_x and y == term_y:
                            next_state_id = loop_state_id
                            reward = self.spot_reward
                    transition_builder.add_transition(state_ids[y][x], action_name, next_state_id, 1, reward)
        mdp.set_transitions(transition_builder.build())
        mdp.build_uniform_policy()
        return mdp

    def draw(self):
        for row in self.grid:
            for cell in row:
                cell.draw()

    def get_random_cell_not_in_spots(self) -> Cell:
        x = random.randrange(self.border_x_down, self.border_x_up, 1)
        key_x = False
        for spot in self.target_spots:
            spot_x = spot.pos_x
            if spot_x == x:
                key_x = True
                break
        if key_x:
            possible_y = random.randrange(self.border_y_down, self.border_y_up, 1)
            key_y = True
            while key_y:
                key_y = False
                possible_y = random.randrange(self.border_y_down, self.border_y_up, 1)
                for spot in self.target_spots:
                    spot_y = spot.pos_y
                    if spot_y == possible_y:
                        key_y = True
                        break
            y = possible_y
        else:
            y = random.randrange(self.border_y_down, self.border_y_up, 1)
        return self.get_cell(x, y)

    def get_cell(self, x, y, accessible=True) -> Cell or None:
        min_x = 0
        max_x = self.width
        min_y = 0
        max_y = self.height
        if accessible:
            min_x = self.border_x_down
            max_x = self.border_x_up
            min_y = self.border_y_down
            max_y = self.border_y_up
        if x < min_x or x >= max_x:
            return None
        if y < min_y or y >= max_y:
            return None
        cell: Cell = self.grid[y][x]
        return cell

    def update_text(self):
        for row in self.grid:
            for cell in row:
                cell.update_text()

    def reset_grid(self):
        for row in self.grid:
            for cell in row:
                cell: Cell = cell
                cell.reset_data()
                cell.is_target = False
                cell.set_default_color(0, 240, 0)
                cell.reset_color()
                cell.place_text("0")
                # cell.update_text()
        self.target_spots = []
        for _ in range(self.spots_count):
            self.__new_target_spot()
        # self.__new_target_spot(3, 3)
        # self.__new_target_spot(0, 0)
        self.mdp = self.__create_mdp()
        # self.mdp.register_callback_for_each_expected_value_calc(self.each_expected_value_update_callback)
        # self.mdp.register_callback_for_sweep_expected_values_calc(self.sweep_expected_values_update_callback)
        self.mdp.register_callback_for_improvement(self.update_state_callback)
        self.mdp.register_improve_end_callback(self.update_state_callback)
        # self.mdp.register_callback_for_evaluation(self.improve_end_callback)

    def __new_target_spot(self, x=-1, y=-1):
        if x >= 0 and y >= 0:
            cell = self.get_cell(x, y)
        else:
            cell = self.get_random_cell_not_in_spots()
        cell.is_target = True
        cell.set_default_color(255, 240, 0)
        cell.reset_color()
        cell.update_text()
        self.target_spots.append(cell)

    def update_state_callback(self, state: State):
        self.update_text()
        # self.win.getMouse()

    def update_states_callback(self, states: [State]):
        for state in states:
            self.update_state_callback(state)

    def evaluate_expected_values(self):
        timer = Timer()
        timer.start()
        iterations = self.mdp.gpi(self.epsilon)
        print("Evaluate time: " + str(timer.stop(5)) + " | Iterations: " + str(iterations))
        # self.sweep_expected_values_update_callback(self.mdp.get_states())
        # print(error)
        return iterations


class Player(object):

    def __init__(self, cell: Cell, border_x, border_y):
        self.cell: Cell = cell
        self.border_x = border_x
        self.border_y = border_y
        self.last_direction = [-1, -1]
        self.dir_map = {"l": [-1, 0],
                        "r": [1, 0],
                        "u": [0, -1],
                        "d": [0, 1]}
        self.dir_names = ["l", "r", "u", "d"]
        # A = {l, r, u, d,}
        # x, y, prob
        self.directions = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        self.cum_reward = 0

    def get_dir_name(self, index):
        return self.dir_names[index]

    def move(self, direct, world: GridWorld):
        self.last_direction = direct
        pos_x = self.cell.pos_x
        pos_y = self.cell.pos_y
        pos_x += direct[0]
        pos_y += direct[1]
        cell: Cell = world.get_cell(pos_x, pos_y)
        if cell is not None:
            self.cum_reward += world.transition_reward
            if cell.is_target:
                self.cum_reward += world.spot_reward
            # self.cell.reset_color()
            self.cell.color = "brown"
            self.cell.update_color()
            self.cell = cell
            self.cell.color = "red"
            self.cell.update_color()
        else:
            self.cum_reward += world.out_bound_reward

    def is_last_direction(self, direction):
        return self.last_direction[0] == direction[0] and self.last_direction[1] == direction[1]

    def update_position(self, world: GridWorld):
        self.last_direction = [-1, -1]
        cell: Cell = world.get_random_cell_not_in_spots()
        # pos_y = 10
        cur_pos_x = self.cell.pos_x
        cur_pos_y = self.cell.pos_y
        pos_x = cell.pos_x
        pos_y = cell.pos_y
        if cur_pos_x == pos_x and cur_pos_y == pos_y:
            return
        self.cell.reset_color()
        self.cell = cell
        self.cell.color = "red"
        self.cell.update_color()

    def update(self):
        self.cell.color = "red"
        self.cell.update_color()

    def check_borders(self, pos_x, pos_y):
        if pos_x > self.border_x - 1:
            pos_x = self.border_x - 1
        elif pos_x < 0:
            pos_x = 0
        if pos_y > self.border_y - 1:
            pos_y = self.border_y - 1
        elif pos_y < 0:
            pos_y = 0
        return pos_x, pos_y

    def move_player_random(self, world):
        direct = self.directions[random.randrange(0, 4)]
        # while self.is_opposite(direct):
        #    direct = self.directions[random.randrange(0, 4)]
        self.move(direct, world)

    def is_opposite(self, direction):
        opp_x = direction[0] * -1
        opp_y = direction[1] * -1
        return self.is_last_direction([opp_x, opp_y])

    def __choose_step(self, next_steps):
        next_direction = None
        for next_step in next_steps:
            if self.is_last_direction(next_step):
                next_direction = next_step
                break
        if next_direction is None:
            next_direction = next_steps[random.randrange(0, len(next_steps), 1)]
        return next_direction

    def move_player_by_policy(self, world: GridWorld):
        cell = self.cell
        next_steps = []
        for i in range(cell.optimal_actions_count):
            action = cell.actions[i]
            next_steps.append(self.dir_map[action.name])
        next_direction = self.__choose_step(next_steps)
        self.move(next_direction, world)
