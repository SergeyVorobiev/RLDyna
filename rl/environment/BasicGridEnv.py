from abc import abstractmethod
from enum import Enum
from time import sleep

from graphics import GraphWin
from gym import Env
from gym.vector.utils import spaces
import numpy as np

from rl.visual import VisualGrid
from rl.visual.Cell import Cell
from rl.visual.DrawGrid import DrawGrid
from rl.visual.GridWorldPolicy import update_policy_colors_grid

road = 0
hole = 1
wall = 2
finish = 3
start = 4
pawn = 5
stay_v = 0
move_v = 1
start_v = -1


# first element of array is a reward and the second is - stay where
# it is (0) / - start from start (-1) / - can stay here (1)

class StateType(Enum):
    blind = 0
    around = 1
    all_map = 2
    none = 3


class DrawInfo(Enum):
    short = 0
    full = 1
    none = 2


class BasicGridEnv(Env):

    def __init__(self, grid_map: list, state_type: StateType, transition_map: list, color_map: list):
        self.__transition_map = transition_map
        self.grid_map = grid_map
        self._x = len(grid_map[0])
        self._y = len(grid_map)
        self.__color_map = color_map
        self.__moves = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        self.action_names = ["left", "right", "top", "bottom"]
        self.action_labels = ["<-", "->", "^", "v"]
        self._draw_info = True
        self._short = False
        self.action_space = spaces.Discrete(4)
        self.__width = 0
        self.__height = 0
        self._draw_map_limiter = 0
        self._draw_map_frame_skip = 0
        if state_type is None:
            state_type = StateType.none
        self._state_type = state_type
        self.__build_width_height()
        self.observation_space = spaces.Box(np.array([0, 0]), np.array([self.__width, self.__height]), dtype=np.int32)
        self.__start_position = self.__get_start_position()
        # self.grid_map[self.__start_position[1]][self.__start_position[0]] = pawn
        self.__position = list(self.__start_position)
        self._score = None
        self.__done = False
        self.__last_position = list(self.__start_position)
        self.win = None
        self.__visual_grid = None
        self._max_score = None
        self.__player_prop = {"score": self._score,
                              "max_score": self._max_score}
        self._episodes = 0
        self._transition_reward = self.__transition_map[0][0]
        self._hole_reward = self.__transition_map[1][0]
        self._wall_reward = self.__transition_map[2][0]
        self._finish_reward = self.__transition_map[3][0]
        self.frame_count = 0
        self._skip_frames = 0
        self._slow_render_time = 0
        self.n_states = self._get_n_states()
        self._steps = 0
        self._max_steps = 0
        self._episode_steps = 0

    def set_state_type(self, state_type: StateType):
        self._state_type = state_type

    def _get_n_states(self):
        size = 0
        for row in self.grid_map:
            size += row.__len__()
        return size

    def get_x(self):
        return self._x

    def get_y(self):
        return self._y

    def __build_width_height(self):
        self.__height = self.grid_map.__len__()
        last_width = -1
        for row in self.grid_map:
            width = row.__len__()
            if last_width == -1:
                last_width = width
            elif width != last_width:
                raise Exception("Wrong grid width")
        self.__width = last_width

    def __get_start_position(self):
        y = 0
        for row in self.grid_map:
            x = 0
            for value in row:
                if value == start:
                    return [x, y]
                x += 1
            y += 1
        raise Exception("Can not find start.")

    def __check_border(self, x, y):
        border = False
        reward = self._transition_reward
        if x < 0:
            x = 0
            reward = self._wall_reward
            border = True
        elif x == self.__width:
            x = self.__width - 1
            reward = self._wall_reward
            border = True
        if y < 0:
            y = 0
            reward = self._wall_reward
            border = True
        elif y == self.__height:
            y = self.__height - 1
            reward = self._wall_reward
            border = True
        return x, y, border, reward

    def draw_path(self, q_values_getter):
        for row in self.__visual_grid:
            for cell in row:
                action = np.argmax(q_values_getter([cell.pos_x, cell.pos_y]))
                move = self.__moves[action]
                x = cell.pos_x + move[0]
                y = cell.pos_y + move[1]
                x, y, border, reward = self.__check_border(x, y)
                if not border:
                    place = self.grid_map[y][x]
                    if place == road:
                        cell = self.__visual_grid[x][y]
                        cell.set_color_and_update(255, 255, 0)

    def draw_info(self, info_type: DrawInfo):
        if info_type == DrawInfo.none:
            self._draw_info = False
        elif info_type == DrawInfo.short:
            self._draw_info = True
            self._short = True
        else:
            self._draw_info = True
            self._short = False

    def step(self, action):
        if self.__done:
            raise Exception("Game is done.")
        self._steps += 1
        self._episode_steps += 1
        move = self.__moves[action]
        x = self.__position[0]
        y = self.__position[1]
        self.__last_position[0] = x
        self.__last_position[1] = y
        x = x + move[0]
        y = y + move[1]
        x, y, border, reward = self.__check_border(x, y)
        if not border:
            place = self.grid_map[y][x]
            transition = self.__transition_map[place]
            reward = transition[0]
            position = transition[1]
            if position == move_v:
                self.__position[0] = x
                self.__position[1] = y
                cell = self.__visual_grid[x][y]
                cell.marked = True
            elif position == start_v:
                self.__position[0] = self.__start_position[0]
                self.__position[1] = self.__start_position[1]
            if place == finish:
                #bonus = random.randrange(0, 4)
                #reward += bonus
                self.__done = True
        self._score += reward
        if self.__done:
            if self._max_score is None or self._max_score < self._score:
                self._max_score = self._score
                self._max_steps = self._steps
            self.game_done()
        self.__player_prop['score'] = self._score
        self.__player_prop['max_score'] = self._max_score
        return self._get_state(), reward, self.__done, self.__player_prop

    def _get_state(self):
        if self._state_type == StateType.blind:
            state = np.array(self.__position, dtype=np.int32)
        elif self._state_type == StateType.around:
            raise Exception("Not implement")
        elif self._state_type == StateType.none:
            raise Exception("State type is wrong")
        else:
            spot = self.grid_map[self.__position[1]][self.__position[0]]
            self.grid_map[self.__position[1]][self.__position[0]] = pawn
            state = np.array(self.grid_map)
            self.grid_map[self.__position[1]][self.__position[0]] = spot
        return state

    @abstractmethod
    def game_done(self):
        ...

    def reset(self):
        self._episodes += 1
        self.__done = False
        self.__position[0] = self.__start_position[0]
        self.__position[1] = self.__start_position[1]
        self._score = 0
        self._episode_steps = 0
        if self.__visual_grid is not None:
            for row in self.__visual_grid:
                for cell in row:
                    cell.marked = False
                    cell.reset_color()
        return self._get_state()

    def set_skip_frame(self, skip_frames: int):
        self._skip_frames = skip_frames

    def draw_map_frame_skip(self, skip_frames: int):
        self._draw_map_frame_skip = skip_frames

    def set_slow_render(self, slow_render_time):
        self._slow_render_time = slow_render_time

    @abstractmethod
    def get_world_params(self):
        ...

    @abstractmethod
    def after_world_created(self):
        ...

    def update_colors(self):
        for row in self.__visual_grid:
            for cell in row:
                spot = self.grid_map[cell.pos_y][cell.pos_x]
                color = self.__color_map[spot]
                cell.set_default_color(color[0], color[1], color[2])
                cell.reset_color()

    def __create_and_draw_world(self):
        name, width, height, start_draw_point_x, start_draw_point_y, cell_width, cell_height = self.get_world_params()
        self.win = GraphWin(name, width, height)
        visual_grid = VisualGrid.build_grid(self.win, self.__width, self.__height, start_draw_point_x,
                                            start_draw_point_y, cell_width, cell_height)
        cell: Cell
        pawn_color = self.__color_map[pawn]
        for row in visual_grid:
            for cell in row:
                spot = self.grid_map[cell.pos_y][cell.pos_x]
                cell.state_name = str(cell.pos_x) + " - " + str(cell.pos_y)
                cell.a_labels = self.action_labels
                color = self.__color_map[spot]
                cell.set_default_color(color[0], color[1], color[2])
                cell.reset_color()
                cell.draw_text = self._draw_info
                if cell.pos_x == self.__start_position[0] and cell.pos_y == self.__start_position[1]:
                    cell.set_color_and_update(pawn_color[0], pawn_color[1], pawn_color[2])
        draw = DrawGrid(visual_grid)
        draw.draw()
        # draw.update_text()
        self.__first_render = False
        self.__visual_grid = visual_grid
        self.after_world_created()

    def _get_state_for_values(self, x, y):
        if self._state_type == StateType.blind:
            state = np.array([x, y], dtype=np.int32)
            spot = self.grid_map[y][x]
            if spot == road or spot == start:
                return state
            return None
        elif self._state_type == StateType.around:
            raise Exception("Not implemented yet")
        else:
            spot = self.grid_map[y][x]
            self.grid_map[y][x] = pawn
            state = np.array(self.grid_map)
            self.grid_map[y][x] = spot
        return state

    def __update_values(self, q_supplier, u_supplier, draw_map = False, draw_path = False):
        if draw_map and self._draw_map_limiter > self._draw_map_frame_skip:
            self._draw_map_limiter = 0
            update_policy_colors_grid(grid=self.__visual_grid, cell_value_func=self.__get_cell_value)
        if q_supplier is not None or u_supplier is not None:
            for row in self.__visual_grid:
                for cell in row:
                    state = self._get_state_for_values(cell.pos_x, cell.pos_y)
                    if state is None:
                        continue
                    if q_supplier is not None:
                        if self.grid_map[cell.pos_y][cell.pos_x] == 0:
                            cell.set_q_values(q_supplier(state))
                    if u_supplier is not None:
                        if self.grid_map[cell.pos_y][cell.pos_x] == 0:
                            cell.set_u_value(u_supplier(state))
                    if draw_path and cell.marked and cell.pos_x != self.__position[0] and cell.pos_y != self.__position[1]:
                        cell.set_color_and_update(200, 100, 255)
                    cell.update_qstext(self._short)

    @staticmethod
    def __get_cell_value(cell):
        return cell.get_max_q()

    @abstractmethod
    def after_render(self):
        ...

    def render(self, mode="human", q_supplier=None, u_supplier=None, draw_map = False, draw_path = False):
        if self.win is None:
            self.__create_and_draw_world()
        else:
            self._draw_map_limiter += 1
            if self.frame_count == self._skip_frames:
                self.frame_count = 0
                self.win.update()
                self.__update_values(q_supplier, u_supplier, draw_map, draw_path)
                self.after_render()
            else:
                self.frame_count += 1
            self.__visual_grid[self.__last_position[0]][self.__last_position[1]].reset_color()
            self.__visual_grid[self.__position[0]][self.__position[1]].set_color_and_update(255, 0, 0)
            if self._slow_render_time > 0:
                sleep(self._slow_render_time)
