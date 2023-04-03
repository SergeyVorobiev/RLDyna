from graphics import Text, Point

from rl.environments.fl.BasicGridEnv import BasicGridEnv, move_v, start_v, stay_v, StateType


class FrozenLakeEnv(BasicGridEnv):

    def __init__(self, rewards, state_type: StateType=None, screen_width=None, screen_height=None, cell_width=120,
                 cell_height=80, begin_from_start_if_get_in_hole: bool = False, grid_map=None, color_map=None):
        if_hole = start_v if begin_from_start_if_get_in_hole else stay_v
        transition_map = [[rewards[0], move_v], [rewards[1], if_hole], [rewards[2], stay_v], [rewards[3], start_v],
                          [-1, move_v]]
        self._name = 'FrozenLake'
        if color_map is None:
            color_map = [[0, 255, 255], [0, 0, 255], [0, 0, 0], [255, 255, 0], [0, 255, 0], [255, 0, 0]]
        if grid_map is None:
            grid_map = [[0, 0, 0, 1, 0, 0, 1, 3],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 1, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 1, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [4, 0, 0, 1, 0, 0, 0, 0]]
        super().__init__(grid_map, state_type, transition_map, color_map)
        self._left_padding = 50
        self._right_padding = 50
        if screen_width is None:
            self._screen_width = self._left_padding + self._right_padding + cell_width * self.get_x()
        else:
            self._screen_width = screen_width
        if screen_height is None:
            self._screen_height = self._left_padding + self._right_padding + cell_height * self.get_y()
        else:
            self._screen_height = screen_height
        self._cell_width = cell_width
        self._cell_height = cell_height
        self._header = Text(Point(300, 20), "")
        self._header.setSize(14)

    def __draw_header(self, create=False):
        if self.win is not None:
            text = "Episode: " + str(self._episodes) + " Max score: " + str(self._max_score) + " Steps: " + str(
                self._max_steps) + " Ep steps: " + str(self._episode_steps)
            self._header.setText(text)
            if create:
                self._header.draw(self.win)

    def game_done(self):
        pass

    def get_max_score(self):
        return self._max_score

    def set_name(self, name):
        self._name = name

    def after_render(self):
        self.__draw_header()

    def get_world_params(self):
        return self._name, self._screen_width, self._screen_height, self._left_padding, self._right_padding, \
               self._cell_width, self._cell_height

    def after_world_created(self):
        self.__draw_header(create=True)
