import numpy as np
from graphics import Rectangle, GraphWin, Text, Point, color_rgb

from rl.visual.Visual import Visual
from rl.mdp.Action import Action


class Cell(Visual):

    def __init__(self, value, color, draw_text, rect: Rectangle, win, pos_x, pos_y):
        super().__init__()
        self.rect = rect
        self.marked = False
        self.draw_text = draw_text
        self.color = color
        self.win: GraphWin = win
        self.default_color = color
        self.state_name = ""
        self.max_action = 0
        self._text_size = 8
        self.max_q = 0
        self.max_action_name = ""
        self.v = 0.0
        self.qs = []
        self.a_labels = []
        self.optimal_q = 0.0
        self.actions: [Action] = []
        self.optimal_actions_count = 0
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.is_target = False
        sp = self.rect.p1
        ep = self.rect.p2
        self.round_number = 1
        self.label = Text(Point((ep.x - sp.x) / 2 + sp.x, (ep.y - sp.y) / 2 + sp.y),
                          str(round(value, self.round_number)))

    def reset_color(self):
        self.color = self.default_color
        self.rect.setFill(self.color)

    def reset_data(self):
        self.v = 0
        self.qs = []
        self.max_q = 0
        self.actions = []
        self.optimal_actions_count = 0
        self.optimal_q = 0

    def set_default_color(self, r, g, b):
        self.default_color = color_rgb(r, g, b)

    def draw(self):
        self.rect.setFill(self.color)
        self.rect.draw(self.win)
        if self.draw_text:
            self.label.setText(self.state_name)
            self.label.setSize(self._text_size)
            self.label.draw(self.win)

    def set_color_and_update(self, r, g, b):
        self.color = color_rgb(r, g, b)
        self.update_color()

    def update_color(self):
        self.rect.setFill(self.color)

    def __get_str(self, head, value):
        return head + str(round(value, self.round_number)) + "\n"

    def update_qstext(self, short=False):
        if self.draw_text:
            text = self.state_name
            tqs = []
            if self.qs.__len__() > 0:
                if not short:
                    for q in self.qs:
                        tqs.append(round(q, self.round_number))
                    text += "\n" + "q: " + str(tqs)
                text += "\n" + str(round(self.max_q, self.round_number)) + " " + self.a_labels[self.max_action]
            self.label.setText(text)

    def update_v_text(self):
        if self.draw_text:
            text = self.state_name
            if self.v is None:
                self.v = 0
            text += "\n" + "v: " + str(round(self.v, self.round_number))
            self.label.setText(text)

    def update_opt_q_text(self):
        if self.draw_text:
            text = self.state_name
            if self.optimal_q is None:
                self.optimal_q = 0
            text += "\n" + "q: " + str(round(self.optimal_q, self.round_number))
            self.label.setText(text)

    def update_text(self):
        if self.draw_text:
            text = self.state_name
            if self.v is None:
                self.v = 0
            text += "\n" + "v: " + str(round(self.v, self.round_number))
            # opt_q = self.optimal_q
            # if opt_q is not None:
            #    opt_q = round(self.optimal_q, self.round_number)
            # text += "\n" + "opt_q: " + str(opt_q)
            # text += "\n" + "opt a: " + str(self.get_optimal_action_names())
            tqs = []

            act_names = []
            for action in self.actions:
                act_names.append(action.name)
                tqs.append(round(action.q, 2))
            text += "\n" + "q:" + str(tqs)  # str(round(self.optimal_q, self.round_number))
            text += "\n" + "ma: " + str(act_names)
            self.label.setText(text)

    def get_max_q(self):
        if self.qs.__len__() == 0:
            return None
        return self.qs[self.max_action]

    def set_q_values(self, q_values: [float]):
        self.qs = q_values
        self.max_q = np.max(q_values)
        self.max_action = np.argmax(self.qs)

    def set_u_value(self, u_value):
        self.v = u_value

    def place_text(self, text):
        self.label.setText(text)

    def set_actions(self, actions: [Action]):
        self.actions = actions

    def set_optimal_actions_count(self, optimal_actions_count: int):
        self.optimal_actions_count = optimal_actions_count

    def set_state_v(self, v: float):
        self.v = v

    def set_state_name(self, state_name: str):
        self.state_name = state_name

    def set_optimal_q(self, optimal_q: float):
        self.optimal_q = optimal_q

    def get_qs(self):
        result = []
        for i in range(self.optimal_actions_count):
            action: Action = self.actions[i]
            result.append(round(action.q, self.round_number))
        return result

    def get_optimal_action_names(self):
        result = []
        for i in range(self.optimal_actions_count):
            action: Action = self.actions[i]
            result.append(action.name)
        return result
