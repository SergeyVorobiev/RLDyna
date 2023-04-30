import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import get_current_fig_manager


class PlotHelper:

    @staticmethod
    def plot(data, legend_name, axis, line_type):
        axis.clear()
        size = data.__len__()
        if size < 1:
            return
        aver = sum(data) / size
        x = np.linspace(1, size, size)
        y = np.linspace(aver, aver, size)
        line, = axis.plot(x, data, line_type)
        axis.plot(x, y, "g--", markersize=8)
        line.set_label(legend_name)
        axis.legend()

    @staticmethod
    def build_subplots(sub_plots_n, win_pos_x, win_pos_y, width=8, height=9):
        fig, axis = plt.subplots(sub_plots_n)
        fig.set_figwidth(width)
        fig.set_figheight(height)
        mgr = get_current_fig_manager()
        position = f"+{win_pos_x}+{win_pos_y}"
        mgr.window.wm_geometry(position)
        plt.show(block=False)
        return fig, axis

    @staticmethod
    def build_3d(win_pos_x, win_pos_y, width=8, height=8):
        fig = plt.figure()
        fig.set_figwidth(width)
        fig.set_figheight(height)
        ax = plt.axes(projection='3d')
        mgr = get_current_fig_manager()
        position = f"+{win_pos_x}+{win_pos_y}"
        mgr.window.wm_geometry(position)
        plt.show(block=False)
        return fig, ax

    @staticmethod
    def draw_all():
        plt.draw()
        plt.pause(0.01)
