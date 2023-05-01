from time import sleep
from graphics import *

from rl.bonus.gridworld.GridWorld import Player, GridWorld
from rl.visual.Colorizer import Colorizer


def build_world():
    win1 = GraphWin('Face', 1800, 950)  # give title and dimensions
    world1 = GridWorld(win=win1,
                       width=19,
                       height=10,
                       epsilon=0.1,
                       spots_count=2,
                       draw_text=True,
                       spot_reward=5,
                       transition_reward=-1,
                       out_bound_reward=-1,
                       out_bound_value=0,
                       start_grid_point=Point(20, 20),
                       cell_size=90)
    world1.reset_grid()
    world1.draw()
    return world1, win1


def run():
    world, win = build_world()
    Colorizer.setup_colors(150, 50, -50)
    player = Player(world.get_cell(0, 0), world.width, world.height)
    player.update_position(world)
    print("Click left mouse button to start")
    win.getMouse()
    world.evaluate_expected_values()
    world.update_text()
    Colorizer.update_value_colors_grid(world.grid, lambda cell: cell.v)
    player.update()

    print("Sleep for a second")
    sleep(1)

    while True:
        player.move_player_by_policy(world)
        if player.cell.is_target:

            print("Sleep for two seconds")
            sleep(2)

            world.reset_grid()
            player.update_position(world)
            world.evaluate_expected_values()
            world.update_text()
            Colorizer.update_value_colors_grid(world.grid, lambda cell: cell.v)
            player.update()

            print("Sleep for two seconds")
            sleep(1)

            # win.getMouse()
    # win.close()


if __name__ == '__main__':
    run()
