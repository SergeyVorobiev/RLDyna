from graphics import Point, Rectangle, color_rgb

from rl.visual.Cell import Cell


def build_grid(win, width, height, start_rect_x, start_rect_y, cell_size_x, cell_size_y):
    grid = []
    start_x = start_rect_x
    for x in range(width):
        grid.append([])
        start_y = start_rect_y
        for y in range(height):
            sp = Point(start_x, start_y)
            ep = Point(start_x + cell_size_x, start_y + cell_size_y)
            rect = Rectangle(sp, ep)
            grid[x].append(Cell(0, color_rgb(0, 240, 0), True, rect, win, x, y))
            start_y += cell_size_y
        start_x += cell_size_x
    return grid


def get_list(grid):
    result = []
    for column in grid:
        for value in column:
            result.append(value)
    return result
