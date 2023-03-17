from rl.visual.Cell import Cell


def update_policy_colors(cells, values, cell_value_func):
    uniques = len(set(values))
    if uniques == 0:
        return
    step = 255 / uniques
    r = 50
    value = None
    for cell in cells:
        cell: Cell = cell
        cell_v = cell_value_func(cell)
        if value != cell_v:
            r += step
            value = cell_v
        color = int(min(r + 10, 255))
        if not cell.is_target:
            cell.set_default_color(color, color, color)
            cell.reset_color()


def update_policy_colors_cells(cells, cell_value_func):
    cells, values = get_sorted_by_value_cells(cells, cell_value_func)
    update_policy_colors(cells, values, cell_value_func)


def update_policy_colors_grid(grid, cell_value_func):
    cells, values = get_sorted_by_value_grid(grid, cell_value_func)
    update_policy_colors(cells, values, cell_value_func)


def get_sorted_by_value_cells(cells, cell_value_func):
    cells_t = []
    values = []
    for cell in cells:
        cell: Cell = cell
        value = cell_value_func(cell)
        if value is None:
            continue
        cells_t.append(cell)
        values.append(value)
    return sorted(cells_t, key=lambda cell1: cell_value_func(cell1), reverse=False), values


def get_sorted_by_value_grid(grid, cell_value_func):
    cells = []
    values = []
    for row in grid:
        for cell in row:
            cell: Cell = cell
            value = cell_value_func(cell)
            if value is None:
                continue
            cells.append(cell)
            values.append(value)
    return sorted(cells, key=lambda cell1: cell_value_func(cell1), reverse=False), values
