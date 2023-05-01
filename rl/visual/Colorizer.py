from rl.visual.Cell import Cell


class Colorizer:
    rc = 0
    gc = 0
    bc = 0

    @staticmethod
    def update_value_colors(cells, values, cell_value_func):
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
                r_color = Colorizer.clip(color + Colorizer.rc)
                g_color = Colorizer.clip(color + Colorizer.gc)
                b_color = Colorizer.clip(color + Colorizer.bc)
                cell.set_default_color(r_color, g_color, b_color)
                cell.reset_color()

    @staticmethod
    def clip(value):
        if value > 255:
            value = min(value - 255, 255)
        elif value < 0:
            value = min(255 + value, 255)
        return value

    @staticmethod
    def setup_colors(r, g, b):
        Colorizer.rc = r
        Colorizer.gc = g
        Colorizer.bc = b

    @staticmethod
    def update_value_colors_cells(cells, cell_value_func):
        cells, values = Colorizer.get_sorted_by_value_cells(cells, cell_value_func)
        Colorizer.update_value_colors(cells, values, cell_value_func)

    @staticmethod
    def update_value_colors_grid(grid, cell_value_func):
        cells, values = Colorizer.get_sorted_by_value_grid(grid, cell_value_func)
        Colorizer.update_value_colors(cells, values, cell_value_func)

    @staticmethod
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

    @staticmethod
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
