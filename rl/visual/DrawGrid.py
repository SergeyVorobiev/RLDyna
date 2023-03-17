class DrawGrid(object):

    def __init__(self, grid: []):
        self.grid: [] = grid

    def draw(self):
        for row in self.grid:
            for cell in row:
                cell.draw()

    def update_text(self):
        for row in self.grid:
            for cell in row:
                cell.update_text()