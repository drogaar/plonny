import numpy as np

class LayerGrid(object):
    """Define a grid on which to place layers for a structured view."""

    def __init__(self):
        self.grid = [[None]]

    def __str__(self):
        format = lambda x: x.name if x is not None else "None"
        rows = ["{}\n".format([format(layer) for layer in row]) for row in self.grid]
        return "".join(rows)

    def set(self, rowIdx, colIdx, layer):
        # enlarge grid as necessary
        self.grid += [[None] for _ in range((rowIdx + 1) - len(self.grid))]
        maxCols = np.max([len(row) for row in self.grid])
        for row in self.grid:
            row += [None for _ in range(max((colIdx + 1), maxCols) - len(row))]

        self.grid[rowIdx][colIdx] = layer

    def exists(self, row, col):
        try:
            return self.grid[row][col] is not None
        except IndexError:
            return False

        return False

    def get(self, row, col):
        if(self.exists(row, col)):
            return self.grid[row][col]
        return None

    def find(self, obj):
        for rowIdx, row in enumerate(self.grid):
            for colIdx, layer in enumerate(row):
                if(obj == layer):
                    return (rowIdx, colIdx)

    def get_row(self, rowIdx):
        return [layer for layer in self.grid[rowIdx] if layer is not None]

    def get_col(self, colIdx):
        column = [row[colIdx] for row in self.grid]
        return [layer for layer in column if layer is not None]

    def rows(self):
        return [self.get_row(rowIdx) for rowIdx in range(len(self.grid))]
    def cols(self):
        return [self.get_col(colIdx) for colIdx in range(len(self.grid[0]))]

if __name__ == "__main__":
    lg = LayerGrid()
    lg.set(0, 2, "test")
    lg.set(0, 3, "test")
    lg.set(0, 4, "test")
    lg.set(0, 5, "test")
    lg.set(3, 1, "test")
    lg.set(6, 0, "test")
    lg.set(8, 7, "test")
    print(lg)
    print(lg.get_row(6))
    print(lg.get_col(0))
