import plonny
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from layergrid import LayerGrid
from plonny import GraphParam
sns.set()

def defineFigure():
    """Setup plot"""
    fig = plt.figure(figsize=(10,10))
    axes = fig.add_subplot(111)
    plt.axis('off')
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    return axes

class Graph(object):
    """Controls a structured grid for a graph."""

    def __init__(self, graph=None):
        self.lgrid = LayerGrid()
        if(isinstance(graph, Graph)):
            self.lgrid = graph.lgrid
        if(isinstance(graph, LayerGrid)):
            self.lgrid = graph

    def graphshow(self, title="Neural Network"):
        ax = defineFigure()
        grid = self.lgrid

        # calculate dimensions for the rectilinear grid
        col_widths   = [max([layer.shape[0] for layer in col]) for col in grid.cols()]
        row_heights  = [max([layer.shape[1] for layer in row]) for row in grid.rows()]

        maxD = max([max([layer.shape[2] for layer in row if len(layer.shape) > 2]) for row in grid.rows()])

        # calculate layer screen dimensions based on grid
        for row in grid.rows():
            for layer in row:
                layer.setDimensions(col_widths, row_heights, maxD)

        # Screen space version of rectilinear grid
        col_widths = [max([layer.width for layer in col]) for col in grid.cols()]
        row_heights =[max([layer.height for layer in row]) for row in grid.rows()]

        # starting point for drawing
        xy = {  'x':.5 * (1 - np.sum(col_widths) - GraphParam.spacing * (len(col_widths))),
                'y':.5 * (1 + np.sum(row_heights))} #use text spacing

        # iterate gridpositions
        for rowIdx, _ in enumerate(grid.rows()):
            for colIdx, _ in enumerate(grid.cols()):
                # Skip empty gridcells
                layer = grid.get(rowIdx, colIdx)
                if(layer is None):
                    continue

                # Set layer locations
                layer.xy        = dict(xy)
                layer.xy['x']   += np.sum(col_widths[:colIdx]) + GraphParam.spacing * colIdx
                layer.xy['y']   -= np.sum(row_heights[:rowIdx]) + GraphParam.label_reserve * rowIdx
                layer.xy['y']   -= .5 * layer.height + GraphParam.txt_margin
                layer.txt_height = xy['y'] - np.sum(row_heights[:rowIdx]) - GraphParam.label_reserve * (rowIdx + 1)
                #Use text spacing

        # set titles locations and plot
        maxheight = .5 * (1 + np.sum(row_heights))
        GraphParam.titleheight = maxheight + 4*GraphParam.txt_margin
        plt.text(0.5, GraphParam.titleheight, title, horizontalalignment='center')
        GraphParam.txt_height = maxheight - 2*GraphParam.txt_margin

        # Iterate layers, plotting their output shapes
        for row in grid.rows():
            for layer in row:
                layer.show(ax, layer.inbound)

        plt.show()

if __name__ == "__main__":
    l0 = plonny.Input((32, 32, 1))
    l02 = plonny.Input((32, 32, 1))
    l1 = plonny.Conv2D([l0, l02], (16, 16, 1), (3,3))
    l2 = plonny.Conv2D([l1], (32, 32, 1), (3,3))
    l3 = plonny.Conv2D([l2], (32, 32, 1), (3,3))
    l4 = plonny.Conv2D([l3], (32, 32, 1), (3,3))
    l5 = plonny.Conv2D([l4], (32, 32, 1), (3,3))

    l22 = plonny.Conv2D([l1], (32, 32, 1), (3,3))

    lg = LayerGrid()
    lg.set(0, 0, l0)
    lg.set(1, 0, l02)
    lg.set(0, 1, l1)
    lg.set(0, 2, l2)
    lg.set(1, 2, l22)
    # lg.set(0, 2, l2)
    # lg.set(0, 3, l3)
    # lg.set(0, 4, l4)
    # lg.set(0, 5, l5)
    print(lg.grid)

    Graph(lg).graphshow()
