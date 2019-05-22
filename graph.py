import plonny
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from layergrid import LayerGrid
from plonny import GraphParam
from plonny import add_layer_name
import copy

sns.set()

def defineFigure():
    """Setup plot"""
    fig = plt.figure(figsize=(10,10))
    axes = fig.add_subplot(111)
    plt.axis('off')
    plt.subplots_adjust(left=0.05, right=0.95, top=1.2, bottom=.3)
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    return fig, axes

class Graph(object):
    """Controls a structured grid for a graph."""

    def __init__(self, graph=None):
        self.lgrid = LayerGrid()
        if(isinstance(graph, Graph)):
            self.lgrid = graph.lgrid
        if(isinstance(graph, LayerGrid)):
            self.lgrid = graph

    def copyDuplicates(self):
        """Copy duplicate layers in a column and return pos in grid."""
        to_merge = []
        grid = self.lgrid
        for rowIdx, _ in enumerate(grid.rows()):
            for colIdx, _ in enumerate(grid.cols()):
                layer = grid.get(rowIdx, colIdx)
                if(layer is None):
                    continue

                if(rowIdx > 0 and layer == grid.get(rowIdx - 1, colIdx)):
                    print(layer.name +  "is equal to " + grid.get(rowIdx - 1, colIdx).name)
                    grid.set(rowIdx, colIdx, copy.deepcopy(layer))
                    to_merge.append((rowIdx, colIdx))
        return to_merge


    def graphshow(self, title="Neural Network"):
        f, ax = defineFigure()
        grid = self.lgrid

        # Copy duplicates to set individual locations, and merge these later on
        to_merge = self.copyDuplicates()

        # calculate dimensions for the rectilinear grid
        col_widths   = [max([layer.shape[0] for layer in col]) for col in grid.cols()]
        # print("rows:", [max([layer.shape[1] for layer in row]) for row in grid.rows()])
        for row in grid.rows():
            for layer in row:
                print("layer:", layer.name, layer.shape)
        row_heights  = [max([layer.shape[1] for layer in row]) for row in grid.rows()]

        # Solve when no depth in row
        # depths = [layer.shape[2] for layer in row if len(layer.shape) > 2]
        # if depths == []: depths = [1]
        maxD = max([max([layer.shape[2] for layer in row if len(layer.shape) > 2]) for row in grid.rows()])
        print(maxD)

        # calculate layer screen dimensions based on grid
        for row in grid.rows():
            for layer in row:
                layer.setDimensions(col_widths, row_heights, maxD)

        # Screen space version of rectilinear grid
        col_widths = [max([layer.width for layer in col]) for col in grid.cols()]
        row_heights =[max([layer.height for layer in row]) for row in grid.rows()]

        # starting point for drawing
        xy = {  'x':.5 * (1 - np.sum(col_widths) - GraphParam.spacing * len(col_widths)),
                'y':1} #use text spacing# + GraphParam.label_reserve * len(row_heights)#.5 * (1 + np.sum(row_heights))
        label_y_pool = []
        label_min = xy['y']

        # iterate gridpositions, rowwise
        for rowIdx, _ in enumerate(grid.rows()):
            for colIdx, _ in enumerate(grid.cols()):
                # Left cube is cutoff!
                # Skip empty gridcells
                layer = grid.get(rowIdx, colIdx)
                if(layer is None):
                    continue

                # Set layer locations
                layer.xy        = dict(xy)
                rh = list(row_heights[1:])
                layer.xy['x']   += np.sum(col_widths[:colIdx]) + GraphParam.spacing * colIdx
                layer.xy['y']    = label_min - row_heights[rowIdx]
                layer.txt_height = label_min - row_heights[rowIdx]
                layer.xy['y']   += .5 * (row_heights[rowIdx] - layer.height) #center tensors


                # Temporarily draw labels to get their dimensions
                txt, _, y1, _, _ = add_layer_name(layer)
                label_y_pool += [y1]
                txt.remove()

            label_min = np.min(label_y_pool)

        # merge duplicate layers
        for rowIdx, colIdx in to_merge:
            l1 = grid.get(rowIdx - 1, colIdx)
            l2 = grid.get(rowIdx, colIdx)

            l1.xy['y']      = (l1.xy['y'] + l2.xy['y']) / 2.
            l1.txt_height   = (l1.txt_height + l2.txt_height) / 2.
            print("merging", rowIdx, colIdx)
            grid.set(rowIdx, colIdx, None)

        # Center the figure after having calculated total height
        miny = label_min
        maxy = xy['y'] - row_heights[rowIdx]
        diff = .5 - .5*(maxy - miny)
        for row in grid.rows():
            for layer in row:
                layer.xy['y'] -= diff
                layer.txt_height -= diff

        # set titles locations and plot
        maxheight = .5 * (1 + np.sum(row_heights))
        GraphParam.titleheight = xy['y'] + 3*GraphParam.spacing - diff#maxheight + 4*GraphParam.txt_margin
        t = plt.text(0.5, GraphParam.titleheight, title, ha='center', va="bottom", size=16)

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
