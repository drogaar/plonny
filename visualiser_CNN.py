import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import seaborn as sns
sns.set()

class Input(object):
    def __init__(self, shape):
        self.shape = shape
        self.graph = [self]

class Layer(object):
    def __init__(self, layer):
        if(not isinstance(layer, Layer) and not isinstance(layer, Input)):
            raise AssertionError("A new layer was requested attached to neither a layer or an input.", str(layer))
        self.graph = layer.graph + [self]
        self.shape = None

    def show(self):
        fig = plt.figure()
        # fig.set_size_inches(8, 2.5)
        # margins?
        ax = fig.add_subplot(111)

        # get max shapes
        maxShape = [0,0]
        for layer in self.graph:
            shp = layer.shape
            for dim in range(2):
                if shp[dim] > maxShape[dim]:
                    maxShape[dim] = shp[dim]
        # for dim in range(2):
            # maxShape[dim] *= 8
        maxShape[0] *= 8
        maxShape[1] *= 1.5

        txtheight = 1
        xy = [0,0]
        spacing = 0.025
        plt.axis('off')
        for ctr, layer in enumerate(self.graph):

            shp = layer.shape
            # xy = (ctr / len(self.graph), 0)

            print(shp[0] / maxShape[0], shp[1] / maxShape[1])
            width = shp[0] / maxShape[0]
            height = shp[1] / maxShape[1]
            xy[1] = .5 - .5 * height

            # Draw shape
            plt.text(xy[0]+.5*width, xy[1]-0.05, shp[0], rotation=0)
            plt.text(xy[0], xy[1] + .5*height, shp[1], rotation=90)
            rect = patches.Rectangle(   tuple(xy),
                                        width,
                                        height,
                                        linewidth=1)#, edgecolor='r', facecolor='none'
            ax.add_patch(rect)

            # Annotate
            plt.text(xy[0], txtheight, type(layer).__name__, rotation=90)

            xy[0] += shp[0] / maxShape[0] + spacing
            # ax.add_patch(patches.Rectangle((0,0), 0.5, 0.5, linewidth=1))
            # fig.patches.extend([rect])

        plt.show()

class Conv2D(Layer):
    def __init__(self, layer, shape):
        Layer.__init__(self, layer);
        self.shape = shape

class Pool(Layer):
    def __init__(self, layer, shape):
        Layer.__init__(self, layer);
        self.shape = shape

class Reshape(Layer):
    def __init__(self, layer, shape):
        Layer.__init__(self, layer);
        self.shape = shape

class FC(Layer):
    def __init__(self, layer, neurons):
        Layer.__init__(self, layer);
        self.neurons = neurons

if __name__ == "__main__":
    # Define model by calling layers
    image_0 = Input((32, 128, 1))

    conv2_1 = Conv2D(image_0, (32, 128, 16))
    Pool_2  = Pool(conv2_1, (16, 64, 16))

    conv2_3 = Conv2D(Pool_2, (16, 64, 64))
    Pool_4  = Pool(conv2_3, (8, 32, 64))

    conv2_5 = Conv2D(Pool_4, (8, 32, 64))
    Pool_6  = Pool(conv2_5, (4, 16, 64))

    conv2_7 = Conv2D(Pool_6, (4, 16, 64))
    conv2_8 = Conv2D(conv2_7, (4, 16, 16))
    conv2_9 = Conv2D(conv2_8, (4, 16, 7))

    resh_10 = Reshape(conv2_9, (8, 56))
    conv_11 = Conv2D(conv2_9, (8, 28))

    conv_11.show()
