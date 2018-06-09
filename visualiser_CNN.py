import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import numpy as np
import seaborn as sns
sns.set()

def show_basiclayer(self, axes, xy, width, height, depth, shp, param):
    """Plot this layer"""
    # Annotate dimensions and layer
    plt.text(xy['x']+.5*width, xy['y']-param['txt_margin'], shp[0], rotation=0)
    plt.text(xy['x'], xy['y'] + .5*height, shp[1], rotation=90)
    if len(shp) > 2:
        plt.text(xy['x'], xy['y'] + height + param['txt_margin'], depth, rotation=45)
    plt.text(xy['x'], param['txtheight'], type(self).__name__, rotation=90)

    # Draw shape
    for z in reversed(range(depth)):
        # Color calculation
        d = (1 - z/depth) * 0.8 + 0.2
        color = tuple([.4* d, .5* d, 1* d])

        # draw
        xyd = {key: xy[key] + param['depth_spacing'] * (z/depth) for key in xy.keys()}
        rect = patches.Rectangle(   tuple(xyd.values()),
                                    width,
                                    height,
                                    color=color,
                                    linewidth=1)#, edgecolor='r', facecolor='none'
        axes.add_patch(rect)

class Input(object):
    def __init__(self, shape):
        self.shape = shape
        self.graph = [self]

    show = show_basiclayer

class Layer(object):
    def __init__(self, layer):
        if(not isinstance(layer, Layer) and not isinstance(layer, Input)):
            raise AssertionError("A new layer was requested attached to neither a layer or an input.", str(layer))
        self.graph = layer.graph + [self]
        self.shape = None

        self.param = {
            'txtheight'       : 1.1,                #where to draw text. plot top=1
            'txt_margin'      : 0.05,               #offsets for shape annotators
            'spacing'         : 0.05,               #horizontal space between shapes
            'maxHeight'       : 0.666               #no higher than 2/3
        }
        #how far deep layers go to back
        self.param['depth_spacing'] = .5*self.param['spacing']

    show = show_basiclayer

    def _width2w(self, width, graph=None):
        """Convert width in shape space to coordinate space width"""
        graph = self.graph if graph is None else graph

        spacings = self.param['spacing'] * (len(graph)-1)
        totalwidths = np.sum([layer.shape[0] for layer in graph])
        return width / totalwidths * (1 - spacings)

    def maxShape(self, graph=None):
        """Returns maximum width and height of all layer outputs"""
        graph = self.graph if graph is None else graph

        maxShape = [0,0]
        for layer in graph:
            for dim in range(2):
                if layer.shape[dim] > maxShape[dim]:
                    maxShape[dim] = layer.shape[dim]
        return {'w':maxShape[0], 'h':maxShape[1]}

    # def outputParam(self, max_shp=None):
    #     """Return startx, starty, width, height & depth of this layers output"""
    #     max_shp = maxShape() if max_shp is None else max_shp
    #
    #     width = self._width2w(layer.shape[0])
    #     height = layer.shape[1] / max_shp['h'] * self.maxHeight
    #     depth = shp[2] if len(shp) > 2 else 1
    #
    #     xy = {}
    #     xy['y'] = .5 - .5 * height
    #     xy['x'] += width + self.spacing
    #     outputshapes = [layer.outputParam(max_shp) for layer in self.graph]
    #     xy['x'] = np.sum([width for (_, width, _, _) in ])
    #     return (xy, width, height, depth)

    def output_shape(self, graph=None):
        maxShape = self.maxShape(graph)

        width = self._width2w(shp[0])
        height = shp[1] / maxShape['h'] * self.param['maxHeight']
        depth = shp[2] if len(shp) > 2 else 1
        return (width, height, depth)




    def graphshow(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.axis('off')

        # get max shapes
        maxShape = self.maxShape()

        # Iterate layers, plotting their output shapes
        xy = {'x':0,'y':0}
        for ctr, layer in enumerate(self.graph):
            shp = layer.shape
            width = self._width2w(shp[0])
            height = shp[1] / maxShape['h'] * self.param['maxHeight']
            depth = shp[2] if len(shp) > 2 else 1
            xy['y'] = .5 - .5 * height

            layer.show(ax, xy, width, height, depth, shp, self.param)

            # Update x position
            xy['x'] += width + self.param['spacing']

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
        self.shape = (1, neurons)

class CTC(Layer):
    def __init__(self, layer, shape):
        Layer.__init__(self, layer);
        self.shape = shape

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
    conv_11 = Conv2D(resh_10, (8, 28))

    # conv_11.show()

    fccc_12 = FC(conv_11, 8)
    fccc_12.graphshow()
