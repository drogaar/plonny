import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import numpy as np
import seaborn as sns

sns.set()

def annotate_tensor(self, param, showdims=(True, True, True)):
    """Annotates an output shape with its dimensions."""
    fontsize = 8
    # width
    if showdims[0]:
        plt.text(   self.xy['x']+.5*self.width,
                    self.xy['y']-param['txt_margin'],
                    # self.xy['y'],
                    self.shape[0], rotation=0, size=fontsize)

    # height
    if showdims[1]:
        t = plt.text(   self.xy['x'],
                    self.xy['y'] + .5*self.height,
                    self.shape[1], rotation=90, size=fontsize)
        # bb = t.get_window_extent(renderer=r)height
        # param['txtheight'] = 0.5 - t.get_window_extent(renderer=plt.gcf().canvas.get_renderer()).height
        # print(param['txtheight'])

    # depth
    if len(self.shape) > 2 and showdims[2]:
        plt.text(   self.xy['x'],
                    self.xy['y'] + self.height + param['txt_margin'],
                    # self.xy['y'] + self.height,
                    self.shape[2], rotation=45, size=fontsize)

def add_layer_name(self, param):
    """Adds the layers name to plot."""
    # plt.text(self.xy['x'], 0.5 - 2*param['txt_margin'], type(self).__name__, rotation=90)
    plt.text(self.xy['x'], param['txtheight'], type(self).__name__, rotation=90)

def draw_tensor(self, axes, param):
    """Draw a three dimenionsal cube depending on given layers output shape"""
    shape2 = self.shape[2] if len(self.shape)>2 else 1
    for z in reversed(range(shape2)):
        # Color calculation
        d = (1 - z/shape2) * 0.8 + 0.2
        color = tuple([.4* d, .5* d, 1* d])

        # draw output shape
        # print((self.depth))
        xyd = {key: self.xy[key] + param['depth_spacing'] * (z/shape2*self.depth) for key in self.xy.keys()}
        rect = patches.Rectangle(   tuple(xyd.values()),
                                    self.width,
                                    self.height,
                                    color=color,
                                    linewidth=1)#, edgecolor='r', facecolor='none'
        axes.add_patch(rect)


def show_basiclayer(self, axes, param, inputs=[]):
    """Default plot for Input and Layer"""
    draw_tensor(self, axes, param)
    annotate_tensor(self, param)
    add_layer_name(self, param)

class Input(object):
    def __init__(self, shape):
        self.shape = shape
        self.graph = [self]

    show = show_basiclayer

class Layer(object):
    def __init__(self, layer, shape):
        if(not isinstance(layer, Layer) and not isinstance(layer, Input)):
            raise AssertionError("A new layer was requested attached to neither a layer or an input.", str(layer))
        self.graph = layer.graph + [self]
        self.shape = shape

        self.param = {
            'txtheight'       : .8,                #where to draw text. plot top=1
            'txt_margin'      : 0.025,               #offsets for shape annotators
            'spacing'         : 0.025,               #horizontal space between shapes
        }
        #how far deep layers go to back
        self.param['depth_spacing'] = .7*self.param['spacing']

    show = show_basiclayer

    def _width2w(self, width, graph=None):
        """Convert width in shape space to coordinate space width"""
        graph = self.graph if graph is None else graph

        spacings = self.param['spacing'] * (len(graph)-1)
        totalwidths = np.sum([layer.shape[0] for layer in graph])
        return width / totalwidths * (1 - spacings)

    def calcMaxShape(self, graph=None):
        """Returns maximum dimensions of all layer outputs"""
        graph = self.graph if graph is None else graph

        maxShape = [0,0,0]
        for layer in graph:
            for dim in range(len(layer.shape)):
                if layer.shape[dim] > maxShape[dim]:
                    maxShape[dim] = layer.shape[dim]
        return {'w':maxShape[0], 'h':maxShape[1], 'd':maxShape[2]}

    def output_shape(self, graph=None):
        """Returns output shape of this layer"""
        maxShape = self.maxShape(graph)

        width = self._width2w(shp[0])
        height = shp[1] / maxShape['h'] * self.param['maxHeight']
        depth = shp[2] if len(shp) > 2 else 1
        return (width, height, depth)

    def graphshow(self, title=None):
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)
        plt.axis('off')
        # plt.tight_layout()
        plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0.05)
        plt.ylim(0, 1)

        # get max shapes
        maxShape = self.calcMaxShape()
        # self.param['txtheight'] = maxShape['h'] / maxShape['w'] * self._width2w(maxShape['w']) + 4*self.param['txt_margin']
        self.param['txtheight'] = 0.5 - .5 * maxShape['h'] / maxShape['w'] * self._width2w(maxShape['w']) - 2*self.param['txt_margin']
        self.param['titleheight'] = 0.5 + .5 * maxShape['h'] / maxShape['w'] * self._width2w(maxShape['w']) + 4*self.param['txt_margin']

        # show title
        if(title is not None):
            plt.text(0.5, self.param['titleheight'], title, horizontalalignment='center')

        # set layer plotting properties
        xy = {'x':0,'y':0}
        for layer in self.graph:
            shp = layer.shape
            layer.width     = self._width2w(shp[0])
            # layer.height    = shp[1] / maxShape['h'] * (shp[1] * layer.width / shp[0])
            layer.height    = shp[1] / shp[0] * layer.width
            layer.depth     = shp[2] / maxShape['d'] if len(shp) > 2 else 0
            layer.maxShape  = maxShape

            # xy['y'] = .5 * (maxShape['h'] / maxShape['w'] * layer.width) - .25 * layer.height + self.param['txt_margin']
            # xy['y'] = .5 * (maxShape['h'] / maxShape['w'] * self._width2w(maxShape['w'])) - .5 * layer.height + self.param['txt_margin']
            xy['y'] = .5 - .5 * layer.height + self.param['txt_margin']

            layer.xy        = dict(xy)

            # Update x position
            xy['x'] += layer.width + self.param['spacing']

        # Iterate layers, plotting their output shapes
        for ctr, layer in enumerate(self.graph):
            if(ctr>0):
                layer.show(ax, self.param, [self.graph[ctr-1]])
            else:
                layer.show(ax, self.param)

        plt.show()

class Pool(Layer):
    def __init__(self, layer, shape):
        Layer.__init__(self, layer, shape)

class Reshape(Layer):
    def __init__(self, layer, shape):
        Layer.__init__(self, layer, shape)

class CTC(Layer):
    """Connectionist Temporal Classification"""
    def __init__(self, layer, shape):
        Layer.__init__(self, layer, shape)

class Upsample(Layer):
    def __init__(self, layer, shape):
        Layer.__init__(self, layer, shape)

class Dropout(Layer):
    def __init__(self, layer, shape):
        Layer.__init__(self, layer, shape)

class Concat(Layer):
    nConcatsUsed = 0
    concat_spacing = 0

    def __init__(self, layer, extra_input_layers):
        shape = list(layer.shape)
        for input_layer in extra_input_layers:
            shape[2] += list(input_layer.shape)[2] if len(input_layer.shape) > 2 else 0

        Layer.__init__(self, layer, shape)
        self.input_layers = extra_input_layers

        Concat.nConcatsUsed += 1
        self.concat_spacing = 0.025 / Concat.nConcatsUsed #txt margin?

    def show(self, axes, param, inputs=[]):
        """Plot this layer"""
        draw_tensor(self, axes, param)
        annotate_tensor(self, param)
        add_layer_name(self, param)

        # connect input layers to this layer
        # down = list(kernel_pos)
        # down[0] = down[0] + self.kernel_rel[0]
        # up = list(down)
        # up[1] += self.kernel_rel[1]
        # dest = dict(self.xy)
        # dest['y'] += self.height
        yPos = self.xy['y'] - param['txt_margin'] - Concat.nConcatsUsed * self.concat_spacing
        for idx, layer in enumerate(self.input_layers):
            # downwards spacing
            plt.plot([layer.xy['x'], layer.xy['x']], [layer.xy['y'], yPos], linewidth=1, color=(0,0,0))
            plt.plot([self.xy['x'], self.xy['x']], [self.xy['y'], yPos], linewidth=1, color=(0,0,0))
            # horizontal connection
            plt.plot([layer.xy['x'], self.xy['x']], [yPos, yPos], linewidth=1, color=(0,0,0))

class FullyConnected(Layer):
    def __init__(self, layer, neurons):
        shape = (1, neurons)
        Layer.__init__(self, layer, shape)

    def show(self, axes, param, inputs=[]):
        """Plot this layer"""
        annotate_tensor(self, param, (False,True,False))
        add_layer_name(self, param)

        n_neurons = self.shape[1]
        for neuron in range(n_neurons):
            # draw neurons
            color = (.4, .5, 1)
            xyn = dict(self.xy)
            xyn['y'] += neuron * self.height / n_neurons
            circ = patches.Circle(      tuple(xyn.values()),
                                        radius = 0.1 * self.height / n_neurons,
                                        color=color,
                                        linewidth=1)
            axes.add_patch(circ)

            # draw connections
            color = (0, 0, 0)
            for layer in inputs:
                for vert in range(layer.shape[1]):
                    xy_dest = ( layer.xy['x'] + layer.width,
                                layer.xy['y'] + (vert/layer.shape[1]) * layer.height)
                    plt.plot([xyn['x'], xy_dest[0]], [xyn['y'], xy_dest[1]], linewidth=.1, alpha=0.3, color=color)

class Conv2D(Layer):
    """2D Convolution"""
    def __init__(self, layer, shape, kernel):
        Layer.__init__(self, layer, shape)
        self.kernel = kernel

    def show(self, axes, param, inputs=[]):
        """Default plot for Input and Layer"""
        draw_tensor(self, axes, param)
        annotate_tensor(self, param)
        add_layer_name(self, param)

        # draw kernel
        input = inputs[0]
        kerncol = (0, 0, 0)
        self.kernel_rel = (self.kernel[0] * input.width / input.shape[0], self.kernel[1] * input.height / input.shape[1])
        kernel_pos = (input.xy['x'] + input.width - self.kernel_rel[0], input.xy['y'] + input.height - self.kernel_rel[1])
        rect = patches.Rectangle(   kernel_pos,
                                    self.kernel_rel[0],
                                    self.kernel_rel[1],
                                    color=kerncol,
                                    alpha=.6,
                                    linewidth=1)#, edgecolor='r', facecolor='none'
        axes.add_patch(rect)

        # draw kernel connections
        down = list(kernel_pos)
        down[0] = down[0] + self.kernel_rel[0]
        up = list(down)
        up[1] += self.kernel_rel[1]
        dest = dict(self.xy)
        dest['y'] += self.height
        plt.plot([down[0], dest['x']], [down[1], dest['y']], linewidth=.5, alpha=0.9, color=kerncol)
        plt.plot([up[0], dest['x']], [up[1], dest['y']], linewidth=.5, alpha=0.9, color=kerncol)
