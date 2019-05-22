import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import numpy as np
import seaborn as sns
import util
from tqdm import tqdm
import math
import cv2

sns.set()

def annotate_tensor(self, showdims=(True, True, True)):
    """Annotates an output shape with its dimensions."""
    def addtext(text, x_offset=0, y_offset=0, **kwargs):
        fontsize = 16
        util.add_text(  self.xy['x'] + x_offset,
                        self.xy['y'] + y_offset,
                        text, size=fontsize, **kwargs)

    if showdims[0]:                                 # width
        addtext(self.shape[0], x_offset = .5*self.width)#, y_offset = -1 * GraphParam.txt_margin
    if showdims[1]:                                 # height
        addtext(self.shape[1], y_offset = .5*self.height, rotation=90, ha='right')
    if showdims[2] and len(self.shape) > 2:         # depth
        # addtext(self.shape[2], y_offset = self.height, rotation=45, va='bottom')
        addtext(self.shape[2], y_offset = self.height + GraphParam.depth_spacing * self.depth, rotation=45, va='bottom')

def add_layer_name(self):
    """Adds the layers name to plot."""
    # txt, x1, y1, x2, y2 = util.add_text(self.xy['x'], self.txt_height, self.name, va='top', ha='left')#, rotation=90
    txt, x1, y1, x2, y2 = util.add_text(self.xy['x'], self.txt_height, self.name, va='top', ha='right', rotation=45, size=18)
    return (txt, x1, y1, x2, y2)

def draw_tensor(self, axes):
    """Draw a three dimenionsal cube depending on given layers output shape"""
    shape2 = self.shape[2] if len(self.shape)>2 else 1
    for z in reversed(range(shape2)):
        # Color calculation
        d = (1 - z/shape2) * 0.6 + 0.4
        color = tuple(np.array(GraphParam.tensorColor) * d)

        # draw output shape
        xyd = {key: self.xy[key] + GraphParam.depth_spacing * (z/shape2*self.depth) for key in self.xy.keys()}
        rect = patches.Rectangle(   tuple(xyd.values()),
                                    self.width,
                                    self.height,
                                    color=color,
                                    zorder=-1,
                                    linewidth=1)#, edgecolor='r', facecolor='none'
        axes.add_patch(rect)

    rect = patches.Rectangle(   tuple(xyd.values()),
                                self.width,
                                self.height,
                                edgecolor=GraphParam.lineColor,
                                fill=False,
                                zorder=-1,
                                linewidth=1)

    # try:
    #     if self.img is not None:
    #         xyd = {key: self.xy[key] for key in self.xy.keys()}
    #
    #         # def transf(x,y):
    #         #     xy_pixels = axes.transData.transform(np.vstack([x,y]).T)
    #         #     xpix, ypix = xy_pixels.T
    #         #
    #         #     # In matplotlib, 0,0 is the lower left corner, whereas it's usually the upper
    #         #     # right for most image software, so we'll flip the y-coords...
    #         #     width, height = plt.gcf().canvas.get_width_height()
    #         #     ypix = height - ypix
    #         #     return xpix, ypix
    #         #
    #         # from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    #         img = cv2.imread(self.img)
    #         # coords_pix = transf(self.width, self.height)
    #         # coords_pix = [int(round(coord[0])) for coord in coords_pix]
    #         # print("COORDS: ", tuple(coords_pix))
    #         # img = cv2.resize(img, tuple(coords_pix))
    #         axicon = plt.gcf().add_axes([xyd['x'], xyd['y'], self.width, self.height])
    #         axicon.imshow(img)#, interpolation='nearest'
    #         axicon.axis('off')
    #
    #         # oi = OffsetImage(img)
    #         # box = AnnotationBbox(oi, tuple(xyd.values()), frameon=False)
    #         # axes.add_artist(box)
    # except:
    #     pass

def show_basiclayer(self, axes, inputs=[]):
    """Default plot for Input and Layer"""
    draw_tensor(self, axes)
    annotate_tensor(self)
    add_layer_name(self)

def calcMaxShape(graph):
    """Returns maximum dimensions of all layer outputs"""
    maxShape = [0,0,0]
    for layer in graph:
        for dim in range(len(layer.shape)):
            if layer.shape[dim] > maxShape[dim]:
                maxShape[dim] = layer.shape[dim]
    return {'w':maxShape[0], 'h':maxShape[1], 'd':maxShape[2]}

def scale2Screen(property, shapes, spacing, screen_width = 1):
    """Convert shape space width or height to screen space dimension"""
    spacings = spacing * len(shapes)
    return property / np.sum(shapes) * (screen_width - spacings)

def setDimsBy(self, widths, heights, maxDepth):
    """Given a row and column of layers, set self's proper width and height."""

    # self.maxShape_h  = calcMaxShape(horizontal)
    # self.maxShape_v  = calcMaxShape(vertical)
    max_height = scale2Screen(self.shape[1], heights, GraphParam.label_reserve, 1) #use labelheight spacing
    max_full_width = self.shape[0] / self.shape[1] * max_height
    max_full_width = np.sum(widths) / self.shape[0] * max_full_width + GraphParam.spacing * len(widths)

    # self.width = scale2Screen(self.shape[0], widths, GraphParam.spacing)
    self.width = scale2Screen(self.shape[0], widths, GraphParam.spacing, min(1, max_full_width))
    self.height    = self.shape[1] / self.shape[0] * self.width
    self.depth     = self.shape[2] / maxDepth if len(self.shape) > 2 else 0
    # self.depth = 0
    # TODO: global max depth to be set.
    # TODO: dynamic layer height


class GraphParam:
    # tensorColor     =   (.4, .5, 1)
    tensorColor     =   (76/255, 114/255, 176/255)#(116/255, 137/255, 106/255)76, 114, 176
    lineColor       =   (0, 0, 0)

    # The plot space is defined as x in [0,1], y in [0,1]
    txt_margin      =   0.025           #offsets for shape annotators
    spacing         =   0.025           #horizontal space between shapes
    depth_spacing   =   .7 * spacing    #3D tensor depth
    max_row_height  =   .5              #maximum height in a structured graph
    label_reserve   =   5*txt_margin
    title_reserve   =   0.2



class Layer(object):
    def __init__(self, shape, layers=None, name=None):
        self.shape = shape
        self.inbound = []

        self.name = type(self).__name__
        if(name is not None):
            self.name = name

        if layers is not None:
            self.inbound = layers

    show = show_basiclayer
    setDimensions = setDimsBy

class Input(Layer):
    def __init__(self, shape, img=None):
        Layer.__init__(self, shape)
        self.img = img

class Pool(Layer):
    def __init__(self, shape, layers=None):
        Layer.__init__(self, shape, layers)

class Reshape(Layer):
    def __init__(self, shape, layers=None):
        Layer.__init__(self, shape, layers)

class CTC(Layer):
    """Connectionist Temporal Classification"""
    def __init__(self, shape, layers=None):
        Layer.__init__(self, shape, layers)

class Upsample(Layer):
    def __init__(self, shape, layers=None):
        Layer.__init__(self, shape, layers)

class Custom(Layer):
    def __init__(self, shape, name, layers=None):
        Layer.__init__(self, shape, layers)
        self.name = name

class Fluid(Custom):
    def __init__(self, shape, name, actual_size=5, layers=None):
        Custom.__init__(self, shape, name, layers)
        self.actual_size = actual_size

    def show_fluidlayer(self, axes, inputs=[]):
        """plot for fluid Layer"""
        draw_tensor(self, axes)
        placeholder = self.shape
        self.shape = (self.shape[0], self.actual_size)
        annotate_tensor(self)
        self.shape = placeholder
        add_layer_name(self)
    show = show_fluidlayer

class Flatten(Fluid):
    def __init__(self, shape, name, actual_size=5, layers=None):
        Fluid.__init__(self, shape, name, actual_size, layers)

    def show(self, axes, inputs=[]):
        def addtext(text, x_offset=0, y_offset=0, **kwargs):
            fontsize = 16
            util.add_text(  self.xy['x'] + x_offset,
                            self.xy['y'] + y_offset,
                            text, size=fontsize, **kwargs)
        addtext(self.actual_size, y_offset = 1*self.height + .025, rotation=90, va = 'bottom')

        """plot for flatten Layer"""
        draw_tensor(self, axes)
        # placeholder = self.shape
        # self.shape = (self.shape[0], self.actual_size)
        # annotate_tensor(self)
        # self.shape = placeholder
        add_layer_name(self)

class Concat(Layer):
    nConcatsUsed = 0
    concat_spacing = 0

    def __init__(self, extra_input_layers, layer=None):
        shape = list(layer.shape)
        for input_layer in extra_input_layers:
            shape[2] += list(input_layer.shape)[2] if len(input_layer.shape) > 2 else 0

        Layer.__init__(self, shape, [layer])
        self.input_layers = extra_input_layers

        Concat.nConcatsUsed += 1
        self.concat_spacing = 0.025 / Concat.nConcatsUsed #txt margin?

    def show(self, axes, inputs=[]):
        """Plot this layer"""
        def addtext(text, x_offset=0, y_offset=0, **kwargs):
            fontsize = 14
            util.add_text(  self.xy['x'] + x_offset,
                            self.xy['y'] + y_offset,
                            text, size=fontsize, **kwargs)
        addtext(self.n_neurons, y_offset = 1*self.height + .025, rotation=90, va = 'bottom')


        draw_tensor(self, axes)
        # annotate_tensor(self)
        add_layer_name(self)

        # show connections to input tensors
        # yPos = self.xy['y'] - GraphParam.txt_margin - Concat.nConcatsUsed * self.concat_spacing
        yPos = self.xy['y'] + .5 * self.height
        for idx, layer in enumerate(self.input_layers):
            # downwards spacing
            # plt.plot([layer.xy['x'], layer.xy['x']], [layer.xy['y'], yPos], linewidth=1, color=GraphParam.lineColor)
            # plt.plot([self.xy['x'], self.xy['x']], [self.xy['y'], yPos], linewidth=1, color=GraphParam.lineColor)
            # horizontal connection
            plt.plot([layer.xy['x']+layer.width, self.xy['x']], [yPos, yPos], linewidth=1, color=GraphParam.lineColor)



class FullyConnected(Layer):
    FClayers = []

    def __init__(self, layer, neurons, alpha=None):
        MAXHEIGHTHARDCODED = 42
        shape = (1, MAXHEIGHTHARDCODED)
        Layer.__init__(self, shape, layer)

        FullyConnected.FClayers += [neurons]
        self.n_neurons = neurons
        self.alpha = None if alpha is None else alpha

    # def setDimensions(self, graph=None):
    #     """fully connected layers use a softer height"""
    #     graph = self.graph if graph is None else graph
    #
    #     self.maxShape  = calcMaxShape(graph)
    #     self.width     = self.convertWidth(self.shape[0], graph)
    #     self.height    = self.shape[1] / np.max(FullyConnected.FClayers) * .5
    #     self.depth     = self.shape[2] / self.maxShape['d'] if len(self.shape) > 2 else 0

    def show(self, axes, inputs=[]):
        """Plot this layer"""
        # annotate_tensor(self, (False,True,False))
        def addtext(text, x_offset=0, y_offset=0, **kwargs):
            fontsize = 16
            util.add_text(  self.xy['x'] + x_offset,
                            self.xy['y'] + y_offset,
                            text, size=fontsize, **kwargs)
        addtext(self.n_neurons, y_offset = 1*self.height + .025, rotation=90, va = 'bottom')

        add_layer_name(self)

        n_neurons = self.n_neurons#self.shape[1]

        # draw connections
        # for layer in inputs:
        #     xy_src = ( layer.xy['x'] + layer.width, layer.xy['y'])
        #     plt.plot([xy_src[0], xyn['x']], [xy_src[1]+layer.height, xyn['y'] - self.height], linewidth=.5, alpha=.9, color=GraphParam.lineColor)
        #     plt.plot([xy_src[0], xyn['x']], [xy_src[1]+layer.height, xyn['y']], linewidth=.5, alpha=.9, color=GraphParam.lineColor)
        #     plt.plot([xy_src[0], xyn['x']], [xy_src[1], xyn['y'] - self.height], linewidth=.5, alpha=.9, color=GraphParam.lineColor)
        #     plt.plot([xy_src[0], xyn['x']], [xy_src[1], xyn['y']], linewidth=.5, alpha=.9, color=GraphParam.lineColor)



        for layer in inputs:
            # alpha_ = .01 + 1 / math.log(layer.shape[1] * n_neurons,1.5)
            if self.alpha is not None:
                alpha_ = self.alpha

            print("alpha_", alpha_)
            for h0 in tqdm(range(layer.shape[1])):
                # if n_neurons > 5 and h0%6 != 0:
                #     continue
                for neuron in range(n_neurons):
                    # if n_neurons > 5 and neuron%16 != 0:
                        # continue
                    xyn = dict(self.xy)
                    xyn['y'] += neuron * self.height / (n_neurons-1)

                    h1 = xyn['y']

                    xy_src = ( layer.xy['x'] + layer.width, layer.xy['y'])
                    # alpha.2
                    plt.plot([xy_src[0], xyn['x']], [xy_src[1]+ h0/layer.shape[1] * layer.height, h1], linewidth=.2, alpha=alpha_, color=GraphParam.lineColor,zorder=1)
                    # plt.plot([xy_src[0], xyn['x']], [xy_src[1]+layer.height, xyn['y'] - self.height], linewidth=.5, alpha=.9, color=GraphParam.lineColor)
                    # plt.plot([xy_src[0], xyn['x']], [xy_src[1]+layer.height, xyn['y']], linewidth=.5, alpha=.9, color=GraphParam.lineColor)
                    # plt.plot([xy_src[0], xyn['x']], [xy_src[1], xyn['y'] - self.height], linewidth=.5, alpha=.9, color=GraphParam.lineColor)
                    # plt.plot([xy_src[0], xyn['x']], [xy_src[1], xyn['y']], linewidth=.5, alpha=.9, color=GraphParam.lineColor)

        # draw neurons

        for neuron in range(n_neurons):
            xyn = dict(self.xy)
            xyn['y'] += neuron * self.height / (n_neurons-1)
            circ = patches.Circle(      tuple(xyn.values()),
                                        radius = 0.1 * self.height / n_neurons,
                                        color=GraphParam.tensorColor,
                                        zorder=2,
                                        linewidth=1)
            axes.add_patch(circ)

class Dropout(FullyConnected):
    # pass
    # def __init__(self, shape, layers=None):
        # Layer.__init__(self, shape, layers)

    def show(self, axes, inputs=[]):
        """Plot this layer"""
        # annotate_tensor(self, (False,True,False))
        def addtext(text, x_offset=0, y_offset=0, **kwargs):
            fontsize = 8
            util.add_text(  self.xy['x'] + x_offset,
                            self.xy['y'] + y_offset,
                            text, size=fontsize, **kwargs)
        addtext(self.n_neurons, y_offset = 1*self.height + .025, rotation=90, va = 'bottom')

        add_layer_name(self)

        # draw neurons
        n_neurons = self.n_neurons#self.shape[1]
        for neuron in range(n_neurons):
            xyn = dict(self.xy)
            xyn['y'] += neuron * self.height / (n_neurons-1)
            circ = patches.Circle(      tuple(xyn.values()),
                                        radius = 0.1 * self.height / n_neurons,
                                        color=GraphParam.tensorColor,
                                        linewidth=1,zorder=2)
            axes.add_patch(circ)

        # draw connections
        for layer in inputs:
            xy_src = ( layer.xy['x'] + layer.width, layer.xy['y'])
            plt.plot([xy_src[0], xyn['x']], [xy_src[1]+layer.height, xyn['y'] - self.height], linewidth=.5, alpha=.9, color=GraphParam.lineColor)
            plt.plot([xy_src[0], xyn['x']], [xy_src[1]+layer.height, xyn['y']], linewidth=.5, alpha=.9, color=GraphParam.lineColor)
            plt.plot([xy_src[0], xyn['x']], [xy_src[1], xyn['y'] - self.height], linewidth=.5, alpha=.9, color=GraphParam.lineColor)
            plt.plot([xy_src[0], xyn['x']], [xy_src[1], xyn['y']], linewidth=.5, alpha=.9, color=GraphParam.lineColor,zorder=1)

class Conv2D(Layer):
    """2D Convolution"""
    def __init__(self, layer, shape, kernel):
        Layer.__init__(self, shape, layer)
        self.kernel = kernel

    def place_kernel(self, input):
        """Returns the position of a kernel in an input"""
        self.kernel_rel = (self.kernel[0] * input.width / input.shape[0], self.kernel[1] * input.height / input.shape[1])
        kernel_pos = (input.xy['x'] + input.width - self.kernel_rel[0], input.xy['y'] + input.height - self.kernel_rel[1])

        if self.xy['y'] < input.xy['y']:
            kernel_pos = (input.xy['x'] + input.width - self.kernel_rel[0], input.xy['y'])
        return kernel_pos

    def place_connections(self, kernel_pos, kernel_size, input):
        """Returns the connection attachment locations to a kernel."""
        a = list(kernel_pos)
        b = list(kernel_pos)
        b[1] += self.kernel_rel[1]

        if self.xy['y'] < input.xy['y']:
            b[0] += self.kernel_rel[0]
            return (a, b)

        a[0] += self.kernel_rel[0]
        return (a, b)

    def show(self, axes, inputs=[]):
        """Default plot for Input and Layer"""
        draw_tensor(self, axes)
        annotate_tensor(self)
        add_layer_name(self)

        # draw kernel
<<<<<<< HEAD
        for input in inputs:
            # input = inputs[0]
            self.kernel_rel = (self.kernel[0] * input.width / input.shape[0], self.kernel[1] * input.height / input.shape[1])
            kernel_pos = self.place_kernel(input)
            rect = patches.Rectangle(   kernel_pos,
                                        self.kernel_rel[0],
                                        self.kernel_rel[1],
                                        color=GraphParam.lineColor,
                                        alpha=.6,
                                        linewidth=1)#, edgecolor='r', facecolor='none'
            axes.add_patch(rect)

            (down, up) = self.place_connections(kernel_pos, self.kernel_rel, input)
            dest = dict(self.xy)
            dest['y'] += self.height
            plt.plot([down[0], dest['x']], [down[1], dest['y']], linewidth=.5, alpha=0.9, color=GraphParam.lineColor)
            plt.plot([up[0], dest['x']], [up[1], dest['y']], linewidth=.5, alpha=0.9, color=GraphParam.lineColor)
=======
        input = inputs[0]
        self.kernel_rel = (self.kernel[0] * input.width / input.shape[0], self.kernel[1] * input.height / input.shape[1])
        kernel_pos = (input.xy['x'] + input.width - self.kernel_rel[0], input.xy['y'] + input.height - self.kernel_rel[1])
        rect = patches.Rectangle(   kernel_pos,
                                    self.kernel_rel[0],
                                    self.kernel_rel[1],
                                    color=GraphParam.lineColor,
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
        plt.plot([down[0], dest['x']], [down[1], dest['y']], linewidth=.5, alpha=0.9, color=GraphParam.lineColor)
        plt.plot([up[0], dest['x']], [up[1], dest['y']], linewidth=.5, alpha=0.9, color=GraphParam.lineColor)

class Conv1D(Conv2D):
    """1D Convolution"""
    def __init__(self, layer, shape, kernel):
        self.kernel = kernel
        Conv2D.__init__(self, layer, shape, (1, kernel))
>>>>>>> a79486b3fe2fae2bb551ef65ea77662b6bd8e4e8
