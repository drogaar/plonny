import matplotlib.pyplot as plt

def add_text(x, y, s, **kwargs):
    """Add text to plot and return the boundary box in (x1, y1, x2, y2) form"""

    def invert_sizes(start_axes, display_size_coords):
        """Given starting point in axis space and size in display space;
        convert size to axis space."""
        x,y = start_axes
        width, height = display_size_coords

        ax = plt.gca()
        ax_inv = ax.transAxes.inverted()

        # convert axis space starting points to display space
        # Assumes left/bottom bbox alignment for calculation
        ax_min_display = ax.transAxes.transform(start_axes)
        ax_max_display = (ax_min_display[0] + width, ax_min_display[1] + height)

        # Invert end point and subtract start to obtain axes space size
        ax_max_axes = ax_inv.transform((ax_max_display[0],  ax_max_display[1]))
        return (ax_max_axes[0] - x, ax_max_axes[1] - y)

    t = plt.text(x, y, s, kwargs)

    # Get text boundary box shape in axis coordinates
    sizes_display   = t.get_window_extent(renderer=plt.gcf().canvas.get_renderer())
    width, height   = invert_sizes((x,y), (sizes_display.width, sizes_display.height))

    # get alignments
    ha = 'left'
    va = 'top'
    if 'ha' in kwargs: ha = kwargs['ha']
    if 'va' in kwargs: va = kwargs['va']

    # Use alignment to locate bounding box
    offset_vertical = {'bottom' : 1, 'center': .5, 'top' : 0}
    offset_horizontal = {'left' : 1, 'center': .5, 'right' : 0}
    startx = x - width + offset_horizontal[ha] * width
    starty = y - height + offset_vertical[va] * height
    endx = x + offset_horizontal[ha] * width
    endy = y + offset_vertical[va] * height

    return (t, startx, starty, endx, endy)
