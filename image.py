import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
from matplotlib.image import AxesImage
import utils

def load_image(filename):
    """
        Loads the specified file as a ndarray of (r, g, b, a)
    """

    return imread(filename, mode='RGBA')

def create_boundary_points(shape):
    """
        Create a set of 20 points equidistant around the boundary. Assumes that the image has values in (0, xmax), (0, ymax)
    """
    xmin = 0
    ymin = 0
    xmax = shape[0] -1
    ymax = shape[0] -1

    xlin = np.floor(np.linspace(0, xmax, 6))
    ylin = np.floor(np.linspace(0, ymax, 6))

    points = []
    for point in xlin:
        points.append([point, 0])
        points.append([point, ymax])

    for point in ylin:
        points.append([0, point])
        points.append([xmax, point])

    return np.unique(np.array(points, dtype=np.float64), axis=0)


def pick_points(image, points = None, close_fig = False, filename= None):
    """
        Plots an image and allows the user to select points on it. 

        @param points, if not None use this set of points as the inital set instead of the automatic boundary points

        @param close_fig, Close the figure after displaying

        @param filename, Save the figure at this file if specified and save-files is specified

        @return The set of chosen points
    """
    plt.ion()
    fig = plt.figure()
    fig_plot = fig.add_subplot(111)

    # Always want to have points on the boundary
    if points is None:
        points = create_boundary_points(image.shape[:2])

    sc = fig_plot.scatter(points[:, 0], points[:, 1])
    num_points = points.shape[0]

    texts = []

    def pick_event(event):
        nonlocal points
        nonlocal num_points
        nonlocal texts
        m = event.mouseevent
        x,y = np.floor([m.xdata, m.ydata])
        print(x,y)

        #allow removing points, if the user clicks approximately near the chosen point
        # remove it instead of adding another point at the location
        distance = np.linalg.norm(points - np.array([x,y]), axis=1)
        points = points[distance > 8]

        #If the user clicked near an existing point, remove it
        if points.shape[0] < num_points:
            num_points = points.shape[0]
            for text in texts:
                fig_plot.texts.remove(text)

            texts = []
            for i,p in enumerate(points):
                texts.append(fig_plot.text(p[0] + .03, p[1] + .03, i + 1))
        # Otherwise add the new point to the list
        else:
            num_points = points.shape[0] + 1
            txt = fig_plot.text(x + .03, y + .03, num_points)
            texts.append(txt)

            points = np.append(points, [[x,y]], axis=0)

        #Update the set of drawn points
        sc.set_offsets(points)
        fig.canvas.draw_idle()
        
    #Allow the user to select up to 100 points
    img = fig_plot.imshow(image, picker=100)

    fig.canvas.mpl_connect('pick_event', pick_event)

    fig.show()

    utils.prompt()

    if utils.save_files and filename is not None:
        plt.savefig(filename)

    if close_fig:
        plt.close()

    return np.array(points)
