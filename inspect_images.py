import scipy
import numpy as np
import os
import os.path
import pandas
import scipy.ndimage
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from tiling import TileCounter
from functools import reduce
from label import *


def filter_result(result, alpha):
    res = []

    for i in range(np.shape(result)[0]):
        for j in range(np.shape(result)[1]):
            if result[i,j,4] > alpha:
                res.append(result[i,j])

    return res


def find_max(labels, pos):
    def get_bigger(x, y):
        if x[pos] > y[pos]:
            return x
        else:
            return y

    min_value = 1000
    res = list(np.zeros(np.shape(labels)[1]) - min_value)
    return reduce(get_bigger, labels, res)


def intersection_over_union(label1, label2):
    label1_minx = label1[0]
    label1_miny = label1[1]
    label1_maxx = label1[0] + label1[2]
    label1_maxy = label1[1] + label1[3]

    label2_minx = label2[0]
    label2_miny = label2[1]
    label2_maxx = label2[0] + label2[2]
    label2_maxy = label2[1] + label2[3]

    x1 = max(label1_minx, label2_minx)
    x2 = min(label1_maxx, label2_maxx)
    y1 = max(label1_miny, label2_miny)
    y2 = min(label1_maxy, label2_maxy)

    intersect = max((x2-x1),0) * max((y2-y1), 0)

    union = ((label1_maxx - label1_minx) * (label1_maxy - label1_miny)
            + (label2_maxx - label2_minx) * (label2_maxy - label2_miny)
            - intersect)

    return intersect / (union + 1e-6) #, intersect, union


def non_max_supp(labels):
    """
        Labels: [x, y, w, h, confidence]
    """
    open_set = list(map(list, np.copy(labels)))
    res = []

    while(len(open_set) > 0):
        max_label = list(find_max(open_set, 4))
        print("Checking label: %s" % max_label)
        res.append(max_label)
        print("Result so far: %s" % res)
        print("Open set: %s" % open_set)
        open_set.remove(max_label)

        head, *_ = open_set
        if intersection_over_union(max_label, head) > 0.5:
            open_set.remove(head)

    return res


def convert_network_output(network_output, alpha = 0.5):
    res = {}
    for i in range(np.shape(network_output)[0]):
        for j in range(np.shape(network_output)[1]):
            if network_output[i,j,5] > alpha:
                res[(i,j)] = np.reshape(network_output[i,j,:4], [2,2])
    return res



def get_image_stats(location : str):
    """
        Collects the dimensions of the image into a pandas.DataFrame
        if z coordinate is NaN, it means the image is black and white
    """
    res = []
    for root, dirs, files in os.walk(location):
        for f in files:
            if f.endswith(".jpg"):
                fullName = os.path.join(location, f)
                im = scipy.ndimage.imread(fullName)
                stats = [f]
                stats.extend(np.shape(im))
                res.append(stats)
    return pandas.DataFrame(data = res, columns = ["file name", "x", "y", "z"])


def order_labels(labels : list) -> list:
    if len(labels) > 1:
        return sorted(labels)
    else:
        return labels


def get_bounding_box(sample_image : dir) -> list:
    """
        This function assumes that the labels describe polygon, not bounding boxes
    """
    return LicensePlateList(read_classic_label(sample_image)).getBoundingBoxCoordinates()


def get_label_file_dir(image_dir : dir):
    return ".".join(image_dir.split(".")[:-1]) + ".txt"


def read_classic_label(sample_image : dir) -> List[np.ndarray]:
    """Reads the license plate coordinates from the file.

    Classic labels are expected in the following form:
    x11, y11, ..., x14, y14
    x21, y21, ..., x24, y24
    ...
    xN1, yN1, ..., xN4, yN4

    i.e. every line contains 4 vertices of the bounding quadrangle

    :param sample_image: location of the correspinding image
    :return: List of license plate bounding polygons in the form [x1, y1, ... ,x4,y4]
    """
    textFileName = get_label_file_dir(sample_image)
    f = open(textFileName, "r")
    line = f.readline().strip().split(",")
    res = []
    while len(line) > 1:
        res.append([int(i) for i in line])
        line = f.readline().strip().split(",")

    #print("Classic label output: %s" % res)
    return res



def get_bounding_polygon(sample_image : dir):
    """
        It assumes that the label is in the same location with the same name
        as the picture except for the ".txt" extension. 
        E.g. if the picture's location (sample_image) is 
            "./samples/pic331.jpg",
        the label must be in 
            "./samples/pic331.txt"
       
        Labels are assumed to be in the following form in the label txt:
            x1,y1,x2,y2[,x3,y3,x4,y4]

        This function works even if only 2 vertices are provided (for bounding boxes).
    """
    textFileName = get_label_file_dir(sample_image)
    f = open(textFileName, "r")
    coords = f.readline().strip().split(",")
    res = []
    while len(coords) > 1:
        res.append(np.array([coords[i:i+2] for i in range(0, len(coords), 2)]).astype("int")) # group by 2
        coords = f.readline().strip().split(",")
    #return order_labels(res)
    return res

"""
def draw_bounding_box_from_polygons(image : np.ndarray, label_polygon : list, output_polygon : list = None):
    draw_bounding_box(image, convert_to_bounding_boxes(label_polygon),
        convert_to_bounding_boxes(output_polygon))
"""

def save_bounding_box(image : np.ndarray, 
                      label_polygon : list,
                      save_file_loc : dir,
                      output_polygon : list = None,
                      tile_num_x = 8,
                      tile_num_y = 8):
    _draw_bounding_box(image, label_polygon, output_polygon, tile_num_x, tile_num_y)
    plt.savefig(save_file_loc)




def draw_bounding_box(image : np.ndarray, 
                      label_polygon : list,
                      output_polygon : list = None,
                      draw_tiles : bool = False,
                      tile_num_x = 16,
                      tile_num_y = 16,
                      height_width : bool = True):
    _draw_bounding_box(image, label_polygon, output_polygon, draw_tiles, tile_num_x, tile_num_y, height_width)
    plt.show()


def _draw_bounding_box_on_pic(ax, label, height_width : bool = False):
    if height_width:
        ax.add_patch(patches.Rectangle(
            (label[0][0], label[0][1]),
            label[1][0], label[1][1],
            fill=False, linewidth=1, color='tab:blue'))
    else:
        ax.add_patch(patches.Rectangle(
            (label[0][0], label[0][1]),
            label[1][0] - label[0][0],
            label[1][1] - label[0][1],
            fill=False, linewidth=1, color='tab:blue'))


def _draw_bounding_box(image : np.ndarray, 
                      label_polygon : list,
                      output_polygon : list = None,
                      draw_tiles : bool = False,
                      tile_num_x = 16,
                      tile_num_y = 16,
                      height_width: bool = True):
    """
        `label_polygon` and `output_polygon` are both expected in the following form:
          [ [x1,y1], [x2,y2], [x3,y3], [x4,y4] ]
        typewise both can be numpy ndarrays or list of lists
    """
    fig, ax = plt.subplots(1)
    shape = np.shape(image)
    size_x = shape[1]
    size_y = shape[0]

    if draw_tiles:
        tileCounter = TileCounter(tile_num_x, tile_num_y, size_x, size_y)
        tile_list = tileCounter.getTiles(label_polygon)
        draw_tiles(ax, tile_list, size_x, size_y)
        draw_grid_on_pic(ax, size_x, size_y, tile_num_x, tile_num_y)

    if len(shape) > 2:
        ax.imshow(image)
    else:
        ax.imshow(image, cmap='gray')

    for one_label in label_polygon:
        # rectangle expects height and width and not 2nd coordinates as the 2nd vertice of the bounding box
        _draw_bounding_box_on_pic(ax, one_label, height_width=height_width)

    if output_polygon is not None:
        for one_output_polygon in output_polygon:
            if height_width:
                ax.add_patch(patches.Rectangle(
                    (one_output_polygon[0][0], one_output_polygon[0][1]),
                    one_output_polygon[1][0], one_output_polygon[1][1],
                    fill=False, linewidth=1, color='tab:blue'))
            else:
                ax.add_patch(patches.Rectangle(
                    (one_output_polygon[0][0], one_output_polygon[0][1]),
                    one_output_polygon[0][0] - one_output_polygon[1][0],
                    one_output_polygon[0][1] - one_output_polygon[1][1],
                    fill=False, linewidth=1, color='tab:blue'))

    plt.show()




def draw_bounding_polygon(image : np.ndarray, label_polygon : list, output_polygon : list = None):
    """
        `label_polygon` and `output_polygon` are both expected in the following form:
          [ [x1,y1], [x2,y2], [x3,y3], [x4,y4] ]
        typewise both can be numpy ndarrays or list of lists
    """
    fig, ax = plt.subplots(1)
    shape = np.shape(image)
    if len(shape) > 2:
        ax.imshow(image)
    else:
        ax.imshow(image, cmap='gray')
    for one_label in label_polygon:
        ax.add_patch(patches.Polygon(one_label, fill=False, linewidth=1, color='tab:green'))

    if output_polygon is not None:
        for one_output_polygon in output_polygon:
            ax.add_patch(patches.Polygon(one_output_polygon, fill=False, linewidth=1, color='tab:red'))

    plt.show()


def plot_output(image, label_output, logit_output, alpha = 0.5):
    converted_label = convert_network_output(logit_output, alpha)
    print("number of filtered elements in logit output: %i" % len(converted_label))
    print("filtered output: %s" % str(converted_label))
    draw_float_bounding_box(
            np.squeeze(image),
            np.reshape(label_output[:,:4], [np.shape(label_output)[0],2,2]), 
            converted_label,
            midrepr = True)


def draw_float_bounding_box(image : np.ndarray,
                            label_polygon : list,
                            output_polygon : list = None,
                            draw_tiles : bool = False,
                            tile_num_x = 5,
                            tile_num_y = 5,
                            midrepr = False):
    """
        `label_polygon` and `output_polygon` are both expected in the following form:
          [ [x1,y1], [x2,y2] ]
        typewise both can be numpy ndarrays or list of lists
    """
    fig, ax = plt.subplots(1)
    #canvas = FigureCanvas(fig)

    shape = np.shape(image)
    size_x = shape[1]
    size_y = shape[0]

    if draw_tiles:
        tileCounter = TileCounter(tile_num_x, tile_num_y, size_x, size_y)
        tile_list = tileCounter.getTiles(label_polygon)
        draw_tiles(ax, tile_list, size_x, size_y)

    for one_label in label_polygon:
        output = np.copy(np.asarray(one_label))

        """
        for l in output:
            l[0] *= size_x
            l[1] *= size_y
        """
        print("output: %s" % str(output))
        output[0][0] *= size_x
        output[1][0] *= size_x
        output[0][1] *= size_y
        output[1][1] *= size_y

        if midrepr:
            output[0][0] -=  (output[1][0] / 2)
            output[0][1] -=  (output[1][1] / 2)
        else:
            output[1][0] -= output[0][0]
            output[1][1] -= output[0][1]

        ax.add_patch(patches.Rectangle( 
            (output[0][0], output[0][1]),
            output[1][0], output[1][1],
            fill=True, linewidth=1, color='tab:green', alpha=0.5))

    """ I don't quite remember why it was this way
    if output_polygon is not None:
        for pos, one_output in output_polygon.items():
            output = np.copy(np.asarray(one_output))

            print("output: (%s, %s)" % (pos, one_output))

            output[0][0] += (pos[1] / tile_num_x)
            output[0][1] += (pos[0] / tile_num_y)

            print("(%.8f, %.8f)" % (output[0][0], output[0][1]))

            output[0][0] *= size_x/tile_num_x
            output[1][0] *= size_x/tile_num_y
            output[0][1] *= size_y
            output[1][1] *= size_y

            if midrepr:
                output[0][0] -=  (output[1][0] / 2)
                output[0][1] -=  (output[1][1] / 2)

            ax.add_patch(patches.Rectangle( 
                (output[0][0], output[0][1]),
                output[1][1], output[1][0],
                fill=True, linewidth=1, color='tab:blue', alpha=0.5))
    """
    if output_polygon is not None:
        for one_l_output in output_polygon:
            one_output = np.copy(np.asarray(one_l_output))
            print("net output: %s" % str(one_output))
            one_output[0][0] *= size_x
            one_output[1][0] *= size_x
            one_output[0][1] *= size_y
            one_output[1][1] *= size_y

            if midrepr:
                one_output[0][0] -= (one_output[1][0] / 2)
                one_output[0][1] -= (one_output[1][1] / 2)
            else:
                one_output[1][0] -= one_output[0][0]
                one_output[1][1] -= one_output[0][1]

            ax.add_patch(patches.Rectangle(
                (one_output[0][0], one_output[0][1]),
                one_output[1][0], one_output[1][1],
                fill=True, linewidth=1, color='tab:red', alpha=0.5))

    ax.imshow(image, cmap='gray')

    #plt.show()

    fig.canvas.draw()

    data = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8', sep='')
    return data.reshape(fig.canvas.get_width_height()[::-1] + (3,))



def draw_tiles(ax, tiles : list, size_x : int, size_y : int):
    for tile in tiles:
        x = int(tile.getX1(size_x))
        y = int(tile.getY1(size_y))
        w = int(tile.getWidth(size_x))
        h = int(tile.getHeight(size_y))

        ax.add_patch(patches.Rectangle((x,y), w, h, alpha=0.8))


def draw_grid_on_pic(ax,
                     size_x : int,
                     size_y : int,
                     num_grid_x : int = 8,
                     num_grid_y : int = 8,
                     grid_width = 1,
                     color = "black"):
    for i in range(1, num_grid_x):
        x = (i/num_grid_x) * size_x
        ax.add_patch(patches.Rectangle( (x, 0), grid_width, size_y, color=color))

    for i in range(1, num_grid_y):
        y = (i/num_grid_y) * size_y
        ax.add_patch(patches.Rectangle( (0, y), size_x, grid_width, color=color))


def save_bounding_box(save_file : dir, image : np.ndarray, label_polygon : list, output_polygon : list = None):
    """
        `label_polygon` and `output_polygon` are both expected in the following form:
          [ [x1,y1], [x2,y2], [x3,y3], [x4,y4] ]
        typewise both can be numpy ndarrays or list of lists
    """
    fig, ax = plt.subplots(1)
    shape = np.shape(image)
    if len(shape) > 2:
        ax.imshow(image)
    else:
        ax.imshow(image, cmap='gray')
    bb = patches.Polygon(label_polygon, fill=True, linewidth=1, color='tab:green', alpha=0.5)
    ax.add_patch(bb)
    if output_polygon is not None:
        bb2 = patches.Polygon(output_polygon, fill=True, linewidth=1, color='tab:red', alpha=0.5) 
        ax.add_patch(bb2)
    plt.savefig(save_file)
