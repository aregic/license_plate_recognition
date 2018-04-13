import scipy
import numpy as np
import os
import os.path
import pandas
import scipy.ndimage
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tiling import TileCounter

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
    print("List: %s" % labels)
    if len(labels) > 1:
        return sorted(labels)
    else:
        return labels


def convert_to_bounding_boxes(labels : list) -> list:
    if labels is None:
        return None
    res = []
    for label in labels:
        x1 = min(label[0][0], label[1][0], label[2][0], label[3][0])
        x2 = max(label[0][0], label[1][0], label[2][0], label[3][0])
        y1 = min(label[0][1], label[1][1], label[2][1], label[3][1])
        y2 = max(label[0][1], label[1][1], label[2][1], label[3][1])
        res.append([x1, y1, x2, y2])
    return res


def get_bounding_box(sample_image : dir) -> list:
    return convert_to_bounding_boxes(get_bounding_polygon(sample_image))


def get_bounding_polygon(sample_image : dir):
    """
        It assumes that the label is in the same location with the same name
        as the picture except for the ".txt" extension. 
        E.g. if the picture's location (sample_image) is 
            "./samples/pic331.jpg",
        the label must be in 
            "./samples/pic331.txt"
       
        Labels are assumed to be in the following form in the label txt:
            x1,y1,x2,y2,x3,y3,x4,y4
    """
    textFileName = ".".join(sample_image.split(".")[:-1]) + ".txt"
    f = open(textFileName, "r")
    coords = f.readline().strip().split(",")
    res = []
    while len(coords) > 1:
        res.append(np.array([coords[i:i+2] for i in range(0, len(coords), 2)]).astype("int")) # group by 2
        coords = f.readline().strip().split(",")
    #return order_labels(res)
    return res


def draw_bounding_box_from_polygons(image : np.ndarray, label_polygon : list, output_polygon : list = None):
    draw_bounding_box(image, convert_to_bounding_boxes(label_polygon),
        convert_to_bounding_boxes(output_polygon))


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
                      tile_num_x = 8,
                      tile_num_y = 8):
    _draw_bounding_box(image, label_polygon, output_polygon, tile_num_x, tile_num_y)
    plt.show()


def _draw_bounding_box(image : np.ndarray, 
                      label_polygon : list,
                      output_polygon : list = None,
                      tile_num_x = 8,
                      tile_num_y = 8):
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
        ax.add_patch(patches.Rectangle( 
            (one_label[0], one_label[1]),
            one_label[2] - one_label[0],
            one_label[3] - one_label[1],
            fill=False, linewidth=1, color='tab:green'))

    if output_polygon is not None:
        for one_output_polygon in output_polygon:
            ax.add_patch(patches.Rectangle( 
                (one_output_polygon[0], one_output_polygon[1]),
                one_output_polygon[2] - one_output_polygon[0],
                one_output_polygon[3] - one_output_polygon[1],
                fill=False, linewidth=1, color='tab:green'))

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


def draw_float_bounding_box(image : np.ndarray,
                            label_polygon : list,
                            output_polygon : list = None,
                            draw_tiles : bool = False,
                            tile_num_x = 8,
                            tile_num_y = 8):
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

    for l in label:
        l[0] *= size_x
        l[1] *= size_y
 
    if len(shape) > 2:
        ax.imshow(image)
    else:
        ax.imshow(image, cmap='gray')


    for one_label in label_polygon:
        label = np.copy(np.asarray(label_polygon))



        ax.add_patch(patches.Polygon(label, fill=False, linewidth=1, color='tab:green'))

    if output_polygon is not None:
        for one_output in output_polygon:
            output = np.copy(np.asarray(one_polygon))
            for l in output:
                l[0] *= size_x
                l[1] *= size_y
            ax.add_patch(patches.Polygon(output, fill=False, linewidth=1, color='tab:red'))

    plt.show()


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
    bb = patches.Polygon(label_polygon, fill=False, linewidth=1, color='tab:green')
    ax.add_patch(bb)
    if output_polygon is not None:
        bb2 = patches.Polygon(output_polygon, fill=False, linewidth=1, color='tab:red') 
        ax.add_patch(bb2)
    plt.savefig(save_file)
