import scipy
import numpy as np
import os
import os.path
import pandas
import scipy.ndimage
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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


def get_bounding_box(sample_image : dir):
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
    coords = f.read().strip().split(",")
    v1,v2,v3,v4 = [coords[i:i+2] for i in range(0, len(coords), 2)] # group by 2
    return np.array([v1,v2,v3,v4]).astype("int")


def draw_bounding_box(image : np.ndarray, label_polygon : list, output_polygon : list = None):
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
    plt.show()
