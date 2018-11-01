import numpy as np
from os import listdir
from os.path import isfile, join
import scipy
import tensorflow as tf
from progress_bar import progress_bar
from inspect_images import *
from enum import Enum
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from etc import threadsafe_generator
from typing import List
from collections import namedtuple

RANDOM_SEED = 2343298
TRAINING_SET_RATIO = 2


class BoundingBox():
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def getMidPoint(self):
        return self.x + (self.w/2), self.y + (self.h/2)

    def getMidPointRepr(self):
        return [self.x + (self.w/2), self.y + (self.h/2), self.w, self.h]


class Preprocessor():
    def __init__(self, size_x, size_y, max_number_of_labels):
        self.size_x = size_x
        self.size_y = size_y
        self.max_number_of_labels = max_number_of_labels


    def preprocess(self, image : np.ndarray, labels : np.ndarray, train_on_tiles = False): 
        im_shape = np.shape(image)
        image = scipy.misc.imresize(image, [self.size_x, self.size_y])
        image = np.reshape(image, [self.size_x, self.size_y, 1])
        image = image - image.mean()
        pic_size = image.size
        image = image / np.std(image)

        scaled_labels = []
        for label in labels:
            scaled_label = self.resize_label(label, im_shape[1], im_shape[0])
            #assert ((scaled_label[0] + scaled_label[2] <= 1) and (scaled_label[1] + scaled_label[3] <= 1)), \
            #    "Error. Original label: %s, converted: %s" % (str(label), str(scaled_label))
            scaled_label = np.append(scaled_label, 1)
            #scaled_label.insert(0, 1)
            scaled_labels.append(np.asarray(scaled_label, dtype=np.float32))

        while len(scaled_labels) < self.max_number_of_labels:
            scaled_labels.append(np.array([0, 0, 0, 0, 0]).astype("float32"))

        while len(scaled_labels) > self.max_number_of_labels:
            del scaled_labels[-1]

        if train_on_tiles:
            #print("Shape of scaled_labels: %s" % str(np.shape(scaled_labels)))
            #print("Scaled labels: %s" % scaled_labels)
            # TODO reshape should not be needed here
            tiles = self.tileCounter.getTilesAsMatrix(np.reshape(scaled_labels, (4, 4)))

            return image.astype("float32"), np.asarray(tiles).astype("float32")

        else:   # ok the else is not really needed...
            #print("scaled labels: %s" % str(scaled_labels))
            return image.astype("float32"), scaled_labels


    def createOutputMatrix(self, labels):
        """
            This function is to create the output matrix for YOLO.

            Shape of output: [tile_x, tile_y, 5]
                the 5 comes from: [c, x, y, w, h], where:
                    c : 1 if object is present, 0 if not
                    x, y : coords of the upper-left corner
                    w, h : widht, height
        """
        res = np.zeros([self.tile_x, self.tile_y, 5])
        for label in labels:
            x = label[0][0]
            y = label[0][1]
            tile_x = min(math.floor( ( x * self.tile_num_x ) ), self.tile_num_x-1)
            tile_y = min(math.floor( ( y * self.tile_num_y ) ), self.tile_num_y-1)           


    def resize_label(self, label : np.ndarray, width : int, height : int):
        newlabel = np.reshape(label.copy(), [4])
        for i in range(0,4,2):
            newlabel[i] = newlabel[i] / float(width)
        for i in range(1,4,2):
            newlabel[i] = newlabel[i] / float(height)

        return newlabel


    def writeDataset(self, data_dir : dir, dataset_file : dir, train_on_tiles = False):
        """
            Creates a dataset out of the images and labels found in data_dir and writes it to the dataset_file.
            Warning! It will overwrite the dataset_file!

            Inputs:
                data_dir : dir - folder which contains the pictures and labels
                dataset_file : dir - location where the dataset is to be written

            Outputs:
                None
        """
        pics = self.get_image_list(data_dir)

        images, labels = (0,0)

        #random.seed(RANDOM_SEED)
        training_set_size = int(TRAINING_SET_RATIO*len(pics))
        training_pics = pics[:training_set_size]
        #training_pics = pics

        with tf.python_io.TFRecordWriter(dataset_file) as output_file:
            for i in range(len(training_pics)):
                picloc = training_pics[i]
                progress_bar(i+1, len(training_pics), "Writing dataset")
                fullpicloc = join(data_dir, picloc)
                pic = scipy.ndimage.imread(fullpicloc, mode="L")
                label = get_bounding_box(fullpicloc)

                pic,label = self.preprocess(pic, label, train_on_tiles)

                pic_feature = _createBytesFeature(pic)
                label_feature = _createBytesFeature(np.asarray(label))

                feature = {'train/image': pic_feature,
                           'train/label': label_feature }

                example = tf.train.Example(features=tf.train.Features(feature=feature))
                output_file.write(example.SerializeToString())


    def loadDataset(self, dataset_file : dir):
        pass


def getMidPoint(label):
    return label[0][0] + (label[0][1] / 2), label[0][1] + (label[1][1] / 2)


def get_image_list(sample_folder : dir, return_with_full_name = False):
    res = []
    for f in listdir(sample_folder):
        fullpath = join(sample_folder, f)
        if isfile(fullpath):
            if f.endswith(".jpg"): 
                if return_with_full_name:
                    res.append(fullpath)
                else:
                    res.append(f)
    return res

"""
def writeLabelsToFile(f, labels : list):
    for label in labels:
        f.write(str(label)[1:-1] + '\n')
"""


def convertPolygonToBoundingBox(sample_dir : dir, output_dir : dir):
    image_list = get_image_list(sample_dir, return_with_full_name = True)
    for image_loc in image_list:
        labels = get_bounding_box(image_loc)
        label_loc = get_label_file_dir(image_loc)
        label_file_name = label_loc.split('/')[-1]
        output_label_loc = join(output_dir, label_file_name)
        print("Labels: %s" % labels)
        if not isfile(output_label_loc):
            with open(output_label_loc, 'w') as f:
                writeLabelsToFile(f, labels)


def _createBytesFeature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(value.tostring())]))


def writeLabelsToFile(labels : list, file_loc : dir):
    reshaped = np.reshape(labels, [len(labels), 4])
    # prevent overwrite
    if not isfile(file_loc):
        with open(file_loc, 'w') as f:
            for label in reshaped:
                f.write(', '.join(map(str, label)) + '\n')



def convertToHeightWidthRepr(label : list):
    res = np.copy(label)
    res[1][0] = res[1][0]-res[0][0]
    res[1][1] = res[1][1]-res[0][1]
    return res


def convertAllToHeightWidthRepr(sample_folder : dir, output_folder : dir):
    for f in listdir(sample_folder):
        fullpath = join(sample_folder, f)
        if isfile(fullpath):
            if f.endswith('.jpg'):
                labels = get_bounding_polygon(fullpath)
                converted = []
                for label in labels:
                    converted.append( convertToHeightWidthRepr(label) )
                output_file_path = '.'.join(join(output_folder, f).split('.')[:-1]) + '.txt'
                writeLabelsToFile(converted, output_file_path)


class InputIterator:
    def getOutputDim(self) -> List[int]:
        raise NotImplementedError("Function not implemented")


class SlidingWindowSampleCreator:
    # TODO add support for other padding types
    class Padding(Enum):
        ZEROES = 1
        DONT_SLIDE_OUTSIDE = 2   # WARNING! Not supported yet!

    def __init__(self, slide_x : int, slide_y : int, window_width : int, window_height : int,
                 normalize_label : bool = False):
        """This class creates small cropped images out of a big one along a sliding window.
        If part of the window would go outside the picture, it will be padded by zeros.

        :param slide_x: window will slide along the x axis by this ammount
        :param slide_y: window will slide along the y axis by this ammount
        :param window_width: width of the sliding window
        :param window_height: height of the sliding window
        :param normalize_label: if True, labels will be between 0.0 and 1.0 (unless they are already -1
            which denotes "no object on this picture"
        """
        self.slide_x = slide_x
        self.slide_y = slide_y
        self.window_width = window_width
        self.window_height = window_height
        self.normalize_label = normalize_label


    def getOutputDim(self):
        # max number of labels: 1, size of label: 4
        return self.window_height, self.window_width, 1, 4


    @threadsafe_generator
    def create_sliding_window_samples(self, image : np.ndarray, labels : np.ndarray = None):
        """creates cropped image via sliding window

        note: label is expected in the (left upper point, right bottom point) representation,
            not in width-height form!

        :param image: ndarray containing the image as grayscale
        :param labels: ndarray in the following form: [[x1, y1, x2, y2], ...]
        :return: cropped part of the image
        """
        #if self.padding == SlidingWindowSampleCreator.Padding.ZEROES:
        num_of_slide_x = math.ceil( ( image.shape[1] - self.window_width ) / self.slide_x )
        num_of_slide_y = math.ceil( ( image.shape[0] - self.window_height ) / self.slide_y )
        pad_x = self.window_width - ( ( image.shape[1] - self.window_width ) % self.slide_x )
        pad_x %= self.window_width
        pad_y = self.window_height - ( ( image.shape[0] - self.window_height ) % self.slide_y )
        pad_y %= self.window_height

        """
        print("Padding added: (%i, %i)" % (pad_x, pad_y))
        print("Number of sliding: (%i, %i)" % (num_of_slide_x, num_of_slide_y))
        print("asdasdsd")
        """

        padded_image = np.pad(image.copy(), ((0, pad_y), (0, pad_x)), mode='constant', constant_values=0)

        if labels is None:
            for i in range(num_of_slide_x+1):
                for j in range(num_of_slide_y+1):
                    x = i*self.slide_x
                    y = j*self.slide_y
                    yield padded_image[y:y + self.window_height, x:x + self.window_width]

        else:
            for i in range(num_of_slide_x + 1):
                for j in range(num_of_slide_y + 1):
                    x = i * self.slide_x
                    y = j * self.slide_y

                    label = filter_and_transform_labels(labels, x, y, self.window_width, self.window_height)

                    if self.normalize_label and label[0] != -1:
                        label = np.array(label).astype(np.float)
                        label[0] /= float(self.window_width)
                        label[1] /= float(self.window_height)
                        label[2] /= float(self.window_width)
                        label[3] /= float(self.window_height)

                    yield padded_image[y:y + self.window_height, x:x + self.window_width], label


    @threadsafe_generator
    def create_sliding_window_from_iter(self, image_iter):
        for i in image_iter:
            for j in self.create_sliding_window_samples(i["image"], i["label"]):
                yield j


EnqueueThread = namedtuple("EnqueueThread",
    [ "pic_batch", "label_batch", "enqueue_op", "enq_image", "enq_label", "examples_in_queue", "queue_close_op"])


def crop_labels(labels : np.ndarray, window_width : int, window_height : int) -> list:
    def crop_x(x):
        return min(max(x, 0), window_width)
    def crop_y(y):
        return min(max(y, 0), window_height)

    return [ [crop_x(l[0]), crop_y(l[1]), crop_x(l[2]), crop_y(l[3])] for l in labels ]


def shift_labels(labels : np.ndarray, x : int, y : int) -> list:
    return [[l[0] - x, l[1] - y, l[2] - x, l[3] - y] for l in labels]


def shift_and_crop_labels(labels : np.ndarray, x : int, y : int, window_width : int, window_height : int) -> np.ndarray:
    return crop_labels(shift_labels(labels, x,y), window_width, window_height)


def filter_and_transform_labels(labels : np.ndarray, x : int, y : int, window_width : int, window_height : int) \
        -> np.ndarray:
    filtered_labels = [l for l in labels if isLabelInWindow(l, x, y, window_width, window_height)]
    if len(filtered_labels) > 0:
        # it is expected that at most one label remains in the picture
        return shift_and_crop_labels(labels, x, y, window_height, window_width)[0]
    else:
        return [-1, -1, -1, -1]


def isLabelInWindow(label : np.ndarray, x : int, y : int, window_width : int, window_height : int) -> bool:
    return (x < label[0] < x + window_width) or\
           (y < label[1] < y + window_height) or\
           (x < label[2] < x + window_width) or\
           (y < label[3] < y + window_height)


def plotImages(images : np.ndarray, square_width = 1.0, square_height = 1.0, normalize = False):
    #numOfKernels = np.shape(images)[0]
    numOfImages = len(images)
    numOfRows = int(np.ceil(np.sqrt(numOfImages)))
    numOfCols = int(np.ceil(numOfImages / numOfRows))

    f, axs = plt.subplots(numOfCols, numOfRows, figsize=(numOfRows, numOfCols))

    # axs is a 2d array
    for ax1 in axs:
        for ax2 in ax1:
            ax2.axis('off')

    for i in range(numOfCols):
        for j in range(numOfRows):
            index = i*numOfRows + j

            if index < numOfImages:
                axs[i, j].imshow(images[index], cmap='gray')
            else:
                return


def plotImagesWithLabels(images_with_labels : np.ndarray, normalized = False):
    #numOfKernels = np.shape(images)[0]
    images = []
    labels = []
    for (im, la) in images_with_labels:
        images.append(im)
        labels.append(la)

    print("Images size: %s, labels size: %s" % (len(images), len(labels)))

    numOfImages = len(images)
    numOfRows = int(np.ceil(np.sqrt(numOfImages)))
    numOfCols = int(np.ceil(numOfImages / numOfRows))

    f, axs = plt.subplots(numOfCols, numOfRows, figsize=(numOfRows, numOfCols))

    # axs is a 2d array
    for ax1 in axs:
        for ax2 in ax1:
            ax2.axis('off')

    for i in range(numOfCols):
        for j in range(numOfRows):
            index = i*numOfRows + j

            if index < numOfImages:
                label = np.array(labels[index]).copy()
                axs[i, j].imshow(images[index], cmap='gray')

                if normalized:
                    label[0] = int(label[0] * images[index].shape[1])
                    label[1] = int(label[1] * images[index].shape[0])
                    label[2] = int(label[2] * images[index].shape[1])
                    label[3] = int(label[3] * images[index].shape[0])

                axs[i, j].add_patch(patches.Rectangle(
                    (label[0], label[1]),
                    label[2] - label[0], label[3] - label[1],
                    fill=False, linewidth=1, color='tab:blue'))
            else:
                plt.show()
                return
