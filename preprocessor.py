import numpy as np
from os import listdir
from os.path import isfile, join
import scipy
import tensorflow as tf
from progress_bar import progress_bar
from inspect_images import *
from tiling import TileCounter

RANDOM_SEED = 2343298
TRAINING_SET_RATIO = 2
MAX_NUMBER_OF_LABELS = 5


class BoundingBox():
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def getMidPoint(self):
        return self.x + (self.w/2), self.y + (self.h/2)


class Preprocessor():
    def __init__(self, size_x, size_y, tile_num_x, tile_num_y):
        self.size_x = size_x
        self.size_y = size_y
        self.tile_num_x = tile_num_x
        self.tile_num_y = tile_num_y
        self.tileCounter = TileCounter(tile_num_x, tile_num_y, 1, 1)


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
            scaled_label.append(1)
            #scaled_label.insert(0, 1)
            scaled_labels.append(np.asarray(scaled_label, dtype=np.float32))

        """
        while len(scaled_labels) < MAX_NUMBER_OF_LABELS:
            scaled_labels.append(np.array([-1, -1, -1, -1, 0]).astype("float32"))

        while len(scaled_labels) > MAX_NUMBER_OF_LABELS:
            del scaled_labels[-1]
        """

        if train_on_tiles:
            #print("Shape of scaled_labels: %s" % str(np.shape(scaled_labels)))
            #print("Scaled labels: %s" % scaled_labels)
            # TODO reshape should not be needed here
            tiles = tileCounter.getTilesAsMatrix(np.reshape(scaled_labels, (4, 4)))

            return image.astype("float32"), np.asarray(tiles).astype("float32")

        else:   # ok the else is not really needed...
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
            newlabel[i] = int(newlabel[i] * self.size_x/float(width))
        for i in range(1,4,2):
            newlabel[i] = int(newlabel[i] * self.size_y/float(height))

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

       
