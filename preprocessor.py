import numpy as np
from os import listdir
from os.path import isfile, join
import scipy
import tensorflow as tf
from progress_bar import progress_bar
from inspect_images import get_bounding_box
from tiling import TileCounter

RANDOM_SEED = 2343298
TRAINING_SET_RATIO = 2


class Preprocessor():
    def __init__(self, size_x, size_y):
        self.size_x = size_x
        self.size_y = size_y


    def preprocess(self, image : np.ndarray, labels : np.ndarray, train_on_tiles = False, 
            tile_num_x = 16, tile_num_y = 16):
        im_shape = np.shape(image)
        image = scipy.misc.imresize(image, [self.size_x, self.size_y])
        image = np.reshape(image, [self.size_x, self.size_y, 1])
        image = image - image.mean()
        pic_size = image.size
        image = image / np.std(image)

        scaled_labels = []
        for label in labels:
            label = self.resize_label(label, im_shape[1], im_shape[0])
            
            scaled_label = np.asarray(label, dtype=np.float32)
            for l in scaled_label:
                l[0] /= self.size_x
                l[1] /= self.size_y
            scaled_labels.append(scaled_label)

        while len(scaled_labels) < 4:
            scaled_labels.append(np.array([[-1, -1], [-1, -1]]).astype("float32"))

        if train_on_tiles:
            tileCounter = TileCounter(tile_num_x, tile_num_y, 1, 1)
            #print("Shape of scaled_labels: %s" % str(np.shape(scaled_labels)))
            #print("Scaled labels: %s" % scaled_labels)
            # TODO reshape should not be needed here
            tiles = tileCounter.getTilesAsMatrix(np.reshape(scaled_labels, (4, 4)))

            return image.astype("float32"), np.asarray(tiles).astype("float32")

        else:   # ok the else is not really needed...
            return image.astype("float32"), scaled_labels


    def resize_label(self, label : np.ndarray, width : int, height : int):
        newlabel = np.reshape(label.copy(), [4])
        for i in range(0,4,2):
            newlabel[i] = int(newlabel[i] * self.size_x/float(width))
        for i in range(1,4,2):
            newlabel[i] = int(newlabel[i] * self.size_y/float(height))

        return np.reshape(newlabel, [2,2])


    def get_image_list(self, sample_folder : dir):
        res = []
        for f in listdir(sample_folder):
            if isfile(join(sample_folder, f)):
                if f.endswith(".jpg"): 
                    res.append(f)

        return res


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
                           'train/label': label_feature    }

                example = tf.train.Example(features=tf.train.Features(feature=feature))
                output_file.write(example.SerializeToString())


    def loadDataset(self, dataset_file : dir):
        pass


def _createBytesFeature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(value.tostring())]))
