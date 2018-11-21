import tensorflow as tf
from collections import namedtuple
from preprocessor import get_image_list, get_bounding_box
import scipy
import random
from etc import *
import numpy as np
import math
from enum import Enum
from typing import List
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import time
from configs import EnvConfig, NetworkConfig


"""
This file contains the input pipeline, or at least most of it.
"""


@threadsafe_generator
def imageLabelIterator(folder : dir, random_seed : int = 54323):
    """reads image and label from given directory.
    Label is expected to be a polygon which will be transformed into a bounding box.

    :param folder: it must contain the images and the labels as .txt files
    :return: one image and label per call
    """
    image_list = get_image_list(folder, return_with_full_name=True)

    buffer = []
    for image_loc in image_list:
        label = get_bounding_box(image_loc)
        image = scipy.ndimage.imread(image_loc, mode="L")
        image = image.astype(np.float) / 256.0
        buffer.append({"label" : label, "image" : image})

    random.seed(random_seed)
    random.shuffle(buffer)

    while True:
        for i in range(len(buffer)):
            yield buffer[i]

        seed = random.randint(0,10000)
        random.seed(seed)
        random.shuffle(buffer)



class InputIterator:
    def getOutputDim(self) -> List[int]:
        raise NotImplementedError("Function not implemented")


class SlidingWindowSampleCreator:
    # TODO add support for other padding types
    class Padding(Enum):
        ZEROES = 1
        DONT_SLIDE_OUTSIDE = 2   # WARNING! Not supported yet!

    def __init__(self, slide_x : int, slide_y : int, window_width : int, window_height : int,
                 normalize_label : bool = False, no_label_weight : float = 1.0, yes_label_weight : float = 1.0):
        """This class creates small cropped images out of a big one along a sliding window.
        If part of the window would go outside the picture, it will be padded by zeros.

        Since there are generally more slices of picture which contain no object (which we want to find),
        it is a good idea to weight pictures with label on it and without differently, so the overwelming number
        of "nothing on the picture" examples doesn't convince the network to just output "no object found" all
        the time. For this there are the no_label_weight and yes_label_weight parameters. It is used in the
        following way:
        1.) label ratio is computed (r): what ratio is inside the current sliding window
        2.) loss is multiplied by (r*yes_label_weight + (1-r)*no_label_weight)
        3.) loss is stored in the +1st element in the label output array

        :param slide_x: window will slide along the x axis by this ammount
        :param slide_y: window will slide along the y axis by this ammount
        :param window_width: width of the sliding window
        :param window_height: height of the sliding window
        :param normalize_label: if True, labels will be between 0.0 and 1.0 (unless they are already -1
            which denotes "no object on this picture"
        :param no_label_weight: a factor of loss when no label is present inside the sliding window
        :param yes_label_weight: a factor of loss when label is present inside the sliding window
        """
        self.slide_x = slide_x
        self.slide_y = slide_y
        self.window_width = window_width
        self.window_height = window_height
        self.normalize_label = normalize_label
        self.no_label_weight = no_label_weight
        self.yes_label_weight = yes_label_weight


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
        :return: cropped part of the image, label, loss_penalty
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

                    label = np.array(
                        filter_and_transform_labels(labels, x, y, self.window_width, self.window_height)
                    ).astype(np.float)

                    #label[4] is expected to be 0 if no label is inside the sliding window
                    if self.normalize_label and label[4] != 0:
                        label[0] /= float(self.window_width)
                        label[1] /= float(self.window_height)
                        label[2] /= float(self.window_width)
                        label[3] /= float(self.window_height)

                    # label[5] contains the ratio of the part which is visible
                    # weighted average of yes_label_weight and no_label_weight
                    #print("Label weight: %.4f" % label[5])
                    label[5] = (label[5]*self.yes_label_weight + (1-label[5])*self.no_label_weight)


                    yield padded_image[y:y + self.window_height, x:x + self.window_width], label[:5], [label[5]]


    @threadsafe_generator
    def create_sliding_window_from_iter(self, image_iter):
        for i in image_iter:
            for j in self.create_sliding_window_samples(i["image"], i["label"]):
                yield j


EnqueueThread = namedtuple("EnqueueThread",
    [ "pic_batch", "label_batch", "enqueue_op", "enq_image", "enq_label", "examples_in_queue", "queue_close_op"])


def get_area(rect : np.ndarray) -> float:
    """Computes the area of rect

    :param rect: expected in the form [x1,y1, x2,y2] i.e. [upper left corner, lower left corner]
    :return: area
    """
    dx = rect[2] - rect[0]
    dy = rect[3] - rect[1]
    return dx*dy


def crop_labels(labels : np.ndarray, window_width : int, window_height : int) -> list:
    """Crops label to given width and height

    WARNING: this function must not get any labels which are outside the window defined by
        window_width and window_height

    :param labels: List of labels in bounding box representation
    :param window_width:
    :param window_height:
    :return: cropped label
    """
    def crop_x(x):
        return min(max(x, 0), window_width)
    def crop_y(y):
        return min(max(y, 0), window_height)

    res = []
    for l in labels:
        cropped_label = np.array([crop_x(l[0]), crop_y(l[1]), crop_x(l[2]), crop_y(l[3])]).astype(np.float)
        visible_ratio = get_area(cropped_label) / get_area(l)
        # that '1' denotes that this sliding window contains - at least partially - a license plate
        res.append(np.append(cropped_label, [1, visible_ratio]))

    return res


def shift_labels(labels : np.ndarray, x : int, y : int) -> list:
    return [[l[0] - x, l[1] - y, l[2] - x, l[3] - y] for l in labels]


def shift_and_crop_labels(labels : np.ndarray, x : int, y : int, window_width : int, window_height : int) -> np.ndarray:
    return crop_labels(shift_labels(labels, x,y), window_width, window_height)


def filter_and_transform_labels(labels : np.ndarray, x : int, y : int, window_width : int, window_height : int) \
        -> np.ndarray:
    """Other than filtering and transforming the label into the current sliding window, it also adds 2 float to
    the array:
    - 1 if license plate is inside the current sliding window, 0 if not
    - ratio of the license plate which is inside the current sliding window

    :param labels: list of
    :param x:
    :param y:
    :param window_width:
    :param window_height:
    :return: [x1,y1, x2,y2, C] - if there is a label inside the window, C will be 1, otherwise C and all coordinates
        will be 0
    """
    filtered_labels = [l for l in labels if isLabelInWindow(l, x, y, window_width, window_height)]
    if len(filtered_labels) > 0:
        # it is expected that at most one label remains in the picture
        return shift_and_crop_labels(labels, x, y, window_height, window_width)[0]
    else:
        return [0, 0, 0, 0, 0, 0]


def isLabelInWindow(label : np.ndarray, x : int, y : int, window_width : int, window_height : int) -> bool:
    #print("Label 0: %s" % label[0])
    # currying
    def isInWindow(p_x : int, p_y : int):
        return isPointInWindow(p_x, p_y, x, y, window_width, window_height)

    return isInWindow(label[0], label[1]) or isInWindow(label[0], label[3]) or \
           isInWindow(label[2], label[1]) or isInWindow(label[2], label[3])

def isPointInWindow(x: int, y : int, window_x : int, window_y : int, window_width : int, window_height : int):
    return (window_x < x < window_x + window_width) and \
           (window_y < y < window_y + window_height)


def plotImagesWithLabels(images_with_labels : np.ndarray, normalized = False):
    #numOfKernels = np.shape(images)[0]
    images = []
    labels = []
    for (im, la, _) in images_with_labels:
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

"""
def readOnTheFly(batch_size : int):
    enq_image = tf.placeholder(tf.float32, shape=[SIZE_X, SIZE_Y, 1])
    enq_label = tf.placeholder(tf.float32, shape=[MAX_NUMBER_OF_LABELS, 5])

    q = tf.RandomShuffleQueue(
        capacity=MIN_QUEUE_EXAMPLES + (NUM_PREPROCESS_THREADS) * FLAGS.batch_size,
        min_after_dequeue=MIN_QUEUE_EXAMPLES + FLAGS.batch_size,
        dtypes=[tf.float32, tf.float32],
        shapes=[[SIZE_X, SIZE_Y, 1], [MAX_NUMBER_OF_LABELS, 5]]
    )

    enqueue_op = q.enqueue([enq_image, enq_label])
    examples_in_queue = q.size()
    queue_close_op = q.close(cancel_pending_enqueues=True)
    image_batch_queue, label_batch_queue = q.dequeue_many(batch_size)

    return image_batch_queue, label_batch_queue, enqueue_op, enq_image, enq_label, examples_in_queue
"""

class InputThread(object):
    def __init__(self, net_config : NetworkConfig, env_config : EnvConfig):
        self.enq_image = tf.placeholder(tf.float32, shape=[net_config.window_height, net_config.window_width, 1])
        self.enq_label = tf.placeholder(tf.float32, shape=[net_config.output_dim, net_config.max_label_num])
        self.enq_loss_multiplier = tf.placeholder(tf.float32, shape=[1])
        self.net_config = net_config
        self.env_config = env_config

        q = tf.RandomShuffleQueue(
            capacity=env_config.min_queue_examples + env_config.number_of_input_threads * net_config.batch_size,
            min_after_dequeue=env_config.min_queue_examples + net_config.batch_size,
            dtypes=[tf.float32, tf.float32, tf.float32],
            shapes=[[net_config.window_height, net_config.window_width, 1],
                    [net_config.output_dim, net_config.max_label_num],
                    1]
        )

        self.enqueue_op = q.enqueue([self.enq_image, self.enq_label, self.enq_loss_multiplier])
        self.examples_in_queue = q.size()
        self.queue_close_op = q.close(cancel_pending_enqueues=True)
        self.image_batch_queue, self.label_batch_queue, self.enq_loss_batch_queue = q.dequeue_many(net_config.batch_size)
        self.coord = tf.train.Coordinator()
        self.threads = []

    def enqueue_thread(self, coord, sample_iterator, sess):
        while not coord.should_stop():
            image, label, loss_multiplier = next(sample_iterator)

            if len(image.shape) < 3:
                image = image.reshape(image.shape[0], image.shape[1], 1)

            label = np.array(label).reshape(5,1)

            try:
                sess.run(self.enqueue_op,
                         feed_dict={
                             self.enq_image: image,
                             self.enq_label: label,
                             self.enq_loss_multiplier: loss_multiplier})
            except tf.errors.CancelledError:
                return

    def start_threads(self, number_of_threads : int, sample_iterator, sess) -> List[threading.Thread]:
        for _ in range(number_of_threads):
            # print("Creating thread %i" % i)
            t = threading.Thread(target=self.enqueue_thread, args=(
                self.coord,
                sample_iterator,
                sess
            ))

            t.setDaemon(True)
            t.start()
            self.threads.append(t)
            self.coord.register_thread(t)
            time.sleep(0.5)

        num_examples_in_queue = sess.run(self.examples_in_queue)
        while num_examples_in_queue < self.env_config.min_queue_examples:
            num_examples_in_queue = sess.run(self.examples_in_queue)
            for t in self.threads:
                if not t.isAlive():
                    self.coord.request_stop()
                    raise ValueError("One or more enqueuing threads crashed...")
            time.sleep(0.1)

        print("# of examples in queue: %i" % num_examples_in_queue)

        return self.threads

    def request_stop(self):
        self.coord.request_stop()
        self.coord.join(self.threads, ignore_live_threads=True)
