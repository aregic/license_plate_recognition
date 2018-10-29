import tensorflow as tf
from collections import namedtuple
from preprocessor import get_image_list, get_bounding_box
import scipy
import random
from etc import *

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
        buffer.append({"label" : label, "image" : image})

    random.seed(random_seed)
    random.shuffle(buffer)

    while True:
        for i in range(len(buffer)):
            yield buffer[i]

        seed = random.randint(0,10000)
        random.seed(seed)
        random.shuffle(buffer)




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