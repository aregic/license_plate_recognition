import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS

"""
# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 128,
                                    "Number of images to process in a batch.")
tf.app.flags.DEFINE_string('data_dir', '/tmp/cifar10_data',
                                   "Path to the CIFAR-10 data directory.")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                                    "Train the model using fp16.")
"""
# TODO replace with tf flag
BATCH_SIZE = 128

# dimensions of the input image
SIZE_X = 512
SIZE_Y = 512

RANDOM_SEED = 3564

tf.set_random_seed = RANDOM_SEED

def pad_image(image : np.ndarray, size_x : int, size_y : int) -> np.ndarray :
    """
        Extends the picture with zeros if needed.
        Expects 2d array as input, i.e. black and white picture
    """
    # it is possible to do it with tensorflow but I find their methods quite lame,
    # so no point. In preprocessing, the bottleneck is the file reading anyway.
    x,y,z = np.shape(image)
    normed_image = image

    if x < size_x:
        normed_image = np.zeros((size_x, y, z))
        for i in range(x):
            for j in range(y):
                for k in range(z):
                    normed_image[i,j,k] = image[i,j,k]

    x,y,z = np.shape(normed_image)

    if y < size_y:
        normed_image2 = np.zeros((x, size_y, z))
        for i in range(x):
            for j in range(y):
                for k in range(z):
                    normed_image2[i,j,k] = normed_image[i,j,k]

    return normed_image2


def tf_pad_image_x(image : np.ndarray, label : np.ndarray, size_x : int, size_y : int) -> (np.ndarray, np.ndarray) :
    shape = tf.shape(image)
    x = shape[0]
    y = shape[1]
    z = shape[2]

    def pad_x():
        pad_needed = size_x-x
        padding = tf.Variable(tf.random_uniform([1], minval=0, maxval=pad_needed, 
                dtype = tf.int32, seed=RANDOM_SEED, name="x_pad_before"))
        padding_tensor = [[x_pad_before[0], size_x-x-x_pad_before[0]], [0,0], [0,0]]
        tf.Print(padding_tensor, [padding_tensor], "padding_tensor: ")
        x_padded_image = tf.pad(image, padding_tensor)

        shifted_label = label
        shifted_label[0] += x_pad_before[0]
        shifted_label[2] += x_pad_before[0]
        shifted_label[4] += x_pad_before[0]
        shifted_label[6] += x_pad_before[0]

        return (x_padded_image, shifted_label)

    return tf.cond(x < size_x, pad_x, lambda : (image, label))


def tf_pad_image_y(image : np.ndarray, label : np.ndarray, size_x : int, size_y : int) -> (np.ndarray, np.ndarray) :
    shape = tf.shape(image)
    x = shape[0]
    y = shape[1]
    z = shape[2]

    def pad_y():
        pad_needed = size_y-y
        y_pad_before = tf.Variable(tf.random_uniform([1], minval=0, maxval=pad_needed, 
                dtype = tf.int32, seed=RANDOM_SEED, name="y_pad_before"))
        padding_tensor = [[0,0], [y_pad_before[0], size_y-y-y_pad_before[0]], [0,0]]
        tf.Print(padding_tensor, [padding_tensor], "padding_tensor: ")
        #print("padding: %s" % y_pad_before.eval())
        y_padded_image = tf.pad(image, padding_tensor)

        shifted_label = label
        shifted_label[1] += y_pad_before[0]
        shifted_label[3] += y_pad_before[0]
        shifted_label[5] += y_pad_before[0]
        shifted_label[7] += y_pad_before[0]

        return (y_padded_image, shifted_label)

    return tf.cond(y < size_y, pad_y, lambda : (image,label))




def tf_pad_image(image : np.ndarray, label : np.ndarray, size_x : int, size_y : int) -> (np.ndarray, np.ndarray) :
    padded_image, padded_label = tf_pad_image_x(image, label, size_x, size_y)
    padded_image, padded_label = tf_pad_image_y(padded_image, padded_label, size_x, size_y)
    return (padded_image, padded_label)

    #return tf_pad_image_y(
            #tf_pad_image_x(image, label, size_x, size_y), label, size_x, size_y)
    #return tf.image.resize_image_with_crop_or_pad(image, 0,0, size_x, size_y)


def resize_image(image : np.ndarray, size_x : int, size_y : int) -> (np.ndarray, np.ndarray):
    padded_image = tf_pad_image(image, size_x, size_y)
