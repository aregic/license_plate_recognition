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

RANDOM_SEED = 854654

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

    pad_needed = tf.maximum(1,size_x-x)
    x_asd = tf.random_uniform(minval=0, maxval=pad_needed, shape=[1], seed=RANDOM_SEED, dtype=tf.int32)
    x_pad_before = tf.Session().run(x_asd)
    asd = x_pad_before[0]

    def pad_x():

        padding_tensor = [[asd, size_x-x-asd], [0,0], [0,0]]
        tf.Print(padding_tensor, [padding_tensor], "padding_tensor: ")
        x_padded_image = tf.pad(image, padding_tensor)

        shifted_label = label
        shift_tensor = [0,asd,0,asd,0,asd,0,asd]
        """
        shifted_label[1] += asd
        shifted_label[3] += asd
        shifted_label[5] += asd
        shifted_label[7] += asd
        """
        shifted_label = tf.add(shifted_label, shift_tensor)

        return (x_padded_image, shifted_label)

    def no_pad():
        return tf.convert_to_tensor(image), tf.convert_to_tensor(label)

    return tf.cond(x < size_x, pad_x, no_pad, name="pading_for_x", strict=True)


def tf_pad_image_y(image : np.ndarray, label : np.ndarray, size_x : int, size_y : int) -> (np.ndarray, np.ndarray) :
    shape = tf.shape(image)
    x = shape[0]
    y = shape[1]
    z = shape[2]

    pad_needed = tf.maximum(1,size_y-y)
    y_asd = tf.random_uniform(minval=0, maxval=pad_needed, shape=[1], seed=RANDOM_SEED, dtype=tf.int32)
    y_pad_before = tf.Session().run(y_asd)
    asd = y_pad_before[0]

    def pad_y():
        padding_tensor = [[0,0], [asd, size_y-y-asd], [0,0]]
        tf.Print(padding_tensor, [padding_tensor], "padding_tensor: ")
        #print("padding: %s" % y_pad_before.eval())
        y_padded_image = tf.pad(image, padding_tensor)

        shifted_label = label
        shift_tensor = [asd,0, asd,0, asd,0, asd,0]
        """
        shifted_label[0] += asd
        shifted_label[2] += asd
        shifted_label[4] += asd
        shifted_label[6] += asd
        """
        shifted_label = tf.add(shifted_label, shift_tensor)

        return (y_padded_image, shifted_label)

    def no_pad():
        return tf.convert_to_tensor(image), tf.convert_to_tensor(label)

    return tf.cond(y < size_y, pad_y, no_pad, name="padding_for_y", strict=True)




def tf_pad_image(image : np.ndarray, label : np.ndarray, size_x : int, size_y : int) -> (np.ndarray, np.ndarray) :
    padded_image, padded_label = tf_pad_image_x(image, label, size_x, size_y)
    padded_image, padded_label = tf_pad_image_y(padded_image, padded_label, size_x, size_y)
    return (padded_image, padded_label)

    #return tf_pad_image_y(
            #tf_pad_image_x(image, label, size_x, size_y), label, size_x, size_y)
    #return tf.image.resize_image_with_crop_or_pad(image, 0,0, size_x, size_y)


def scale_image(image : np.ndarray, label : np.ndarray, size_x : int, size_y : int) -> (np.ndarray, np.ndarray):
    float_image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    shape = tf.shape(image)

    scaled_label = label
    """
    scaled_label[0] *= size_y / shape[1]
    scaled_label[1] *= size_x / shape[0]
    scaled_label[2] *= size_y / shape[1]
    scaled_label[3] *= size_x / shape[0]
    scaled_label[4] *= size_y / shape[1]
    scaled_label[5] *= size_x / shape[0]
    scaled_label[6] *= size_y / shape[1]
    scaled_label[7] *= size_x / shape[0]
    """
    scale_tensor = [5,5,5,5,5,5,5,5]
    for i in range(0,8,2):
        #scaled_label[i] = tf.to_int32(scaled_label[i] * size_y / shape[1])
        scale_tensor[i] = size_y / shape[1]

    for i in range(1,8,2):
        #scaled_label[i] = tf.to_int32(scaled_label[i] * size_x / shape[0])
        scale_tensor[i] = size_x / shape[0]

    scaled_label = tf.to_int32(tf.multiply(tf.cast(scaled_label, tf.float64), tf.convert_to_tensor(scale_tensor)))

    scaled_image = tf.image.resize_images(float_image, [size_x, size_y],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return scaled_image, scaled_label


def preprocess_image(image : np.ndarray, label : np.ndarray, size_x : int, size_y : int) -> (np.ndarray, np.ndarray):
    padded_image, padded_label = tf_pad_image(image, label, size_x, size_y)
    #print("padded label: %s" % tf.Session().run(padded_label))
    scaled_image, scaled_label = scale_image(padded_image, padded_label, size_x, size_y)

    grayscale_image = tf.squeeze(tf.image.rgb_to_grayscale(scaled_image))

    print("this has been called")

    return grayscale_image, scaled_label
