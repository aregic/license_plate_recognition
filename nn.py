import tensorflow as tf
import random
import sys
from etc import threadsafe_generator
from gaborwavelet import orthonormalInit

from preprocessor import *

FLAGS = tf.app.flags.FLAGS

# for ipython compatibility, because if 'autoreload' re-imports the file,
# it would duplicate these definitions
"""
if "batch_size" not in tf.app.flags.FLAGS.__flags.keys():
    tf.app.flags.DEFINE_integer('batch_size', 10,
                                        "Number of images to process in a batch.")
    tf.app.flags.DEFINE_string('data_dir', '/media/regic/2nd_SD/datasets/height_width', #'./samples',
                                       "Path to the data directory.")
    tf.app.flags.DEFINE_string('log_dir', './logs',
                                       "Path to the directory where the logs are stored.")
    tf.app.flags.DEFINE_boolean('use_fp16', False,
                                        "Train the model using fp16.")
    tf.app.flags.DEFINE_boolean('log_device_placement', False,
                                "Whether to log device placement.")
    tf.app.flags.DEFINE_integer('log_frequency', 20,
                                "How often to log results to the console.")
    tf.app.flags.DEFINE_integer('max_steps', 1000000,
                                "Number of batches to run."
    tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoints',
                                       "Path to dir where checkpoints are stored")
    tf.app.flags.DEFINE_integer('save_frequency', 400,
                                "Number of batches to run."
    tf.app.flags.DEFINE_boolean('debug', False, "Use debug mode")
    tf.app.flags.DEFINE_boolean('detailed_log', True, "Add pictures to log")
"""

# dimensions of the input image
#SIZE_X = 512
#SIZE_Y = 512

# number of tiles for the YOLO architecture
TILE_NUMBER_X = 5
TILE_NUMBER_Y = 5

# sizes must be multiples of the respective TILE_NUMBER_[X|Y]s!
SIZE_X = 400 # tried: 256
SIZE_Y = 400 # tried: 256

MAX_NUMBER_OF_LABELS = 5
BOUNDING_BOX_PER_CELL = 1

# network is expected to ouput a bounding box which has 4 coordinates (2 vertices)
OUTPUT_DIM = 4


WEIGHT_DECAY = 0.0

RANDOM_SEED = 32122
tf.set_random_seed = RANDOM_SEED

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 500
NUM_EPOCHS_PER_DECAY = 50.0
LEARNING_RATE_DECAY_FACTOR = 0.97
INITIAL_LEARNING_RATE = 1.0 #1e-2
MOMENTUM_LEARNING = 0.1
NUM_PREPROCESS_THREADS = 8
MIN_QUEUE_EXAMPLES = 20
TRAINING_SET_RATIO = 0.8    # this ratio of the inputs will be used for training
ALPHA_CUT = 0.5

def progress_bar(count, total, prefix='', suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    if count != total:
        sys.stdout.write('%s[%s] %s%s ...%s\r' % (prefix, bar, percents, '%', suffix))
    else:
        print('%s[%s] %s%s ...%s\r' % (prefix, bar, percents, '%', suffix))
    sys.stdout.flush()


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


def initialize_uninitialized_vars(sess):
    from itertools import compress
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([~(tf.is_variable_initialized(var)) \
                                   for var in global_vars])
    not_initialized_vars = list(compress(global_vars, is_not_initialized))

    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))

class SampleLoader:
    def __init__(self, size_x : int, size_y : int, max_number_of_labels : int, data_dir : dir):
        self.size_x = size_x
        self.size_y = size_y
        self.max_number_of_labels = max_number_of_labels
        self.data_dir = data_dir

    @threadsafe_generator
    def get_samples(self):
        preprocessor = Preprocessor(self.size_x, self.size_y, self.max_number_of_labels)
        pics = get_image_list(self.data_dir)
        images, labels = (0,0)

        random.seed(RANDOM_SEED)
        training_set_size = int(TRAINING_SET_RATIO*len(pics))
        training_pics = pics[:training_set_size]
        #training_pics = pics

        samples = []
        for i in range(len(training_pics)):
            picloc = training_pics[i]
            progress_bar(i+1, len(training_pics), "Loading images: ")
            fullpicloc = join(self.data_dir, picloc)
            pic = scipy.ndimage.imread(fullpicloc, mode="L")
            # The line below is for when the label files contain bounding polygons and
            # not bounding boxes (the latter only provides 2 vertices: top-left and bottom-right corners).
            #label = get_bounding_box(fullpicloc)

            # despite the name, this one works for bounding boxes too.
            # TODO: fix this
            labels = get_bounding_polygon(fullpicloc)
            midpoint_labels = []
            for l in labels:
                boundingBox = BoundingBox(*np.reshape(l, [4]))
                midpoint_labels.append(boundingBox.getMidPointRepr())

            pic,label = preprocessor.preprocess(pic, midpoint_labels)
            """
            try:
                pic,label = preprocessor.preprocess(pic, midpoint_labels)
            except AssertionError as error:
                print("Assertion error while processing picture %s" % picloc)
                print("Error: %s" % error)
                exit(-1)
            """

            samples.append({ "pic" : pic, "label" : label})

        #for i in range(len(training_pics)):
        while True:
            seed = random.randint(0,10000)
            random.seed(seed)
            random.shuffle(samples)
            for i in range(len(samples)):
                yield samples[i]["pic"], samples[i]["label"]

        #return { "image" : images, "label" : labels }


def load_preproc_enqueue_thread(sess, coord, enqueue_op, queue_images, queue_labels, sample_iterator):
    while not coord.should_stop():
        image, label = next(sample_iterator)

        try:
            sess.run(enqueue_op, feed_dict={queue_images: image, queue_labels: label})
        except tf.errors.CancelledError:
            return


def getPositiveTiles(tiles, alpha):
    """
        This function does an alpha cut, i.e. the tiles with value above or equal to alpha will be ones 
        and the tiles with value below will be zeros.
    """
    cut = [[[int(x >= alpha) for x in xx] for xx in xxx] for xxx in tiles]
    return np.asarray(cut)


def optimistic_restore(session, save_file):
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
            if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    name2var = dict(zip(map(lambda x:x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = name2var[saved_var_name]
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
    saver = tf.train.Saver(restore_vars)
    saver.restore(session, save_file)


"""
def readOnTheFly():
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
    image_batch_queue, label_batch_queue = q.dequeue_many(FLAGS.batch_size)

    return image_batch_queue, label_batch_queue, enqueue_op, enq_image, enq_label, examples_in_queue
"""

def readFromDataSet(dataset_file : dir, train_on_tiles : bool):
    feature = {'train/label': tf.FixedLenFeature([], tf.string),
               'train/image': tf.FixedLenFeature([], tf.string)}

    filename_queue = tf.train.string_input_producer([dataset_file], 
            #num_epochs=NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN,
            #shuffle=True)
            )
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features=feature)
    decoded_image = tf.decode_raw(features['train/image'], tf.float32)
    decoded_label = tf.decode_raw(features['train/label'], tf.float32)

    image = tf.reshape(decoded_image, [SIZE_X, SIZE_Y, 1])
    if train_on_tiles:
        label = tf.reshape(decoded_label, [TILE_NUMBER_X, TILE_NUMBER_Y])
    else:
        label = tf.reshape(decoded_label, [MAX_NUMBER_OF_LABELS, 2, 2])

    capacity=MIN_QUEUE_EXAMPLES + (NUM_PREPROCESS_THREADS) * FLAGS.batch_size
    min_after_dequeue=MIN_QUEUE_EXAMPLES + FLAGS.batch_size

    pic_batch, label_batch = tf.train.shuffle_batch(
        [image, label], batch_size=FLAGS.batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue, enqueue_many=False, num_threads=8)

    return pic_batch, label_batch


def create_layer(image : tf.Tensor, shape: List[int], initialial_value: float,
                 scope_name: str, dropout_rate: float = 0.0, leaky_alpha: float = 0.1,
                 const_init=None, batch_norm=False, show_tensor=False):
    if const_init is None:
        initializer = tf.constant(orthonormalInit(shape))
        # initializer = tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32) # stddev originally: 5e-2
    else:
        initializer = tf.constant(const_init)

    with tf.variable_scope(scope_name) as scope:
        kernel = variable_with_weight_decay('weights',
                                            shape=None,
                                            wd=None,
                                            initializer=initializer)

        conv = tf.nn.conv2d(image, kernel, [1, 1, 1, 1], padding='SAME')
        biases = variable_on_cpu('biases', shape[3], tf.constant_initializer(initialial_value))
        pre_activation = tf.nn.bias_add(conv, biases)
        # conv1 = tf.nn.relu(pre_activation, name=scope.name)
        # conv1 = tf.maximum(pre_activation, leaky_alpha * pre_activation, name="leaky_relu")

        conv1 = leakyRelu(pre_activation, leaky_alpha)

        if show_tensor:
            kernels = tf.transpose(kernel[:, :, 0, :], [2, 0, 1])
            tf.summary.image(
                "kernel",
                tf.reshape(kernels, [32, shape[0], shape[1], 1]),
                max_outputs=32)

        if batch_norm:
            batch_normed = tf.layers.batch_normalization(conv1, training=True, trainable=True)
        else:
            batch_normed = conv1

        print("Layer %s initalized." % scope_name)

        if dropout_rate != 0.0:
            dropped_conv1 = tf.nn.dropout(batch_normed, dropout_rate)
            return dropped_conv1
        else:
            return batch_normed


def leakyRelu(pre_activation, leaky_alpha: float = 0.1):
    return tf.maximum(pre_activation, leaky_alpha * pre_activation, name="leaky_relu")


def variable_with_weight_decay(name, shape, wd, initializer):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
      name: name of the variable
      shape: list of ints
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.

    Returns:
      Variable Tensor
    """
    var = variable_on_cpu(
        name,
        shape,
        initializer)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.

    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable

    Returns:
      Variable Tensor
    """
    # with tf.device('/cpu:0'):
    with tf.device('/gpu:0'):
        # dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        dtype = tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def normal_variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of the variable at initalization
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.

    Returns:
      Variable Tensor
    """
    initializer = tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32)

    return variable_with_weight_decay(name, shape, wd, initializer)