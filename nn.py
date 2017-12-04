import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np
import scipy
from inspect_images import get_bounding_box
import time
from datetime import datetime
from os import listdir
from os.path import isfile, join
import random
import sys

FLAGS = tf.app.flags.FLAGS

# for ipython compatibility, because if 'autoreload' re-imports the file,
# it will duplicate these definitions
if "batch_size" not in tf.app.flags.FLAGS.__flags.keys():
    tf.app.flags.DEFINE_integer('batch_size', 10,
                                        "Number of images to process in a batch.")
    tf.app.flags.DEFINE_string('data_dir', './samples',
                                       "Path to the CIFAR-10 data directory.")
    tf.app.flags.DEFINE_boolean('use_fp16', False,
                                        "Train the model using fp16.")
    tf.app.flags.DEFINE_boolean('log_device_placement', False,
                                """Whether to log device placement.""")
    tf.app.flags.DEFINE_integer('log_frequency', 10,
                                """How often to log results to the console.""")
    tf.app.flags.DEFINE_integer('max_steps', 1000000,
                                """Number of batches to run.""")
    tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoints',
                                       "Path to dir where checkpoints are stored")
    tf.app.flags.DEFINE_boolean('debug', False, "Use debug mode")

# dimensions of the input image
SIZE_X = 256
SIZE_Y = 256

WEIGHT_DECAY = 0.0

RANDOM_SEED = 32122
#tf.set_random_seed = RANDOM_SEED

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 500
NUM_EPOCHS_PER_DECAY = 10.0
LEARNING_RATE_DECAY_FACTOR = 0.95
INITIAL_LEARNING_RATE = 0.00002
NUM_PREPROCESS_THREADS = 1
MIN_QUEUE_EXAMPLES = 0
TRAINING_SET_RATIO = 0.8    # this ratio of the inputs will be used for training


def progress_bar(count, total, prefix='', suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    if count != total:
        sys.stdout.write('%s[%s] %s%s ...%s\r' % (prefix, bar, percents, '%', suffix))
    else:
        print('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
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


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var


def create_layer(image, shape : np.ndarray, initialial_value : float, scope_name):
    with tf.variable_scope(scope_name) as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=shape,
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(image, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(initialial_value))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        return conv1


OUTPUT_DIM=8


def inference(image):
    layer1 = create_layer(image, [5,5,1,64], 0.0, "conv1")
    pool1 = tf.nn.max_pool(layer1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name="pool1")
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name="norm1")
    layer2 = create_layer(norm1, [5,5,64,64], 0.1, "conv2")
    pool2 = tf.nn.max_pool(layer2, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name="pool2")
    norm2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name="norm2")

    with tf.variable_scope('local3') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                          stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        dropped_local3 = tf.nn.dropout(local3, 0.8)
 
    # local4
    with tf.variable_scope('local4') as scope:
        weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(dropped_local3, weights) + biases, name=scope.name)
        dropped_local4 = tf.nn.dropout(local4, 0.8)

    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights', [192, OUTPUT_DIM],
                                              stddev=1/192.0, wd=0.0)
        biases = _variable_on_cpu('biases', [OUTPUT_DIM],
                                  tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(dropped_local4, weights), biases, name=scope.name)
        dropped_softmax = tf.nn.dropout(softmax_linear, 0.9)

    return tf.reshape(dropped_softmax, [FLAGS.batch_size, 4,2])


def _loss(logits, label):
    mean_squared_error = tf.losses.mean_squared_error(labels=label, predictions=logits)
    tf.add_to_collection("losses", mean_squared_error)
    tf.summary.scalar("loss", mean_squared_error)
    return mean_squared_error


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 16
  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

  # Display the training images in the visualizer.
  tf.summary.image('images', images)

  return images, tf.reshape(label_batch, [batch_size])


def train(total_loss, global_step):
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)

    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    with tf.control_dependencies([apply_gradient_op]):
        train_op = tf.no_op(name='train')

    return train_op


def resize_label(label : np.ndarray, width : int, height : int):
    newlabel = np.reshape(label.copy(), [8])
    for i in range(0,8,2):
        newlabel[i] = int(newlabel[i] * SIZE_X/float(width))
    for i in range(1,8,2):
        newlabel[i] = int(newlabel[i] * SIZE_Y/float(height))

    return np.reshape(newlabel, [4,2])


def get_image_list(sample_folder : dir):
    res = []
    for f in listdir(sample_folder):
        if isfile(join(sample_folder, f)):
            if f.endswith(".jpg"): 
                res.append(f)

    return res


def preprocess(image : np.ndarray, label : np.ndarray):
    im_shape = np.shape(image)
    image = scipy.misc.imresize(image, [SIZE_X, SIZE_Y])
    image = np.reshape(image, [SIZE_X, SIZE_Y, 1])

    label = resize_label(label, im_shape[1], im_shape[0])

    return image, label


def eval_on_pic(picloc : dir):
    tf.reset_default_graph()
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()


        images, labels = (0,0)
        
        images = []
        labels = []

        pic = scipy.ndimage.imread(picloc, mode="L")

        fullpicloc = picloc
        
        image = scipy.ndimage.imread(fullpicloc, mode="L")
        label = get_bounding_box(fullpicloc)

        image, label = preprocess(image, label)
        float_image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        
        images, labels = tf.train.shuffle_batch(
                [float_image, label],
                batch_size=FLAGS.batch_size,
                num_threads=NUM_PREPROCESS_THREADS,
                capacity=MIN_QUEUE_EXAMPLES + 3 * FLAGS.batch_size,
                min_after_dequeue=MIN_QUEUE_EXAMPLES)           
        """
        images.append(float_image)
        labels.append(label)
        """

        images = tf.convert_to_tensor(images, dtype=tf.float32)
        labels = np.asarray(labels)
        logits = inference(images)
        print("logits: %s" % logits)
        loss = _loss(logits,labels)

        train_op = train(loss, global_step)
        saver = tf.train.Saver()

        with tf.Session().as_default() as sess:
            if FLAGS.debug:
                sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            saver.restore(sess, join(FLAGS.checkpoint_dir, "model.ckpt-1500"))
            print("model restored")
            inferenced_label = sess.run(logits)
            res = sess.run(loss)
            res_images = np.reshape(sess.run(images), [1, SIZE_X, SIZE_Y])

        return res_images, labels, inferenced_label, res


def eval_network(picloc : dir, eval_size = 10):
    tf.reset_default_graph()
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()


        pics = get_image_list(picloc)
        images, labels = (0,0)
        
        #random.seed(RANDOM_SEED)
        random.shuffle(pics)
        training_set_size = int(TRAINING_SET_RATIO*len(pics))
        validation_pics = pics[training_set_size:]
        images = []
        labels = []

        for i in range(FLAGS.batch_size):
            pic = validation_pics[i]
            #bar.update(i+1)
            progress_bar(i+1, eval_size)

            fullpicloc = join(picloc, pic)
            
            image = scipy.ndimage.imread(fullpicloc, mode="L")
            label = get_bounding_box(fullpicloc)

            image, label = preprocess(image, label)
            float_image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            
            """
            images, labels = tf.train.shuffle_batch(
                    [float_image, label],
                    batch_size=FLAGS.batch_size,
                    num_threads=NUM_PREPROCESS_THREADS,
                    capacity=MIN_QUEUE_EXAMPLES + 3 * FLAGS.batch_size,
                    min_after_dequeue=MIN_QUEUE_EXAMPLES)           
            """
            images.append(float_image)
            labels.append(label)

        images = tf.convert_to_tensor(images, dtype=tf.float32)
        labels = np.asarray(labels)
        logits = inference(images)
        print("logits: %s" % logits)
        loss = _loss(logits,labels)

        train_op = train(loss, global_step)
        saver = tf.train.Saver()

        with tf.Session().as_default() as sess:
            if FLAGS.debug:
                sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            saver.restore(sess, join(FLAGS.checkpoint_dir, "model.ckpt-10"))
            print("model restored")
            inferenced_label = sess.run(logits)
            res = sess.run(loss)
            res_images = np.reshape(sess.run(images), [eval_size, SIZE_X, SIZE_Y])

        return res_images, labels, inferenced_label, res


def get_samples():
    pics = get_image_list(FLAGS.data_dir)
    images, labels = (0,0)

    random.seed(RANDOM_SEED)
    training_set_size = int(TRAINING_SET_RATIO*len(pics))
    training_pics = pics[:training_set_size]
    #training_pics = pics

    i = 0

    images = []
    labels = []
    #for i in range(len(training_pics)):
    while True:
        seed = random.randint(0,10000)
        random.seed(seed)
        random.shuffle(training_pics)
        for i in range(len(training_pics)):
            pic = training_pics[i]
            progress_bar(i+1, len(training_pics))

            fullpicloc = join(FLAGS.data_dir, pic)
            
            image = scipy.ndimage.imread(fullpicloc, mode="L")
            label = get_bounding_box(fullpicloc)

            image, label = preprocess(image, label)
            yield image, label
            #images.append(image)
            #labels.append(label)

    #return { "image" : images, "label" : labels }



def train_on_lots_of_pics(picloc : dir):
    tf.reset_default_graph()
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()

        images = tf.placeholder(tf.int8, shape=[FLAGS.batch_size, SIZE_X, SIZE_Y, 1])
        labels = tf.placeholder(tf.int32, shape=[FLAGS.batch_size, 4, 2])
        float_images = tf.image.convert_image_dtype(images, dtype=tf.float32)

        """
        images, labels = tf.train.shuffle_batch(
                [float_image, label],
                batch_size=FLAGS.batch_size,
                num_threads=NUM_PREPROCESS_THREADS,
                capacity=MIN_QUEUE_EXAMPLES + 3 * FLAGS.batch_size,
                min_after_dequeue=MIN_QUEUE_EXAMPLES)           
        """

        logits = inference(float_images)
        loss = _loss(logits,labels)

        train_op = train(loss, global_step)

        #dataset = tf.contrib.data.Dataset.from_generator(get_samples, (tf.int8, tf.int32))
        #value = ds.make_one_shot_iterator().get_next()

        class _LoggerHook(tf.train.SessionRunHook):
            """Logs loss and runtime."""

            def begin(self):
              self._step = -1
              self._start_time = time.time()

            def before_run(self, run_context):
                self._step += 1
                return tf.train.SessionRunArgs(loss)  # Asks for loss value.
  
            def after_run(self, run_context, run_values):
                if self._step % FLAGS.log_frequency == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time
      
                    loss_value = run_values.results
                    examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                    sec_per_batch = float(duration / FLAGS.log_frequency)
      
                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print (format_str % (datetime.now(), self._step, loss_value,
                                               examples_per_sec, sec_per_batch))

        samples = get_samples()

        with tf.train.MonitoredTrainingSession(
            checkpoint_dir=FLAGS.checkpoint_dir,
            hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                   tf.train.NanTensorHook(loss),
                   _LoggerHook()],
            config=tf.ConfigProto(
                log_device_placement=FLAGS.log_device_placement)) as mon_sess:
            #if FLAGS.debug:
            #    mon_sess = tf_debug.LocalCLIDebugWrapperSession(tf.Session().as_default())
            while not mon_sess.should_stop():
                in_images = []
                in_labels = []
                for i in range(FLAGS.batch_size):
                    in_image, in_label = next(samples)
                    in_images.append(in_image)
                    in_labels.append(in_label)
                mon_sess.run(train_op, feed_dict={images: in_images, labels: in_labels})
