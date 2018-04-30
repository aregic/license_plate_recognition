import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np
from inspect_images import get_bounding_box, get_bounding_polygon
import time
from datetime import datetime
import random
import sys
import threading

from preprocessor import *
from gaborwavelet import getBasicKernels, orthonormalInit

FLAGS = tf.app.flags.FLAGS

# for ipython compatibility, because if 'autoreload' re-imports the file,
# it would duplicate these definitions
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
                                """Whether to log device placement.""")
    tf.app.flags.DEFINE_integer('log_frequency', 20,
                                """How often to log results to the console.""")
    tf.app.flags.DEFINE_integer('max_steps', 1000000,
                                """Number of batches to run.""")
    tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoints',
                                       "Path to dir where checkpoints are stored")
    tf.app.flags.DEFINE_integer('save_frequency', 400,
                                """Number of batches to run.""")
    tf.app.flags.DEFINE_boolean('debug', False, "Use debug mode")
    tf.app.flags.DEFINE_boolean('detailed_log', True, "Add pictures to log")

# dimensions of the input image
#SIZE_X = 512
#SIZE_Y = 512

# number of tiles for the YOLO architecture
TILE_NUMBER_X = 9
TILE_NUMBER_Y = 13

# sizes must be multiples of the respective TILE_NUMBER_[X|Y]s!
SIZE_X = 512 # tried: 256
SIZE_Y = 512 # tried: 256

MAX_NUMBER_OF_LABELS = 5
BOUNDING_BOX_PER_CELL = 1

# network is expected to ouput a bounding box which has 4 coordinates (2 vertices)
OUTPUT_DIM=4


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


def eval_on_pic(picloc : dir):
    tf.reset_default_graph()
    preprocessor = Preprocessor(SIZE_X, SIZE_Y)
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()


        images, labels = (0,0) # ??? TODO remove if everything works
        
        images = []
        labels = []

        pic = scipy.ndimage.imread(picloc, mode="L")

        fullpicloc = picloc
        
        image = scipy.ndimage.imread(fullpicloc, mode="L")
        label = get_bounding_box(fullpicloc)

        image, label = preprocessor.preprocess(image, label)
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




class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.it)


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


@threadsafe_generator
def get_samples():
    preprocessor = Preprocessor(SIZE_X, SIZE_Y)
    pics = get_image_list(FLAGS.data_dir)
    images, labels = (0,0)

    random.seed(RANDOM_SEED)
    training_set_size = int(TRAINING_SET_RATIO*len(pics))
    training_pics = pics[:training_set_size]
    #training_pics = pics

    samples = []
    for i in range(len(training_pics)):
        picloc = training_pics[i]
        progress_bar(i+1, len(training_pics), "Loading images: ")
        fullpicloc = join(FLAGS.data_dir, picloc)
        pic = scipy.ndimage.imread(fullpicloc, mode="L")
        # The line below is for when the label files contain bounding polygons and
        # not bounding boxes (the latter only provides 2 vertices: top-left and bottom-right corners).
        #label = get_bounding_box(fullpicloc)

        # despite the name, this one works for bounding boxes too.
        # TODO: fix this
        label = get_bounding_polygon(fullpicloc)
        
        pic,label = preprocessor.preprocess(pic, label)

        samples.append({ "pic" : pic, "label" : label})


    i = 0

    #for i in range(len(training_pics)):
    while True:
        seed = random.randint(0,10000)
        random.seed(seed)
        random.shuffle(samples)
        for i in range(len(samples)):
            """
            pic = training_pics[i]
            progress_bar(i+1, len(training_pics))

            fullpicloc = join(FLAGS.data_dir, pic)
            
            image = scipy.ndimage.imread(fullpicloc, mode="L")
            label = get_bounding_box(fullpicloc)

            image, label = preprocess(image, label)
            yield image, label
            """
            #images.append(image)
            #labels.append(label)
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


def readOnTheFly():
    enq_image = tf.placeholder(tf.float32, shape=[SIZE_X, SIZE_Y, 1])
    enq_label = tf.placeholder(tf.float32, shape=[MAX_NUMBER_OF_LABELS, 2, 2])

    q = tf.RandomShuffleQueue(
        capacity=MIN_QUEUE_EXAMPLES + (NUM_PREPROCESS_THREADS) * FLAGS.batch_size,
        min_after_dequeue=MIN_QUEUE_EXAMPLES + FLAGS.batch_size,
        dtypes=[tf.float32, tf.float32],
        shapes=[[SIZE_X, SIZE_Y, 1], [MAX_NUMBER_OF_LABELS, 2, 2]]
    )

    enqueue_op = q.enqueue([enq_image, enq_label])
    examples_in_queue = q.size()
    queue_close_op = q.close(cancel_pending_enqueues=True)
    image_batch_queue, label_batch_queue = q.dequeue_many(FLAGS.batch_size)

    return image_batch_queue, label_batch_queue, enqueue_op, enq_image, enq_label, examples_in_queue


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


def train_on_lots_of_pics(dataset_file : dir, train_on_tiles = False, use_dataset = False):
    tf.reset_default_graph()
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()

        if use_dataset:
            pic_batch, label_batch = readFromDataSet(dataset_file, train_on_tiles)
        else:
            pic_batch, label_batch, enqueue_op, enq_image, enq_label, examples_in_queue = readOnTheFly()

        logits, tiles = inference(pic_batch)

        if train_on_tiles:
            # if dataset is created with train_on_tiles is True, label_batch will actually
            # contain the tile matrix...
            # TODO fix this
            if FLAGS.detailed_log:
                tf.summary.image(
                        "tile label",
                        tf.reshape(label_batch, [FLAGS.batch_size, TILE_NUMBER_X, TILE_NUMBER_Y, 1]))
                alpha_cut = np.full([FLAGS.batch_size, TILE_NUMBER_X, TILE_NUMBER_Y], ALPHA_CUT, dtype=np.float32)
                tf.summary.image(
                        "tile output",
                        tf.cast(
                            tf.reshape(tf.greater_equal(tiles, tf.convert_to_tensor(alpha_cut)),
                                [FLAGS.batch_size, TILE_NUMBER_X, TILE_NUMBER_Y, 1]),
                            dtype=tf.float32))

            loss = _loss(tiles,label_batch)
            train_op = train(loss, global_step)

        else:
                

            loss = _loss(logits,label_batch)
            train_op = train(loss, global_step)

        coord = tf.train.Coordinator()

        if not use_dataset:
            samples_iter = get_samples()

        with tf.Session().as_default() as sess:
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())

            if not use_dataset:
                threads = []
                for i in range(NUM_PREPROCESS_THREADS):
                    
                    #print("Creating thread %i" % i)
                    t = threading.Thread(target=load_preproc_enqueue_thread, args=(
                        sess, coord, enqueue_op, enq_image, enq_label, samples_iter
                    ))

                    t.setDaemon(True)
                    t.start()
                    threads.append(t)
                    coord.register_thread(t)
                    time.sleep(0.5)

                num_examples_in_queue = sess.run(examples_in_queue)
                while num_examples_in_queue < MIN_QUEUE_EXAMPLES:
                    num_examples_in_queue = sess.run(examples_in_queue)
                    for t in threads:
                        if not t.isAlive():
                            coord.request_stop()
                            raise ValueError("One or more enqueuing threads crashed...")
                    time.sleep(0.1)

                print("# of examples in queue: %i" % num_examples_in_queue)

            # Create a coordinator and run all QueueRunner objects
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            last_time = time.time()
            saver = tf.train.Saver()
            #saver = tf.train.import_meta_graph(join(FLAGS.checkpoint_dir, "my_model.meta"))
            checkpoint_file = join(FLAGS.checkpoint_dir, "my_model")
            if isfile(join(FLAGS.checkpoint_dir, "checkpoint")):
                #saver.restore(sess,tf.train.latest_checkpoint(FLAGS.checkpoint_dir))
                optimistic_restore(sess, checkpoint_file)
                initialize_uninitialized_vars(sess)
                print("Model loaded.")

            sum_writer = tf.summary.FileWriter(FLAGS.log_dir, graph=tf.get_default_graph())
            merged = tf.summary.merge_all()
            for i in range(FLAGS.max_steps):
                if (i % FLAGS.save_frequency) != 0 or i == 0:
                    if (i % FLAGS.log_frequency) != 0:
                        sess.run([loss, train_op])
                    #print("Number of items in queue: %i" % queued_size)
                    else:
                        if train_on_tiles:
                            loss_value, res_tiles, res_label_tiles, _ = sess.run([loss, tiles, label_batch, train_op])
                            tilenum = np.count_nonzero(res_label_tiles)
                            positive_tiles = getPositiveTiles(res_tiles, ALPHA_CUT)
                            #print("Positive tiles: %s" % positive_tiles)
                            missed = np.count_nonzero(res_label_tiles - positive_tiles)
                            act_time = time.time()
                            exec_time = act_time - last_time
                            samples_per_sec = FLAGS.batch_size * FLAGS.log_frequency / exec_time
                            print("Step %i, loss: %f, execution time: %.4f, samples/second: %.4f" 
                                    % (i, loss_value, exec_time, samples_per_sec) )
                            print("Tiles in label: %i, tile in output: %i, missed: %i" % 
                                    (tilenum, np.count_nonzero(positive_tiles), missed))
                        else:
                            loss_value, _ = sess.run([loss, train_op])
                            act_time = time.time()
                            exec_time = act_time - last_time
                            samples_per_sec = FLAGS.batch_size * FLAGS.log_frequency / exec_time
                            print("Step %i, loss: %f, execution time: %.4f, samples/second: %.4f" 
                                    % (i, loss_value, exec_time, samples_per_sec) )
 
                        last_time = time.time()
                else:
                    sys.stdout.write("Saving model... ")
                    sys.stdout.flush()
                    summary, loss_value, _ = sess.run([merged, loss, train_op])
                    sum_writer.add_summary(summary, i)
                    saver.save(sess, join(FLAGS.checkpoint_dir, "my_model"))
                    print("saved.")
                    if (i % FLAGS.log_frequency) == 0:
                        act_time = time.time()
                        exec_time = act_time - last_time
                        samples_per_sec = FLAGS.batch_size * FLAGS.log_frequency / exec_time
                        print("Step %i, loss: %f, execution time: %.4f, samples/second: %.4f" 
                                % (i, loss_value, exec_time, samples_per_sec) )
                        last_time = time.time()

            coord.request_stop()
            coord.join(threads)


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


        """
        with tf.train.MonitoredTrainingSession(
            checkpoint_dir=FLAGS.checkpoint_dir,
            hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                   tf.train.NanTensorHook(loss),
                   _LoggerHook()],
            config=tf.ConfigProto(
                log_device_placement=FLAGS.log_device_placement)) as mon_sess:
            #if FLAGS.debug:
            #    mon_sess = tf_debug.LocalCLIDebugWrapperSession(tf.Session().as_default())
            print("Training started...")
            while not mon_sess.should_stop():
                queued_size = mon_sess.run(examples_in_queue)
                print("Queued examples: %i" %queued_size)
                print("Inner cycle")
                mon_sess.run(train_op)
        """


def eval_network(train_on_tiles = False, use_dataset = False):
    tf.reset_default_graph()
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()

        if use_dataset:
            pic_batch, label_batch = readFromDataSet(dataset_file, train_on_tiles)
        else:
            pic_batch, label_batch, enqueue_op, enq_image, enq_label, examples_in_queue = readOnTheFly()

        logits, tiles = inference(pic_batch)

        if train_on_tiles:
            # if dataset is created with train_on_tiles is True, label_batch will actually
            # contain the tile matrix...
            # TODO fix this
            if FLAGS.detailed_log:
                tf.summary.image(
                        "tile label",
                        tf.reshape(label_batch, [FLAGS.batch_size, TILE_NUMBER_X, TILE_NUMBER_Y, 1]))
                alpha_cut = np.full([FLAGS.batch_size, TILE_NUMBER_X, TILE_NUMBER_Y], ALPHA_CUT, dtype=np.float32)
                tf.summary.image(
                        "tile output",
                        tf.cast(
                            tf.reshape(tf.greater_equal(tiles, tf.convert_to_tensor(alpha_cut)),
                                [FLAGS.batch_size, TILE_NUMBER_X, TILE_NUMBER_Y, 1]),
                            dtype=tf.float32))

            loss = _loss(tiles,label_batch)
            train_op = train(loss, global_step)

        else:
                

            loss = _loss(logits,label_batch)
            train_op = train(loss, global_step)

        coord = tf.train.Coordinator()

        if not use_dataset:
            samples_iter = get_samples()

        with tf.Session().as_default() as sess:
            threads = []
            for i in range(NUM_PREPROCESS_THREADS):
                
                t = threading.Thread(target=load_preproc_enqueue_thread, args=(
                    sess, coord, enqueue_op, enq_image, enq_label, samples_iter
                ))

                t.setDaemon(True)
                t.start()
                threads.append(t)
                coord.register_thread(t)
                time.sleep(0.5)

            num_examples_in_queue = sess.run(examples_in_queue)
            while num_examples_in_queue < MIN_QUEUE_EXAMPLES:
                num_examples_in_queue = sess.run(examples_in_queue)
                for t in threads:
                    if not t.isAlive():
                        coord.request_stop()
                        raise ValueError("One or more enqueuing threads crashed...")
                time.sleep(0.1)

            print("# of examples in queue: %i" % num_examples_in_queue)
            sess.run(tf.global_variables_initializer())

            last_time = time.time()
            saver = tf.train.Saver()
            #saver = tf.train.import_meta_graph(join(FLAGS.checkpoint_dir, "my_model.meta"))
            if isfile(join(FLAGS.checkpoint_dir, "checkpoint")):
                saver.restore(sess,tf.train.latest_checkpoint(FLAGS.checkpoint_dir))

            print("Running model...")
            loss_value, res_images, res_labels, orig_labels, _ = (
                sess.run([loss, pic_batch, logits, label_batch, train_op]))
            print("Done.")
            act_time = time.time()
            exec_time = act_time - last_time
            samples_per_sec = FLAGS.batch_size * FLAGS.log_frequency / exec_time
            print("Step %i, loss: %f, execution time: %.4f, samples/second: %.4f" 
                    % (i, loss_value, exec_time, samples_per_sec) )
            last_time = time.time()

        coord.request_stop()

        return res_images, res_labels, loss_value, orig_labels
