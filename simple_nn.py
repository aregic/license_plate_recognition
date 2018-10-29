import tensorflow as tf
from nn import *
import numpy as np
from gaborwavelet import getBasicKernels
from collections import namedtuple
import threading
import time

# it is only to define what configurations are needed
# it contains the configuration that belongs to the network and is independent
#   of the environment it is running in
# FIXME : the decay / sample depends on examples_per_epoch too in the EnvConfig
NetworkConfig = namedtuple("SimpleNetConfig",
        ["use_gabor_wavelet", "batch_size", "epoch_per_decay", "initial_learning_rate", "learning_decay",
         "learning_momentum", "model_name", "slide_x", "slide_y", "window_width", "window_height",
         "output_dim", "max_label_num"])

# configuration of the running environment. The variables here are performance related
#   (yes, technically batch_size should be here but that one is encoded in the network itself
#   so it is in NetworkConfig for now
EnvConfig = namedtuple("EnvConfig",
        ["examples_per_epoch", "number_of_input_threads", "max_number_of_steps", "log_frequency", "save_frequency",
         "checkpoint_dir", "sample_dir", "min_queue_examples", "log_dir"])

class SimpleNet:
    def __init__(self, network_config : NetworkConfig, environment_config : EnvConfig):
        # for typehinting
        class MyConfig(object):
            def __init__(self, network_config : NetworkConfig, environment_config : EnvConfig):
                self.network = network_config
                self.environment = environment_config

        self.config = MyConfig(network_config, environment_config)


    def inference(self, image):
        if self.config.network.use_gabor_wavelet:
            # init lower layer to predefined kernels (gabor wavelets, edge detectors, etc.)
            basic_kernels = np.asarray([getBasicKernels()]).astype(np.float32)

            # the swapping will probably transpose the kernels but that doesn't matter at the moment
            basic_kernels = np.swapaxes(basic_kernels, 0, 2)
            basic_kernels = np.swapaxes(basic_kernels, 1, 3)

            layer1 = create_layer(image, [7,7,1,32], 0.0, 5e-2, "conv1", const_init = basic_kernels)

        else:
            layer1 = create_layer(image, [3, 3, 1, 32], 0.0, 5e-2, "conv1", show_tensor=True)

        norm1 = tf.nn.lrn(layer1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name="norm1")
        layer2 = create_layer(norm1, [3, 3, 32, 32], 0.1, 5e-2, "conv2", dropout_rate=0.5)
        pool2 = tf.nn.max_pool(layer2, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='SAME', name="pool2")
        norm2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name="norm2")

        layer3 = create_layer(norm2, [3, 3, 32, 32], 0.1, 5e-2, "conv3", dropout_rate=0.5)
        pool3 = tf.nn.max_pool(layer3, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='SAME', name="pool3")
        norm3 = tf.nn.lrn(pool3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name="norm3")

        layer4 = create_layer(norm3, [3, 3, 32, 32], 0.1, 5e-2, "conv4", dropout_rate=0.5)
        norm4 = tf.nn.lrn(layer4, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name="norm4")
        layer5 = create_layer(norm4, [1, 1, 32, 32], 0.1, 3e-2, "conv5", dropout_rate=0.5)
        norm5 = tf.nn.lrn(layer5, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name="norm5")

        layer6 = create_layer(norm5, [3, 3, 32, 32], 0.1, 2e-2, "conv6", dropout_rate=0.5)
        pool6 = tf.nn.max_pool(layer6, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='SAME', name="pool6")
        norm6 = tf.nn.lrn(pool6, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name="norm6")

        with tf.variable_scope('local3') as scope:
            # Move everything into depth so we can perform a single matrix multiply.
            reshape = tf.reshape(norm6, [self.config.network.batch_size, -1])
            dim = reshape.get_shape()[1].value
            weights = normal_variable_with_weight_decay('weights', shape=[dim, 192],
                                                         stddev=0.04, wd=0.004)
            biases = variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
            # local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
            pre_act3 = tf.matmul(reshape, weights) + biases
            local3 = leakyRelu(pre_act3)
            bnormed_3 = tf.layers.batch_normalization(local3, training=True)
            # dropped_local3 = tf.nn.dropout(local3, 0.5)
            dropped_local3 = tf.nn.dropout(bnormed_3, 0.5)

        """
        # local4
        with tf.variable_scope('local4') as scope:
            weights = _normal_variable_with_weight_decay('weights', shape=[384, 192],
                                                  stddev=0.04, wd=0.004)
            biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
            #local4 = tf.nn.relu(tf.matmul(dropped_local3, weights) + biases, name=scope.name)
            pre_act4 = tf.matmul(dropped_local3, weights) + biases
            local4 = leakyRelu(pre_act4)
            dropped_local4 = tf.nn.dropout(local4, 0.8)
        """

        with tf.variable_scope('tile_output_layer') as scope:
            tile_weights = normal_variable_with_weight_decay(
                'weights',
                [192, self.config.network.output_dim * self.config.network.max_label_num],
                stddev=1 / 192.0, wd=0.0)
            tile_biases = variable_on_cpu('biases', [self.config.network.output_dim * self.config.network.max_label_num],
                                           tf.constant_initializer(0.0))
            tile_softmax_linear = tf.add(tf.matmul(dropped_local3, tile_weights), tile_biases, name=scope.name)
            bnormed_softmax = tf.layers.batch_normalization(tile_softmax_linear, training=True)
            tile_dropped_softmax = tf.nn.dropout(bnormed_softmax, 0.9)

            tile_result = tf.reshape(
                tile_dropped_softmax,
                [self.config.network.batch_size, self.config.network.output_dim, self.config.network.max_label_num])

        return tile_result


    def train(self, input_iterator : InputIterator):
        #size_x, size_y, max_label_num, label_size = input_iterator.getOutputDim()
        size_x = self.config.network.window_height
        size_y = self.config.network.window_width
        max_label_num = self.config.network.max_label_num
        label_size = self.config.network.output_dim


        tf.reset_default_graph()
        with tf.Graph().as_default():
            input_thread = InputThread(self.config.network, self.config.environment)
            global_step = tf.contrib.framework.get_or_create_global_step()

            logits = self.inference(input_thread.image_batch_queue)
            loss = tf.nn.l2_loss(logits - input_thread.label_batch_queue)
            tf.summary.scalar("loss", loss / self.config.network.batch_size)

            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(extra_update_ops):
                train_op, _ = self.train_op(loss, global_step)

            with tf.Session().as_default() as sess:
                sess.run(tf.local_variables_initializer())
                sess.run(tf.global_variables_initializer())

                input_thread.start_threads(self.config.environment.number_of_input_threads,
                                           input_iterator, sess)

                last_time = time.time()
                saver = tf.train.Saver()

                checkpoint_file = join(self.config.environment.checkpoint_dir, self.config.network.model_name)
                if isfile(join(self.config.environment.checkpoint_dir, "checkpoint")):
                    # saver.restore(sess,tf.train.latest_checkpoint(self.checkpoint_dir))
                    optimistic_restore(sess, checkpoint_file)
                    initialize_uninitialized_vars(sess)
                    print("Model loaded. Time needed: %.4f" % (last_time - time.time()))
                    last_time = time.time()

                sum_writer = tf.summary.FileWriter(self.config.environment.log_dir, graph=tf.get_default_graph())
                merged = tf.summary.merge_all()
                log_freq = self.config.environment.log_frequency
                losses = np.zeros(log_freq)
                batch_size = self.config.network.batch_size

                print("Starting to learn...")

                for i in range(self.config.environment.max_number_of_steps):
                    # sadly for performance reasons this block must be super-ugly
                    if (i % self.config.environment.save_frequency) != 0 or i == 0:
                        losses[i % log_freq], _ = sess.run([loss, train_op])
                        if (i % log_freq) == 0 and i != 0:
                            loss_value = np.average(losses)
                            act_time = time.time()
                            exec_time = act_time - last_time
                            samples_per_sec = batch_size * log_freq / exec_time
                            print("Step %i, avg loss: %f, execution time: %.4f, samples/second: %.4f"
                                  % (i, loss_value, exec_time, samples_per_sec))

                            last_time = time.time()

                    else:
                        sys.stdout.write("Saving model... ")
                        sys.stdout.flush()
                        summary, losses[0], _ = sess.run([merged, loss, train_op])
                        loss_value = np.average(losses)
                        sum_writer.add_summary(summary, i)
                        saver.save(sess, join(self.config.environment.checkpoint_dir, self.config.network.model_name))
                        print("saved.")
                        if (i % log_freq) == 0:
                            act_time = time.time()
                            exec_time = act_time - last_time
                            samples_per_sec = batch_size * log_freq / exec_time
                            print("Step %i, avg loss: %f, execution time: %.4f, samples/second: %.4f"
                                  % (i, loss_value, exec_time, samples_per_sec))
                            last_time = time.time()

                input_thread.request_stop()

            # better be safe than sorry...
            input_thread.request_stop()

    def train_op(self, loss, global_step) -> (tf.Tensor, tf.Tensor):
        num_batches_per_epoch = self.config.environment.examples_per_epoch / self.config.network.batch_size
        decay_steps = int(num_batches_per_epoch * self.config.network.epoch_per_decay)

        lr = tf.train.exponential_decay(self.config.network.initial_learning_rate,
                                        global_step,
                                        decay_steps,
                                        self.config.network.learning_decay,
                                        staircase=True)
        tf.summary.scalar('learning_rate', lr)

        # opt = tf.train.GradientDescentOptimizer(lr)
        # opt = tf.train.AdamOptimizer(lr)
        opt = tf.train.MomentumOptimizer(lr, self.config.network.learning_momentum)
        grads = opt.compute_gradients(loss)
        apply_grad_op = opt.apply_gradients(grads, global_step=global_step)

        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)

        with tf.control_dependencies([apply_grad_op]):
            train_op = tf.no_op(name='train')

        return train_op, grads


class InputThread(object):
    def __init__(self, net_config : NetworkConfig, env_config : EnvConfig):
        self.enq_image = tf.placeholder(tf.float32, shape=[net_config.window_width, net_config.window_height, 1])
        self.enq_label = tf.placeholder(tf.float32, shape=[net_config.output_dim, net_config.max_label_num])
        self.net_config = net_config

        q = tf.RandomShuffleQueue(
            capacity=env_config.min_queue_examples + env_config.number_of_input_threads * net_config.batch_size,
            min_after_dequeue=env_config.min_queue_examples + net_config.batch_size,
            dtypes=[tf.float32, tf.float32],
            shapes=[[net_config.window_width, net_config.window_height, 1],
                    [net_config.output_dim, net_config.max_label_num]]
        )

        self.enqueue_op = q.enqueue([self.enq_image, self.enq_label])
        self.examples_in_queue = q.size()
        self.queue_close_op = q.close(cancel_pending_enqueues=True)
        self.image_batch_queue, self.label_batch_queue = q.dequeue_many(net_config.batch_size)
        self.coord = tf.train.Coordinator()
        self.threads = []

    def enqueue_thread(self, coord, sample_iterator, sess):
        while not coord.should_stop():
            image, label = next(sample_iterator)

            if len(image.shape) < 3:
                image = image.reshape(image.shape[0], image.shape[1], 1)

            label = np.array(label).reshape(4,1)

            try:
                sess.run(self.enqueue_op, feed_dict={self.enq_image: image, self.enq_label: label})
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
        while num_examples_in_queue < MIN_QUEUE_EXAMPLES:
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
        self.coord.join(self.threads)