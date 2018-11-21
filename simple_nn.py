import tensorflow as tf
from nn import *
from gaborwavelet import getBasicKernels
from configs import NetworkConfig, EnvConfig
from data_feeder import *


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

            layer1 = create_layer(image, [7,7,1,32], 0.0, "conv1", const_init = basic_kernels,
                                  leaky_alpha=self.config.network.leaky_alpha)

        else:
            layer1 = create_layer(image, [3, 3, 1, 32], 0.0, "conv1", show_tensor=True,
                                  leaky_alpha=self.config.network.leaky_alpha)

        norm1 = tf.nn.lrn(layer1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name="norm1")
        layer2 = create_layer(norm1, [3, 3, 32, 64], 0.1, "conv2", dropout_rate=self.config.network.cnn_dropout_rate,
                              leaky_alpha=self.config.network.leaky_alpha)
        pool2 = tf.nn.max_pool(layer2, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='SAME', name="pool2")
        norm2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name="norm2")

        """
        layer3 = create_layer(norm2, [3, 3, 64, 32], 0.1, "conv3", dropout_rate=self.config.network.cnn_dropout_rate,
                              leaky_alpha=self.config.network.leaky_alpha)
        pool3 = tf.nn.max_pool(layer3, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='SAME', name="pool3")
        norm3 = tf.nn.lrn(pool3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name="norm3")

        layer4 = create_layer(norm3, [3, 3, 32, 32], 0.1, "conv4", dropout_rate=self.config.network.cnn_dropout_rate,
                              leaky_alpha=self.config.network.leaky_alpha)
        norm4 = tf.nn.lrn(layer4, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name="norm4")
        layer5 = create_layer(norm4, [1, 1, 32, 32], 0.1, "conv5", dropout_rate=self.config.network.cnn_dropout_rate,
                              leaky_alpha=self.config.network.leaky_alpha)
        norm5 = tf.nn.lrn(layer5, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name="norm5")

        layer6 = create_layer(norm5, [3, 3, 32, 32], 0.1, "conv6", dropout_rate=self.config.network.cnn_dropout_rate,
                              leaky_alpha=self.config.network.leaky_alpha)
        pool6 = tf.nn.max_pool(layer6, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='SAME', name="pool6")
        norm6 = tf.nn.lrn(pool6, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name="norm6")
        """

        with tf.variable_scope('local3'):
            # Move everything into depth so we can perform a single matrix multiply.
            reshape = tf.reshape(norm2, [self.config.network.batch_size, -1])
            dim = reshape.get_shape()[1].value
            weights = normal_variable_with_weight_decay('weights', shape=[dim, 192],
                                                         stddev=1.0/192.0, wd=0.004)
            biases = variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
            # local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
            pre_act3 = tf.matmul(reshape, weights) + biases
            local3 = leakyRelu(pre_act3)
            #bnormed_3 = tf.layers.batch_normalization(local3, training=True)
            dropped_local3 = tf.nn.dropout(local3, 0.7)
            #dropped_local3 = tf.nn.dropout(bnormed_3, 0.5)

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
            #bnormed_softmax = tf.layers.batch_normalization(tile_softmax_linear, training=True)
            tile_dropped_softmax = tf.nn.dropout(tile_softmax_linear, 0.9)

            tile_result = tf.reshape(
                tile_dropped_softmax,
                #bnormed_softmax,
                [self.config.network.batch_size, self.config.network.output_dim, self.config.network.max_label_num])

        return tile_result


    def train(self, input_iterator : InputIterator, is_training : bool = True):
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
            l1_loss = tf.reshape(logits - input_thread.label_batch_queue, [self.config.network.batch_size, 5])

            loss = tf.nn.l2_loss(
                tf.matmul(l1_loss, input_thread.enq_loss_batch_queue, transpose_a=True))
            tf.summary.scalar("loss", loss / self.config.network.batch_size)

            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(extra_update_ops):
                train_op, grads = self.train_op(loss, global_step)

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
                    print("Model loaded. Time needed: %.4f" % (time.time() - last_time))
                    last_time = time.time()

                sum_writer = tf.summary.FileWriter(self.config.environment.log_dir, graph=tf.get_default_graph())
                merged = tf.summary.merge_all()
                log_freq = self.config.environment.log_frequency
                losses = np.zeros(log_freq)
                batch_size = self.config.network.batch_size

                if is_training:
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

                else:
                    collected_grads = {g: g[1].name for g in grads if g[0] is not None}
                    loss, image, orig_label, output, gradients = sess.run([loss, input_thread.image_batch_queue,
                                                                           input_thread.label_batch_queue, logits,
                                                                           collected_grads])
                    input_thread.request_stop()
                    return loss, image, orig_label, output, gradients

            # better be safe than sorry...
            input_thread.request_stop()

    def train_op(self, loss, global_step) -> (tf.Tensor, list):
        num_batches_per_epoch = self.config.environment.examples_per_epoch / self.config.network.batch_size
        decay_steps = int(num_batches_per_epoch * self.config.network.epoch_per_decay)

        lr = tf.train.exponential_decay(self.config.network.initial_learning_rate,
                                        global_step,
                                        decay_steps,
                                        self.config.network.learning_decay,
                                        staircase=True)
        tf.summary.scalar('learning_rate', lr)

        #opt = tf.train.GradientDescentOptimizer(lr)
        opt = tf.train.AdamOptimizer(lr)
        #opt = tf.train.MomentumOptimizer(lr, self.config.network.learning_momentum)
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
