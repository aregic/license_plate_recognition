import tensorflow as tf
from nn import *
from gaborwavelet import orthonormalInit, getBasicKernels
import time
import threading
from data_feeder import EnqueueThread


class Network:
    pass


class Yolo:
    # based on https://github.com/nilboy/tensorflow-yolo/blob/python2.7/yolo/net/yolo_net.py

    def __init__(self, common_params, net_params, test=False):
        """
            common params: a params dict
            net_params   : a params dict

            WARNING : log frequency is expected to be a multiple of batch_size!
        """
        # super(YoloNet, self).__init__(common_params, net_params)
        # process params
        self.image_size = int(common_params['image_size'])
        self.num_classes = int(common_params['num_classes'])
        self.cell_size = int(net_params['cell_size'])
        self.boxes_per_cell = int(net_params['boxes_per_cell'])
        self.batch_size = int(common_params['batch_size'])
        self.weight_decay = float(net_params['weight_decay'])
        self.detailed_log = bool(common_params.get('detailed_log', True))
        self.checkpoint_dir = str(common_params['checkpoint dir'])
        self.log_dir = str(common_params['log dir'])
        self.max_steps = int(common_params['max steps'])
        self.save_frequency = int(common_params['save frequency'])
        self.log_frequency = int(common_params['log frequency'])
        self.max_number_of_boxes_per_image = int(common_params.get('max number of boxes per image', 5))

        self.num_of_preprocess_threads = int(common_params.get('number of preprocessor threads', 8))
        self.min_queue_examples = int(common_params["minimum examples in the input queue"])

        self.examples_per_epoch = int(common_params.get('examples per epoch', 10000))
        self.epoch_per_decay = int(common_params.get('epochs per decay', 100))
        self.initial_learning_rate = float(net_params['initial learning rate'])
        self.learning_decay = float(net_params['learning decay'])
        self.momentum = float(net_params.get('learning momentum'))
        self.use_gabor_wavelet = bool(net_params.get('use gabor wavelet', False))
        self.data_dir = str(common_params["sample directory"])

        self.preprocessor = Preprocessor(
            self.image_size,
            self.image_size,
            self.max_number_of_boxes_per_image)

        if not test:
            self.object_scale = float(net_params['object_scale'])
            self.noobject_scale = float(net_params['noobject_scale'])
            self.class_scale = float(net_params['class_scale'])
            self.coord_scale = float(net_params['coord_scale'])

        self.network = Network()

    def inference(self, image):
        if self.use_gabor_wavelet:
            # init lower layer to predefined kernels (gabor wavelets, edge detectors, etc.)
            basic_kernels = np.asarray([getBasicKernels()]).astype(np.float32)

            # the swapping will probably transpose the kernels but that doesn't matter at the moment
            basic_kernels = np.swapaxes(basic_kernels, 0, 2)
            basic_kernels = np.swapaxes(basic_kernels, 1, 3)

            layer1 = create_layer(image, [7,7,1,32], 0.0, 5e-2, "conv1", const_init = basic_kernels)

        else:
            layer1 = create_layer(image, [3, 3, 1, 32], 0.0, 5e-2, "conv1", show_tensor=True)

        """
        if self.detailed_log:
            images = layer1[0]
            images = tf.transpose(layer1[0, 0:, 0:], [2,0,1])
            tf.summary.image(
                "Layer 1", 
                tf.reshape(images, [32,self.image_size,self.image_size,1]),
                max_outputs = 32)
        """

        # pool1 = tf.nn.max_pool(layer1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name="pool1")
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

        """
        layer7 = create_layer(norm6, [3,3,32,32], 0.1, 1e-3, "conv7", dropout_rate=0.5, batch_norm = True)
        norm7 = tf.nn.lrn(layer7, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name="norm7")
        layer8 = create_layer(norm7, [3,3,128,64], 0.1, 2e-3, "conv8", dropout_rate=0.5)
        norm8 = tf.nn.lrn(layer8, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name="norm8")
        layer9 = create_layer(norm8, [3,3,64,32], 0.1, 3e-3, "conv9", dropout_rate=0.5, batch_norm = True)
        norm10 = tf.nn.lrn(layer9, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name="norm6")
        """

        with tf.variable_scope('local3') as scope:
            # Move everything into depth so we can perform a single matrix multiply.
            reshape = tf.reshape(norm6, [self.batch_size, -1])
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
            tile_weights = normal_variable_with_weight_decay('weights', [192, self.cell_size * self.cell_size * 6],
                                                              stddev=1 / 192.0, wd=0.0)
            tile_biases = variable_on_cpu('biases', [self.cell_size * self.cell_size * 6],
                                           tf.constant_initializer(0.0))
            tile_softmax_linear = tf.add(tf.matmul(dropped_local3, tile_weights), tile_biases, name=scope.name)
            bnormed_softmax = tf.layers.batch_normalization(tile_softmax_linear, training=True)
            tile_dropped_softmax = tf.nn.dropout(bnormed_softmax, 0.9)

            tile_result = tf.reshape(
                tile_dropped_softmax,
                [self.batch_size, self.cell_size, self.cell_size, 6])

        return tile_result

    def train(self, total_loss, global_step):
        num_batches_per_epoch = self.examples_per_epoch / self.batch_size
        decay_steps = int(num_batches_per_epoch * self.epoch_per_decay)

        lr = tf.train.exponential_decay(self.initial_learning_rate,
                                        global_step,
                                        decay_steps,
                                        self.learning_decay,
                                        staircase=True)
        tf.summary.scalar('learning_rate', lr)

        # opt = tf.train.GradientDescentOptimizer(lr)
        # opt = tf.train.AdamOptimizer(lr)
        opt = tf.train.MomentumOptimizer(lr, self.momentum)
        grads = opt.compute_gradients(total_loss)

        grads_list = list(grads)

        first_layer_lr_multiplier = 5
        conv_weight_multiplier = 10

        grads_list[0] = (grads_list[0][0] * first_layer_lr_multiplier, grads_list[0][1])

        for i in range(len(grads_list)):
            grad = grads_list[i][0]
            var = grads_list[i][1]
            if grad is not None and 'conv' in var.name and 'weights' in var.name:
                grads_list[i] = (grad * conv_weight_multiplier, var)

        grads = grads_list

        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)

        with tf.control_dependencies([apply_gradient_op]):
            self.network.train_op = tf.no_op(name='train')

        return self.network.train_op, grads

    def get_inference_result(self, dataset_file: dir):
        """
            Leave train on tiles on False for now.
        """
        tf.reset_default_graph()
        with tf.Graph().as_default():
            global_step = tf.contrib.framework.get_or_create_global_step()
            # global_step_tensor = tf.Variable(1, trainable=False, name='global_step')

            pic_batch, label_batch, enqueue_op, enq_image, enq_label, examples_in_queue = self.readOnTheFly()

            # logits, tiles = self.inference(pic_batch)
            logits = self.inference(pic_batch)

            """
            label_batch_filtered = tf.boolean_mask(
                    label_batch,
                    tf.cast(label_batch[:,:,4], dtype=tf.bool))
            label_batch_filtered = tf.Print(label_batch_filtered, [label_batch_filtered], "\n\nLABEL BATCH FILTERED: ", summarize=250)
            label_batch_filtered = tf.Print(label_batch_filtered, [label_batch], "\n\nLABEL BATCH: ", summarize=250)
            """
            number_of_boxes = tf.cast(tf.count_nonzero(label_batch[:, :, 4], 1), dtype=tf.int32)
            # number_of_boxes = tf.Print(number_of_boxes, [number_of_boxes], "\n\nNumber of boxes: ",
            #        summarize=10)
            self.network.loss, _ = self.loss(logits, label_batch, number_of_boxes)
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(extra_update_ops):
                self.network.train_op, grads = self.train(self.network.loss, global_step)

            coord = tf.train.Coordinator()

            samples_iter = get_samples()

            with tf.Session().as_default() as sess:
                sess.run(tf.local_variables_initializer())
                sess.run(tf.global_variables_initializer())
                # tf.train.global_step(sess, global_step_tensor)

                threads = []
                for i in range(self.num_of_preprocess_threads):
                    # print("Creating thread %i" % i)
                    t = threading.Thread(target=load_preproc_enqueue_thread, args=(
                        sess, coord, enqueue_op, enq_image, enq_label, samples_iter
                    ))

                    t.setDaemon(True)
                    t.start()
                    threads.append(t)
                    coord.register_thread(t)
                    time.sleep(0.5)

                num_examples_in_queue = sess.run(examples_in_queue)
                while num_examples_in_queue < self.min_queue_examples:
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
                # saver = tf.train.import_meta_graph(join(self..checkpoint_dir, "my_model.meta"))
                checkpoint_file = join(self.checkpoint_dir, "my_model")
                if isfile(join(self.checkpoint_dir, "checkpoint")):
                    # saver.restore(sess,tf.train.latest_checkpoint(self.checkpoint_dir))
                    optimistic_restore(sess, checkpoint_file)
                    initialize_uninitialized_vars(sess)
                    print("Model loaded.")

                sum_writer = tf.summary.FileWriter(self.log_dir, graph=tf.get_default_graph())
                merged = tf.summary.merge_all()

                collected_grads = {g: g[1].name for g in grads if g[0] is not None}
                loss_result, logit_result, label_result, pic_result, collected_grad_result, _ = (
                    sess.run([self.network.loss, logits, label_batch, pic_batch,
                              collected_grads, self.network.train_op]))

                # return sess.run(list(collected_grads.keys())), list(collected_grads.values())

                coord.request_stop()
                coord.join(threads)

                return loss_result, logit_result, label_result, pic_result, collected_grad_result


    def readOnTheFly(self, size_x : int, size_y : int, label_max_num : int) -> EnqueueThread:
        enq_image = tf.placeholder(tf.float32, shape=[size_x, size_y, 1])
        enq_label = tf.placeholder(tf.float32, shape=[label_max_num, 5])

        q = tf.RandomShuffleQueue(
            capacity=self.min_queue_examples + self.num_of_preprocess_threads * self.batch_size,
            min_after_dequeue=self.min_queue_examples + self.batch_size,
            dtypes=[tf.float32, tf.float32],
            shapes=[[size_x, size_y, 1], [label_max_num, 5]]
        )

        enqueue_op = q.enqueue([enq_image, enq_label])
        examples_in_queue = q.size()
        queue_close_op = q.close(cancel_pending_enqueues=True)
        image_batch_queue, label_batch_queue = q.dequeue_many(self.batch_size)

        return EnqueueThread(image_batch_queue, label_batch_queue, enqueue_op, enq_image,
                                 enq_label, examples_in_queue, queue_close_op)


    def startThreads(self, enqueue_thread_info : EnqueueThread, sess, coord, samples_iter) -> list:
        threads = []
        for i in range(self.num_of_preprocess_threads):
            # print("Creating thread %i" % i)
            t = threading.Thread(target=load_preproc_enqueue_thread, args=(
                sess,
                coord,
                enqueue_thread_info.enqueue_op,
                enqueue_thread_info.enq_image,
                enqueue_thread_info.enq_label,
                samples_iter
            ))

            t.setDaemon(True)
            t.start()
            threads.append(t)
            coord.register_thread(t)
            time.sleep(0.5)

        num_examples_in_queue = sess.run(enqueue_thread_info.examples_in_queue)
        while num_examples_in_queue < MIN_QUEUE_EXAMPLES:
            num_examples_in_queue = sess.run(enqueue_thread_info.examples_in_queue)
            for t in threads:
                if not t.isAlive():
                    coord.request_stop()
                    raise ValueError("One or more enqueuing threads crashed...")
            time.sleep(0.1)

        print("# of examples in queue: %i" % num_examples_in_queue)

        return threads


    def train_on_lots_of_pics(self, sliding_window : bool = False):
        """
        :param sliding_window If true, training will be done on small part of the picture at a time
            using sliding window
        """
        losses = list(np.zeros(int(self.log_frequency / self.batch_size)))
        tf.reset_default_graph()
        with tf.Graph().as_default():
            global_step = tf.contrib.framework.get_or_create_global_step()

            if sliding_window:
                enqueue_thread = self.readOnTheFly(self.image_size, self.image_size,
                                                   self.max_number_of_boxes_per_image)
            else:
                enqueue_thread = self.readOnTheFly(self.image_size, self.image_size,
                                                   self.max_number_of_boxes_per_image)

            # logits, tiles = self.inference(pic_batch)
            logits = self.inference(enqueue_thread.pic_batch)

            """
            label_batch_filtered = tf.boolean_mask(
                    label_batch,
                    tf.cast(label_batch[:,:,4], dtype=tf.bool))
            label_batch_filtered = tf.Print(label_batch_filtered, [label_batch_filtered], "\n\nLABEL BATCH FILTERED: ", summarize=250)
            label_batch_filtered = tf.Print(label_batch_filtered, [label_batch], "\n\nLABEL BATCH: ", summarize=250)
            """
            number_of_boxes = tf.cast(tf.count_nonzero(enqueue_thread.label_batch[:, :, 4], 1), dtype=tf.int32)
            # number_of_boxes = tf.Print(number_of_boxes, [number_of_boxes], "\n\nNumber of boxes: ",
            #        summarize=10)
            self.network.loss, _ = self.loss(logits, enqueue_thread.label_batch, number_of_boxes)

            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(extra_update_ops):
                self.network.train_op, _ = self.train(self.network.loss, global_step)

            # train_op = self.train(loss, global_step)

            coord = tf.train.Coordinator()

            sampleLoader = SampleLoader(self.image_size, self.image_size, self.max_number_of_boxes_per_image,
                                        self.data_dir)

            samples_iter = sampleLoader.get_samples()

            with tf.Session().as_default() as sess:
                sess.run(tf.local_variables_initializer())
                sess.run(tf.global_variables_initializer())

                threads = self.startThreads(enqueue_thread, sess, coord, samples_iter)

                # Create a coordinator and run all QueueRunner objects
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)

                last_time = time.time()
                saver = tf.train.Saver()
                # saver = tf.train.import_meta_graph(join(self.checkpoint_dir, "my_model.meta"))
                checkpoint_file = join(self.checkpoint_dir, "my_model")
                if isfile(join(self.checkpoint_dir, "checkpoint")):
                    # saver.restore(sess,tf.train.latest_checkpoint(self.checkpoint_dir))
                    optimistic_restore(sess, checkpoint_file)
                    initialize_uninitialized_vars(sess)
                    print("Model loaded.")

                sum_writer = tf.summary.FileWriter(self.log_dir, graph=tf.get_default_graph())
                merged = tf.summary.merge_all()
                for i in range(self.max_steps):
                    log_per_batch = int(self.log_frequency / self.batch_size)
                    if (i % self.save_frequency) != 0 or i == 0:
                        if (i % self.log_frequency) != 0 or i == 0:
                            losses[i % log_per_batch], _ = sess.run([self.network.loss, self.network.train_op])
                        else:
                            losses[0], _ = sess.run([self.network.loss, self.network.train_op])
                            loss_value = np.average(losses)
                            act_time = time.time()
                            exec_time = act_time - last_time
                            samples_per_sec = self.batch_size * self.log_frequency / exec_time
                            print("Step %i, avg loss: %f, execution time: %.4f, samples/second: %.4f"
                                  % (i, loss_value, exec_time, samples_per_sec))

                            last_time = time.time()
                    else:
                        sys.stdout.write("Saving model... ")
                        sys.stdout.flush()
                        summary, losses[0], _ = sess.run([merged, self.network.loss, self.network.train_op])
                        loss_value = np.average(losses)
                        sum_writer.add_summary(summary, i)
                        saver.save(sess, join(self.checkpoint_dir, "my_model"))
                        print("saved.")
                        if (i % self.log_frequency) == 0:
                            act_time = time.time()
                            exec_time = act_time - last_time
                            samples_per_sec = self.batch_size * self.log_frequency / exec_time
                            print("Step %i, avg loss: %f, execution time: %.4f, samples/second: %.4f"
                                  % (i, loss_value, exec_time, samples_per_sec))
                            last_time = time.time()

                coord.request_stop()
                coord.join(threads)

    @staticmethod
    def iou(boxes1, boxes2):
        """
            calculate ious

            Args:
              boxes1: 4-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
              boxes2: 1-D tensor [4] ===> (x_center, y_center, w, h)
            Return:
              iou: 3-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        """
        # boxes1 layout: x1, y1, x2, y2
        # where the bounding box's left upper left corner is (x1, y1), bottom right is (x2, y2)
        boxes1 = tf.stack([boxes1[:, :, :, 0] - boxes1[:, :, :, 2] / 2, boxes1[:, :, :, 1] - boxes1[:, :, :, 3] / 2,
                           boxes1[:, :, :, 0] + boxes1[:, :, :, 2] / 2, boxes1[:, :, :, 1] + boxes1[:, :, :, 3] / 2])
        boxes1 = tf.transpose(boxes1, [1, 2, 3, 0])
        boxes2 = tf.stack([boxes2[0] - boxes2[2] / 2, boxes2[1] - boxes2[3] / 2,
                           boxes2[0] + boxes2[2] / 2, boxes2[1] + boxes2[3] / 2])

        # calculate the left up point
        lu = tf.maximum(boxes1[:, :, :, 0:2], boxes2[0:2])
        rd = tf.minimum(boxes1[:, :, :, 2:], boxes2[2:])

        # intersection
        intersection = rd - lu

        inter_square = intersection[:, :, :, 0] * intersection[:, :, :, 1]
        mask = tf.cast(intersection[:, :, :, 0] > 0, tf.float32) * tf.cast(intersection[:, :, :, 1] > 0, tf.float32)
        inter_square = mask * inter_square

        # calculate the boxs1 square and boxs2 square
        square1 = boxes1[:, :, :, 2] * boxes1[:, :, :, 3]
        square2 = boxes2[2] * boxes2[3]

        return inter_square / (square1 + square2 - inter_square + 1e-6)

    def body1(self, num, object_num, loss, predict, labels, nilboy):
        """
            calculate loss

            Args:
                predict: 3-D tensor [cell_size, cell_size, 5 * boxes_per_cell]
                labels : [max_objects, 5]  (x_center, y_center, w, h, class)
        """
        filtered_labels = tf.boolean_mask(labels, tf.cast(labels[:, 4], dtype=tf.bool))
        label = filtered_labels[num:num + 1, :]
        label = tf.reshape(label, [-1])

        # label = tf.Print(label, [label], "\n\nLABEL: ", summarize=5)

        # calculate objects  tensor [CELL_SIZE, CELL_SIZE]
        min_x = (label[0] - (label[2] / 2)) * self.cell_size
        max_x = (label[0] + (label[2] / 2)) * self.cell_size

        min_y = (label[1] - (label[3] / 2)) * self.cell_size
        max_y = (label[1] + (label[3] / 2)) * self.cell_size

        # due to rouding error the bounding box can slightly leave the picture,
        # which might result in index out of bounds, so clip it
        min_x = tf.clip_by_value(tf.floor(min_x), 0, self.cell_size - 1)
        min_y = tf.clip_by_value(tf.floor(min_y), 0, self.cell_size - 1)

        max_x = tf.clip_by_value(tf.ceil(max_x), 0, self.cell_size - 1)
        max_y = tf.clip_by_value(tf.ceil(max_y), 0, self.cell_size - 1)

        temp = tf.cast(tf.stack([max_y - min_y, max_x - min_x]), dtype=tf.int32)
        # temp = tf.Print(temp, [min_x, min_y, max_x, max_y], "\n\nMaximums: ", summarize=4)
        objects = tf.ones(temp, tf.float32)

        temp = tf.cast(tf.stack([min_y, self.cell_size - max_y, min_x, self.cell_size - max_x]), tf.int32)
        # temp = tf.Print(temp, [temp], "\n\nPadding for objects: ", summarize=4)
        temp = tf.reshape(temp, (2, 2))
        objects = tf.pad(objects, temp, "CONSTANT")

        # calculate objects  tensor [CELL_SIZE, CELL_SIZE]
        # calculate responsible tensor [CELL_SIZE, CELL_SIZE]
        center_x = label[0] * self.cell_size
        center_x = tf.floor(center_x)

        center_y = label[1] * self.cell_size
        center_y = tf.floor(center_y)

        response = tf.ones([1, 1], tf.float32)

        temp = tf.cast(tf.stack([center_y, self.cell_size - center_y - 1, center_x, self.cell_size - center_x - 1]),
                       tf.int32)
        temp = tf.reshape(temp, (2, 2))
        response = tf.pad(response, temp, "CONSTANT")
        # objects = response

        # calculate iou_predict_truth [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        predict_boxes = predict[:, :, self.num_classes + self.boxes_per_cell:]

        predict_boxes = tf.reshape(predict_boxes, [self.cell_size, self.cell_size, self.boxes_per_cell, 4])
        predict_boxes = predict_boxes * [1 / self.cell_size, 1 / self.cell_size, 1, 1]

        base_boxes = np.zeros([self.cell_size, self.cell_size, 4])

        for y in range(self.cell_size):
            for x in range(self.cell_size):
                base_boxes[x, y, :] = [x / self.cell_size, y / self.cell_size, 0, 0]

        base_boxes = np.tile(
            np.resize(base_boxes, [self.cell_size, self.cell_size, 1, 4]), [1, 1, self.boxes_per_cell, 1])

        predict_boxes = base_boxes + predict_boxes

        iou_predict_truth = self.iou(predict_boxes, label[0:4])
        # calculate C [cell_size, cell_size, boxes_per_cell]
        C = iou_predict_truth * tf.reshape(response, [self.cell_size, self.cell_size, 1])

        # calculate I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        I = iou_predict_truth * tf.reshape(response, (self.cell_size, self.cell_size, 1))

        max_I = tf.reduce_max(I, 2, keep_dims=True)

        I = tf.cast((I >= max_I), tf.float32) * tf.reshape(response, (self.cell_size, self.cell_size, 1))

        # calculate no_I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        no_I = tf.ones_like(I, dtype=tf.float32) - I

        p_C = predict[:, :, self.num_classes:self.num_classes + self.boxes_per_cell]

        # calculate truth x,y,sqrt_w,sqrt_h 0-D
        x = label[0]
        y = label[1]

        sqrt_w = tf.sqrt(tf.abs(label[2]))
        sqrt_h = tf.sqrt(tf.abs(label[3]))

        # calculate predict p_x, p_y, p_sqrt_w, p_sqrt_h 3-D [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        p_x = predict_boxes[:, :, :, 0]
        p_y = predict_boxes[:, :, :, 1]

        p_sqrt_w = tf.sqrt(tf.minimum(self.image_size * 1.0, tf.maximum(0.0, predict_boxes[:, :, :, 2])))
        p_sqrt_h = tf.sqrt(tf.minimum(self.image_size * 1.0, tf.maximum(0.0, predict_boxes[:, :, :, 3])))

        # calculate truth p 1-D tensor [NUM_CLASSES]
        P = tf.one_hot(tf.cast(label[4], tf.int32), self.num_classes, dtype=tf.float32)

        # calculate predict p_P 3-D tensor [CELL_SIZE, CELL_SIZE, NUM_CLASSES]
        p_P = predict[:, :, 0:self.num_classes]

        class_loss = tf.nn.l2_loss(tf.reshape(objects, (self.cell_size, self.cell_size, 1))
                                   * (p_P - P)) * self.class_scale

        object_loss = tf.nn.l2_loss(I * (p_C - C)) * self.object_scale

        noobject_loss = tf.nn.l2_loss(no_I * (p_C)) * self.noobject_scale

        coord_loss = (tf.nn.l2_loss(I * (p_x - x) / (self.image_size / self.cell_size)) +
                      tf.nn.l2_loss(I * (p_y - y) / (self.image_size / self.cell_size)) +
                      tf.nn.l2_loss(I * (p_sqrt_w - sqrt_w)) / self.image_size +
                      tf.nn.l2_loss(I * (p_sqrt_h - sqrt_h)) / self.image_size) * self.coord_scale

        nilboy = I

        return num + 1, object_num, [loss[0] + class_loss, loss[1] + object_loss, loss[2] + noobject_loss,
                                     loss[3] + coord_loss], predict, labels, nilboy

    def loss(self, predicts, labels, objects_num):
        """
            Add Loss to all the trainable variables

            Args:
              predicts: 4-D tensor [batch_size, cell_size, cell_size, 5 * boxes_per_cell]
                  ===> (num_classes, boxes_per_cell, 4 * boxes_per_cell)
              labels  : 3-D tensor of [batch_size, max_objects, 5]
              objects_num: 1-D tensor [batch_size]
        """

        def cond1(num, object_num, loss, predict, label, nilboy):
            return num < object_num

        class_loss = tf.constant(0, tf.float32)
        object_loss = tf.constant(0, tf.float32)
        noobject_loss = tf.constant(0, tf.float32)
        coord_loss = tf.constant(0, tf.float32)
        loss = [0, 0, 0, 0]
        for i in range(self.batch_size):
            predict = predicts[i, :, :, :]
            label = labels[i, :, :]
            object_num = objects_num[i]
            nilboy = tf.ones([self.cell_size, self.cell_size, 1])
            tuple_results = tf.while_loop(cond1, self.body1,
                                          [
                                              tf.constant(0), object_num,
                                              [class_loss, object_loss, noobject_loss, coord_loss],
                                              predict, label, nilboy
                                          ])
            for j in range(4):
                loss[j] = loss[j] + tuple_results[2][j]
                nilboy = tuple_results[5]

        tf.add_to_collection('losses', (loss[0] + loss[1] + loss[2] + loss[3]) / self.batch_size)

        tf.summary.scalar('class_loss', loss[0] / self.batch_size)
        tf.summary.scalar('object_loss', loss[1] / self.batch_size)
        tf.summary.scalar('noobject_loss', loss[2] / self.batch_size)
        tf.summary.scalar('coord_loss', loss[3] / self.batch_size)
        tf.summary.scalar('weight_loss', tf.add_n(tf.get_collection('losses')) - (
                    loss[0] + loss[1] + loss[2] + loss[3]) / self.batch_size)

        return tf.add_n(tf.get_collection('losses'), name='total_loss'), nilboy
