import tensorflow as tf
from nn import *


class Yolo:
    # taken from https://github.com/nilboy/tensorflow-yolo/blob/python2.7/yolo/net/yolo_net.py


    def __init__(self, common_params, net_params, test=False):
        """
            common params: a params dict
            net_params   : a params dict
        """
        #super(YoloNet, self).__init__(common_params, net_params)
        #process params
        self.image_size = int(common_params['image_size'])
        self.num_classes = int(common_params['num_classes'])
        self.cell_size = int(net_params['cell_size'])
        self.boxes_per_cell = int(net_params['boxes_per_cell'])
        self.batch_size = int(common_params['batch_size'])
        self.weight_decay = float(net_params['weight_decay'])
        self.detailed_log = bool(common_params.get('detailed_log', True))
        self.checkpoint_dir = str(common_params['checkpoint dir'])
        self.log_dir = str(common_params['log dir'])
        self.max_steps = str(common_params['max steps'])
        self.log_frequency = str(common_params['log frequency'])

        self.num_of_preprocess_threads = int(common_params.get('number of preprocessor threads', 8))

        self.examples_per_epoch = int(common_params.get('examples per epoch', 10000))
        self.epoch_per_decay = int(common_params.get('epochs per decay', 100))
        self.initial_learning_rate = float(net_params.get('initial learning rate', 1e-5))
        self.learning_decay = float(net_params.get('learning decay', 1))

        self.preprocessor = Preprocessor(self.image_size, self.image_size, self.cell_size, self.cell_size)

        if not test:
            self.object_scale = float(net_params['object_scale'])
            self.noobject_scale = float(net_params['noobject_scale'])
            self.class_scale = float(net_params['class_scale'])
            self.coord_scale = float(net_params['coord_scale'])


    def inference(self, image):
        # lower layers are like inception-v3's
        basic_kernels = np.asarray([getBasicKernels()]).astype(np.float32)
        # the swapping will probably transpose the kernels but that doesn't matter at the moment
        basic_kernels = np.swapaxes(basic_kernels, 0,2)
        basic_kernels = np.swapaxes(basic_kernels, 1,3)

        layer1 = create_layer(image, [7,7,1,32], 0.0, 5e-2, "conv1", const_init = basic_kernels)

        if self.detailed_log:
            images = layer1[0]
            images = tf.transpose(layer1[0, 0:, 0:], [2,0,1])
            tf.summary.image(
                "Layer 1", 
                tf.reshape(images, [32,self.image_size,self.image_size,1]),
                max_outputs = 32)

        pool1 = tf.nn.max_pool(layer1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name="pool1")
        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name="norm1")
        layer2 = create_layer(norm1, [3,3,32,32], 0.1, 5e-2, "conv2", dropout_rate=0.8)
        pool2 = tf.nn.max_pool(layer2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name="pool2")
        norm2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name="norm2")
         
        layer3 = create_layer(norm2, [1,1,32,32], 0.1, 5e-2, "conv3", dropout_rate=0.8)
        pool3 = tf.nn.max_pool(layer3, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name="pool3")
        norm3 = tf.nn.lrn(pool3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name="norm3")

        layer4 = create_layer(norm3, [3,3,32,64], 0.1, 5e-2, "conv4", dropout_rate=0.8)
        norm4 = tf.nn.lrn(layer4, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name="norm4")

        layer5 = create_layer(norm4, [1,1,64,64], 0.1, 3e-2, "conv5", dropout_rate=0.5)
        norm5 = tf.nn.lrn(layer5, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name="norm5")
        layer6 = create_layer(norm5, [3,3,64,32], 0.1, 2e-2, "conv6", dropout_rate=0.5)
        pool6 = tf.nn.max_pool(layer6, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name="pool6")
        norm6 = tf.nn.lrn(pool6, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name="norm6")

        """
        layer7 = create_layer(norm6, [1,1,128,128], 0.1, 1e-3, "conv7", dropout_rate=0.5)
        norm7 = tf.nn.lrn(layer7, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name="norm7")
        layer8 = create_layer(norm7, [3,3,128,256], 0.1, 2e-3, "conv8", dropout_rate=0.5)
        norm8 = tf.nn.lrn(layer8, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name="norm8")
        layer9 = create_layer(norm8, [3,3,256,512], 0.1, 3e-3, "conv9", dropout_rate=0.5)
        norm10 = tf.nn.lrn(layer9, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name="norm6")
        """

        with tf.variable_scope('local3') as scope:
            # Move everything into depth so we can perform a single matrix multiply.
            reshape = tf.reshape(norm6, [self.batch_size, -1])
            dim = reshape.get_shape()[1].value
            weights = _normal_variable_with_weight_decay('weights', shape=[dim, 384],
                                              stddev=0.04, wd=0.004)
            biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
            #local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
            pre_act3 = tf.matmul(reshape, weights) + biases
            local3 = leakyRelu(pre_act3)
            dropped_local3 = tf.nn.dropout(local3, 0.5)
     
        # local4
        with tf.variable_scope('local4') as scope:
            weights = _normal_variable_with_weight_decay('weights', shape=[384, 192],
                                                  stddev=0.04, wd=0.004)
            biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
            #local4 = tf.nn.relu(tf.matmul(dropped_local3, weights) + biases, name=scope.name)
            pre_act4 = tf.matmul(dropped_local3, weights) + biases
            local4 = leakyRelu(pre_act4)
            dropped_local4 = tf.nn.dropout(local4, 0.8)

        with tf.variable_scope('tile_output_layer') as scope:
            tile_weights = _normal_variable_with_weight_decay('weights', [192, self.cell_size * self.cell_size * 6],
                                                  stddev=1/192.0, wd=0.0)
            tile_biases = _variable_on_cpu('biases', [self.cell_size * self.cell_size * 6],
                                      tf.constant_initializer(0.0))
            tile_softmax_linear = tf.add(tf.matmul(dropped_local4, tile_weights), tile_biases, name=scope.name)
            tile_dropped_softmax = tf.nn.dropout(tile_softmax_linear, 0.9)

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

        #opt = tf.train.GradientDescentOptimizer(lr)
        opt = tf.train.AdamOptimizer(lr)
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


    def train_on_lots_of_pics(self, dataset_file : dir, train_on_tiles = False, use_dataset = False):
        """
            Leave train on tiles on False for now.
        """
        tf.reset_default_graph()
        with tf.Graph().as_default():
            global_step = tf.contrib.framework.get_or_create_global_step()

            if use_dataset:
                pic_batch, label_batch = readFromDataSet(dataset_file, train_on_tiles)
            else:
                pic_batch, label_batch, enqueue_op, enq_image, enq_label, examples_in_queue = readOnTheFly()

            #logits, tiles = self.inference(pic_batch)
            logits = self.inference(pic_batch)

            if train_on_tiles:
                # if dataset is created with train_on_tiles is True, label_batch will actually
                # contain the tile matrix...
                # TODO fix this
                if self.detailed_log:
                    tf.summary.image(
                            "tile label",
                            tf.reshape(label_batch, [self.batch_size, self.cell_size, self.cell_size, 1]))
                    alpha_cut = np.full([self.batch_size, self.cell_size, self.cell_size], ALPHA_CUT, dtype=np.float32)
                    tf.summary.image(
                            "tile output",
                            tf.cast(
                                tf.reshape(tf.greater_equal(tiles, tf.convert_to_tensor(alpha_cut)),
                                    [self.batch_size, self.cell_size, self.cell_size, 1]),
                                dtype=tf.float32))

                loss = self.loss(tiles,label_batch)
                train_op = self.train(loss, global_step)

            else:
                loss = self.loss(logits,label_batch, label_batch.get_shape())
                train_op = train(loss, global_step)

            coord = tf.train.Coordinator()

            if not use_dataset:
                samples_iter = get_samples()

            with tf.Session().as_default() as sess:
                sess.run(tf.local_variables_initializer())
                sess.run(tf.global_variables_initializer())

                if not use_dataset:
                    threads = []
                    for i in range(self.num_of_preprocess_threads):
                        
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
                #saver = tf.train.import_meta_graph(join(self..checkpoint_dir, "my_model.meta"))
                checkpoint_file = join(self.checkpoint_dir, "my_model")
                if isfile(join(self.checkpoint_dir, "checkpoint")):
                    #saver.restore(sess,tf.train.latest_checkpoint(self.checkpoint_dir))
                    optimistic_restore(sess, checkpoint_file)
                    initialize_uninitialized_vars(sess)
                    print("Model loaded.")

                sum_writer = tf.summary.FileWriter(self.log_dir, graph=tf.get_default_graph())
                merged = tf.summary.merge_all()
                for i in range(self.max_steps):
                    if (i % self.save_frequency) != 0 or i == 0:
                        if (i % self.log_frequency) != 0:
                            sess.run([loss, train_op])
                        else:
                            if train_on_tiles:
                                loss_value, res_tiles, res_label_tiles, _ = sess.run([loss, tiles, label_batch, train_op])
                                tilenum = np.count_nonzero(res_label_tiles)
                                positive_tiles = getPositiveTiles(res_tiles, ALPHA_CUT)
                                missed = np.count_nonzero(res_label_tiles - positive_tiles)
                                act_time = time.time()
                                exec_time = act_time - last_time
                                samples_per_sec = self.batch_size * self.log_frequency / exec_time
                                print("Step %i, loss: %f, execution time: %.4f, samples/second: %.4f" 
                                        % (i, loss_value, exec_time, samples_per_sec) )
                                print("Tiles in label: %i, tile in output: %i, missed: %i" % 
                                        (tilenum, np.count_nonzero(positive_tiles), missed))
                            else:
                                loss_value, _ = sess.run([loss, train_op])
                                act_time = time.time()
                                exec_time = act_time - last_time
                                samples_per_sec = self.batch_size * self.log_frequency / exec_time
                                print("Step %i, loss: %f, execution time: %.4f, samples/second: %.4f" 
                                        % (i, loss_value, exec_time, samples_per_sec) )
     
                            last_time = time.time()
                    else:
                        sys.stdout.write("Saving model... ")
                        sys.stdout.flush()
                        summary, loss_value, _ = sess.run([merged, loss, train_op])
                        sum_writer.add_summary(summary, i)
                        saver.save(sess, join(self.checkpoint_dir, "my_model"))
                        print("saved.")
                        if (i % self.log_frequency) == 0:
                            act_time = time.time()
                            exec_time = act_time - last_time
                            samples_per_sec = self.batch_size * self.log_frequency / exec_time
                            print("Step %i, loss: %f, execution time: %.4f, samples/second: %.4f" 
                                    % (i, loss_value, exec_time, samples_per_sec) )
                            last_time = time.time()

                coord.request_stop()
                coord.join(threads)


    def iou(self, boxes1, boxes2):
        """
            calculate ious

            Args:
              boxes1: 4-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
              boxes2: 1-D tensor [4] ===> (x_center, y_center, w, h)
            Return:
              iou: 3-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        """
        boxes1 = tf.pack([boxes1[:, :, :, 0] - boxes1[:, :, :, 2] / 2, boxes1[:, :, :, 1] - boxes1[:, :, :, 3] / 2,
                          boxes1[:, :, :, 0] + boxes1[:, :, :, 2] / 2, boxes1[:, :, :, 1] + boxes1[:, :, :, 3] / 2])
        boxes1 = tf.transpose(boxes1, [1, 2, 3, 0])
        boxes2 =  tf.pack([boxes2[0] - boxes2[2] / 2, boxes2[1] - boxes2[3] / 2,
                          boxes2[0] + boxes2[2] / 2, boxes2[1] + boxes2[3] / 2])

        #calculate the left up point
        lu = tf.maximum(boxes1[:, :, :, 0:2], boxes2[0:2])
        rd = tf.minimum(boxes1[:, :, :, 2:], boxes2[2:])

        #intersection
        intersection = rd - lu 

        inter_square = intersection[:, :, :, 0] * intersection[:, :, :, 1]
        mask = tf.cast(intersection[:, :, :, 0] > 0, tf.float32) * tf.cast(intersection[:, :, :, 1] > 0, tf.float32)
        inter_square = mask * inter_square

        #calculate the boxs1 square and boxs2 square
        square1 = (boxes1[:, :, :, 2] - boxes1[:, :, :, 0]) * (boxes1[:, :, :, 3] - boxes1[:, :, :, 1])
        square2 = (boxes2[2] - boxes2[0]) * (boxes2[3] - boxes2[1])

        return inter_square/(square1 + square2 - inter_square + 1e-6)


    def body1(self, num, object_num, loss, predict, labels, nilboy):
        """
            calculate loss

            Args:
                predict: 3-D tensor [cell_size, cell_size, 5 * boxes_per_cell]
                labels : [max_objects, 5]  (x_center, y_center, w, h, class)
        """
        label = labels[num:num+1, :]
        label = tf.reshape(label, [-1])

        #calculate objects  tensor [CELL_SIZE, CELL_SIZE]
        min_x = (label[0] - label[2] / 2) / (self.image_size / self.cell_size)
        max_x = (label[0] + label[2] / 2) / (self.image_size / self.cell_size)

        min_y = (label[1] - label[3] / 2) / (self.image_size / self.cell_size)
        max_y = (label[1] + label[3] / 2) / (self.image_size / self.cell_size)

        min_x = tf.floor(min_x)
        min_y = tf.floor(min_y)

        max_x = tf.ceil(max_x)
        max_y = tf.ceil(max_y)

        temp = tf.cast(tf.stack([max_y - min_y, max_x - min_x]), dtype=tf.int32)
        objects = tf.ones(temp, tf.float32)

        temp = tf.cast(tf.stack([min_y, self.cell_size - max_y, min_x, self.cell_size - max_x]), tf.int32)
        temp = tf.reshape(temp, (2, 2))
        objects = tf.pad(objects, temp, "CONSTANT")

        #calculate objects  tensor [CELL_SIZE, CELL_SIZE]
        #calculate responsible tensor [CELL_SIZE, CELL_SIZE]
        center_x = label[0] / (self.image_size / self.cell_size)
        center_x = tf.floor(center_x)

        center_y = label[1] / (self.image_size / self.cell_size)
        center_y = tf.floor(center_y)

        response = tf.ones([1, 1], tf.float32)

        temp = tf.cast(tf.stack([center_y, self.cell_size - center_y - 1, center_x, self.cell_size -center_x - 1]), tf.int32)
        temp = tf.reshape(temp, (2, 2))
        response = tf.pad(response, temp, "CONSTANT")
        #objects = response

        #calculate iou_predict_truth [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        predict_boxes = predict[:, :, self.num_classes + self.boxes_per_cell:]
        

        predict_boxes = tf.reshape(predict_boxes, [self.cell_size, self.cell_size, self.boxes_per_cell, 4])
        predict_boxes = predict_boxes * [self.image_size / self.cell_size, self.image_size / self.cell_size, self.image_size, self.image_size]

        base_boxes = np.zeros([self.cell_size, self.cell_size, 4])

        for y in range(self.cell_size):
          for x in range(self.cell_size):
            #nilboy
            base_boxes[y, x, :] = [self.image_size / self.cell_size * x, self.image_size / self.cell_size * y, 0, 0]
        base_boxes = np.tile(np.resize(base_boxes, [self.cell_size, self.cell_size, 1, 4]), [1, 1, self.boxes_per_cell, 1])

        predict_boxes = base_boxes + predict_boxes

        iou_predict_truth = self.iou(predict_boxes, label[0:4])
        #calculate C [cell_size, cell_size, boxes_per_cell]
        C = iou_predict_truth * tf.reshape(response, [self.cell_size, self.cell_size, 1])

        #calculate I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        I = iou_predict_truth * tf.reshape(response, (self.cell_size, self.cell_size, 1))
        
        max_I = tf.reduce_max(I, 2, keep_dims=True)

        I = tf.cast((I >= max_I), tf.float32) * tf.reshape(response, (self.cell_size, self.cell_size, 1))

        #calculate no_I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        no_I = tf.ones_like(I, dtype=tf.float32) - I 

        p_C = predict[:, :, self.num_classes:self.num_classes + self.boxes_per_cell]

        #calculate truth x,y,sqrt_w,sqrt_h 0-D
        x = label[0]
        y = label[1]

        sqrt_w = tf.sqrt(tf.abs(label[2]))
        sqrt_h = tf.sqrt(tf.abs(label[3]))

        #calculate predict p_x, p_y, p_sqrt_w, p_sqrt_h 3-D [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        p_x = predict_boxes[:, :, :, 0]
        p_y = predict_boxes[:, :, :, 1]

        p_sqrt_w = tf.sqrt(tf.minimum(self.image_size * 1.0, tf.maximum(0.0, predict_boxes[:, :, :, 2])))
        p_sqrt_h = tf.sqrt(tf.minimum(self.image_size * 1.0, tf.maximum(0.0, predict_boxes[:, :, :, 3])))

        #calculate truth p 1-D tensor [NUM_CLASSES]
        P = tf.one_hot(tf.cast(label[4], tf.int32), self.num_classes, dtype=tf.float32)

        #calculate predict p_P 3-D tensor [CELL_SIZE, CELL_SIZE, NUM_CLASSES]
        p_P = predict[:, :, 0:self.num_classes]

        class_loss = tf.nn.l2_loss(tf.reshape(objects, (self.cell_size, self.cell_size, 1)) * (p_P - P)) * self.class_scale

        object_loss = tf.nn.l2_loss(I * (p_C - C)) * self.object_scale

        noobject_loss = tf.nn.l2_loss(no_I * (p_C)) * self.noobject_scale

        coord_loss = (tf.nn.l2_loss(I * (p_x - x)/(self.image_size/self.cell_size)) +
                     tf.nn.l2_loss(I * (p_y - y)/(self.image_size/self.cell_size)) +
                     tf.nn.l2_loss(I * (p_sqrt_w - sqrt_w))/ self.image_size +
                     tf.nn.l2_loss(I * (p_sqrt_h - sqrt_h))/self.image_size) * self.coord_scale

        nilboy = I

        return num + 1, object_num, [loss[0] + class_loss, loss[1] + object_loss, loss[2] + noobject_loss, loss[3] + coord_loss], predict, labels, nilboy


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
            nilboy = tf.ones([7,7,2])
            tuple_results = tf.while_loop(cond1, self.body1, 
                    [
                        tf.constant(0), object_num, [class_loss, object_loss, noobject_loss, coord_loss],
                        predict, label, nilboy
                    ])
            for j in range(4):
                loss[j] = loss[j] + tuple_results[2][j]
                nilboy = tuple_results[5]

        tf.add_to_collection('losses', (loss[0] + loss[1] + loss[2] + loss[3])/self.batch_size)

        tf.scalar_summary('class_loss', loss[0]/self.batch_size)
        tf.scalar_summary('object_loss', loss[1]/self.batch_size)
        tf.scalar_summary('noobject_loss', loss[2]/self.batch_size)
        tf.scalar_summary('coord_loss', loss[3]/self.batch_size)
        tf.scalar_summary('weight_loss', tf.add_n(tf.get_collection('losses')) - (loss[0] + loss[1] + loss[2] + loss[3])/self.batch_size )

        return tf.add_n(tf.get_collection('losses'), name='total_loss'), nilboy


def _normal_variable_with_weight_decay(name, shape, stddev, wd):
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

  return _variable_with_weight_decay(name, shape, wd, initializer)


def _variable_with_weight_decay(name, shape, wd, initializer):
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
  var = _variable_on_cpu(
      name,
      shape,
      initializer)
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


def create_layer(image, shape : np.ndarray, initialial_value : float, stddev : float, 
                 scope_name : str, dropout_rate : float = 0.0, leaky_alpha : float = 0.1, 
                 const_init = None):
    if const_init is None:
        initializer = tf.constant(orthonormalInit(shape))
        #initializer = tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32) # stddev originally: 5e-2
    else:
        initializer = tf.constant(const_init)

    with tf.variable_scope(scope_name) as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=None,
                                             wd=0.0,
                                             initializer = initializer)


        conv = tf.nn.conv2d(image, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', shape[3], tf.constant_initializer(initialial_value))
        pre_activation = tf.nn.bias_add(conv, biases)
        #conv1 = tf.nn.relu(pre_activation, name=scope.name)
        #conv1 = tf.maximum(pre_activation, leaky_alpha * pre_activation, name="leaky_relu")
        conv1 = leakyRelu(pre_activation, leaky_alpha)

        print("Layer %s initalized." % scope_name)

        if dropout_rate != 0.0:
            dropped_conv1 = tf.nn.dropout(conv1, dropout_rate)
            return dropped_conv1
        else:
            return conv1


def leakyRelu(pre_activation, leaky_alpha : float = 0.1):
    return tf.maximum(pre_activation, leaky_alpha * pre_activation, name="leaky_relu")


