import scipy.ndimage
import pandas as pd
import configparser
from yolo import Yolo
import ruamel.yaml
from simple_nn import *
from data_feeder import *


def doit():
    stats = pd.DataFrame.from_csv("./stats.csv")
    smallest = stats[stats["x"]==stats["x"].min()]
    picloc = "./samples/" + smallest["file name"].values[0]
    #biggest = stats[stats["x"]==stats["x"].max()]
    #picloc2 = "./samples/pic154.jpg"
    picloc2 = "./samples/pic107.jpg"
    pic = scipy.ndimage.imread(picloc)
    pic2 = scipy.ndimage.imread(picloc2)
    x,y = np.shape(pic2)
    pic2 = np.reshape(pic2, [x,y,1])

    label = get_bounding_box(picloc) 
    label2 = get_bounding_box(picloc2) 

    tf.reset_default_graph()
    sess = tf.Session()
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)

    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        respic, reslabel = preprocess_image(pic, list(label.astype("int32").flat), 512,512)
        respic2, reslabel2 = preprocess_image(pic2, list(label2.astype("int32").flat), 512,512)
        #respic, reslabel = preprocess_image(pic, list(label.astype("int32").flat), 512,512)
        respic = respic.eval()
        respic2 = respic2.eval()
        reslabel = reslabel
        print("reslabel: %s" % reslabel)
        #reslabel = np.asarray( [r.eval() for r in reslabel] )
        reslabel = reslabel.eval()
        reslabel = reslabel.reshape((4,2))
        reslabel2 = reslabel2.eval()
        reslabel2 = reslabel2.reshape((4,2))
        
        draw_bounding_box(respic, reslabel)
        draw_bounding_box(respic2, reslabel2)

    return respic2, reslabel2, label2

"""
pic, label, origlabel = doit()
print("Result: %s, %s, %s" % (pic,label,origlabel))
"""
"""
draw_bounding_box(pic, label)
"""
def run_yolo():
    config = configparser.ConfigParser()
    config.read('config.cfg')

    yolo = Yolo(config['common params'], config['net params'])
    yolo.train_on_lots_of_pics('asd')


def run_simple_nn():
    config = ruamel.yaml.load(open("./simple_net_config.yaml"))
    network_config = NetworkConfig(**config["SimpleNetConfig"])
    env_config = EnvConfig(**config["EnvConfig"])

    simple_net = SimpleNet(network_config, env_config)

    image_iter = imageLabelIterator(env_config.sample_dir)
    s = SlidingWindowSampleCreator(network_config.slide_x, network_config.slide_y, network_config.window_width,
                                   network_config.window_height, normalize_label=True,
                                   yes_label_weight=network_config.yes_label_weight,
                                   no_label_weight=network_config.no_label_weight)

    sliding_window_iter = s.create_sliding_window_from_iter(image_iter)

    simple_net.train(sliding_window_iter)


run_simple_nn()
"""
#train_on_one_pic("./samples/pic107.jpg")
train_on_lots_of_pics("dataset256_tiles.tfrecord", train_on_tiles = False, use_dataset = False)
"""


