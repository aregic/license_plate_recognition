import numpy as np
import scipy.ndimage
import pandas as pd
import matplotlib.pyplot as plt
from inspect_images import *
from nn import *
from tensorflow.python import debug as tf_debug
import configparser
from yolo import Yolo


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

config = configparser.ConfigParser()
config.read('config.cfg')

yolo = Yolo(config['common params'], config['net params'])
yolo.train_on_lots_of_pics('asd')

"""
#train_on_one_pic("./samples/pic107.jpg")
train_on_lots_of_pics("dataset256_tiles.tfrecord", train_on_tiles = False, use_dataset = False)
"""


