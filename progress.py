import numpy as np
import scipy.ndimage
import pandas as pd
import matplotlib.pyplot as plt
from inspect_images import *
from nn import *
from tensorflow.python import debug as tf_debug


def doit():
    stats = pd.DataFrame.from_csv("./stats.csv")
    smallest = stats[stats["x"]==stats["x"].min()]
    picloc = "./samples/" + smallest["file name"].values[0]
    pic = scipy.ndimage.imread(picloc)
    label = get_bounding_box(picloc) 
    tf.reset_default_graph()
    sess = tf.Session()
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)

    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        respic, reslabel = tf_pad_image(pic, list(label.astype("int32").flat), 512,512)
        respic = respic.eval()
        reslabel = reslabel
        reslabel = np.asarray( [r.eval() for r in reslabel] )
        reslabel = reslabel.reshape((4,2))

    return respic, reslabel, label

pic, label, origlabel = doit()
print("Result: %s, %s, %s" % doit())
draw_bounding_box(pic, label)
