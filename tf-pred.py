import argparse

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from data import standardize

prefix = 'hair_recognition'


def main(pb_file, img_file):
    """
    Predict and visualize by TensorFlow.
    :param pb_file:
    :param img_file:
    :return:
    """
    with tf.gfile.GFile(pb_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name=prefix)

    for op in graph.get_operations():
        print(op.name)

    x = graph.get_tensor_by_name('%s/input_1:0' % prefix)
    y = graph.get_tensor_by_name('%s/output_0:0' % prefix)

    images = np.load(img_file).astype(float)
    img_h = images.shape[1]
    img_w = images.shape[2]

    with tf.Session(graph=graph) as sess:
        for img in images:
            batched = img.reshape(-1, img_h, img_w, 3)
            normalized = standardize(batched)

            pred = sess.run(y, feed_dict={
                x: normalized
            })
            plt.imshow(pred.reshape(img_h, img_w))
            plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pb_file',
        type=str,
        default='artifacts/224_1_1.pb',
    )
    parser.add_argument(
        '--img_file',
        type=str,
        default='data/images-224.npy',
        help='image file as numpy format'
    )
    args, _ = parser.parse_known_args()
    main(**vars(args))
