import argparse

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

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

    with tf.Session(graph=graph) as sess:
        images = np.load(img_file).astype(float)
        images /= 255

        for img in images:
            pred = sess.run(y, feed_dict={
                x: img.reshape(-1, 128, 128, 3)
            })
            plt.imshow(pred.reshape(128, 128))
            plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pb_file',
        type=str,
        default='artifacts/025-128-adam.pb',
    )
    parser.add_argument(
        '--img_file',
        type=str,
        default='data/images-128.npy',
        help='image file as numpy format'
    )
    args, _ = parser.parse_known_args()
    main(**vars(args))
