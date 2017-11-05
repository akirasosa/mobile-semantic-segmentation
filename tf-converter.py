import argparse

import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from tensorflow.python.framework import graph_io
from tensorflow.python.framework.graph_util_impl import convert_variables_to_constants

from nets.MobileUNet import custom_objects

num_output = 1


def main(input_model_path, output_dir, output_fn):
    """
    Convert hdf5 file to protocol buffer file to be used with TensorFlow.
    :param input_model_path:
    :param output_dir:
    :param output_fn:
    :return:
    """
    K.set_learning_phase(0)

    model = load_model(input_model_path, custom_objects=custom_objects())

    pred_node_names = ['output_%s' % n for n in range(num_output)]
    print('output nodes names are: ', pred_node_names)

    for idx, name in enumerate(pred_node_names):
        tf.identity(model.output[idx], name=name)

    sess = K.get_session()
    constant_graph = convert_variables_to_constants(sess,
                                                    sess.graph.as_graph_def(),
                                                    pred_node_names)
    graph_io.write_graph(constant_graph, output_dir, output_fn, as_text=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_model_path',
        type=str,
        default='artifacts/224_1_1.h5',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='artifacts',
    )
    parser.add_argument(
        '--output_fn',
        type=str,
        default='224_1_1.pb',
    )
    args, _ = parser.parse_known_args()

    main(**vars(args))
