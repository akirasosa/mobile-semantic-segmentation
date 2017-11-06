import argparse

import tensorflow as tf
from keras import backend as K
from keras import Input
from keras.engine import Model
from keras.models import load_model
from keras.applications.mobilenet import DepthwiseConv2D, relu6
from keras.layers import BatchNormalization, Activation, Conv2D, concatenate, Conv2DTranspose
from tensorflow.python.framework import graph_io
from tensorflow.python.framework.graph_util_impl import convert_variables_to_constants

from nets.MobileUNet import custom_objects

num_output = 1

def conv_net(inputs):
    x = Conv2D(128, (3, 3), padding='same', strides=(2, 2))(inputs)
    return Model(inputs, x)

def depthwise_conv_net(inputs):
    x = DepthwiseConv2D((3, 3), padding='same', strides=(2, 2))(inputs)
    x = Conv2D(128, (1, 1))(x)
    
    return Model(inputs, x)

def convert(model, output_dir, output_fn):
    """
    Convert hdf5 file to protocol buffer file to be used with TensorFlow.
    :param input_model_path:
    :param output_dir:
    :param output_fn:
    :return:
    """
    K.set_learning_phase(0)
    print('input_model_path: ')
    print model.summary()

    pred_node_names = ['output_%s' % n for n in range(num_output)]
    print('output nodes names are: ', pred_node_names)

    for idx, name in enumerate(pred_node_names):
        tf.identity(model.output[idx], name=name)

    sess = K.get_session()

    constant_graph = convert_variables_to_constants(sess,
                                                    sess.graph.as_graph_def(),
                                                    pred_node_names)
    graph_io.write_graph(constant_graph, output_dir, output_fn, as_text=False)

    K.clear_session()

def main(output_dir):
    """
    Generate protobuf model for benchmark.
    One by one model by uncomment and run
    """
    inputs = Input(shape=(64, 64, 64))

    experiments = [
        # {'name': 'conv_net', 'model': conv_net(inputs)},
        {'name': 'depthwise_conv', 'model': depthwise_conv_net(inputs)},
    ]

    for e in experiments:
        convert(e['model'], output_dir, '{}.pb'.format(e['name']))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output_dir',
        type=str,
        default='artifacts',
    )
    args, _ = parser.parse_known_args()

    main(**vars(args))
