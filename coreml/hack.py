import six
from coremltools.converters.keras import _keras2_converter
from coremltools.converters.keras._layers2 import convert_activation
from keras.layers import Activation

from layers.BilinearUpSampling import BilinearUpSampling2D


def _convert_upsample_bilinear(builder, layer, input_names, output_names, keras_layer):
    input_name, output_name = (input_names[0], output_names[0])

    if type(keras_layer.size) is int:
        fh = fw = keras_layer.size
    elif len(keras_layer.size) == 2:
        if keras_layer.size[0] != keras_layer.size[1]:
            raise ValueError("Upsample with different rows and columns not supported.")
        else:
            fh = keras_layer.size[0]
            fw = keras_layer.size[1]
    else:
        raise ValueError("Unrecognized upsample factor format %s" % (str(keras_layer.size)))

    builder.add_upsample(name=layer,
                         scaling_factor_h=fh,
                         scaling_factor_w=fw,
                         input_name=input_name,
                         output_name=output_name,
                         mode='BILINEAR')


# See https://github.com/apple/coremltools/pull/44
def _convert_activation_custom(builder, layer, input_names, output_names, keras_layer):
    if six.PY2:
        act_name = keras_layer.activation.func_name
    else:
        act_name = keras_layer.activation.__name__

    if act_name == 'relu6':
        input_name, output_name = (input_names[0], output_names[0])
        relu_output_name = output_name + '_relu'
        builder.add_activation(layer, 'RELU', input_name, relu_output_name)
        # negate it
        neg_output_name = relu_output_name + '_neg'
        builder.add_activation(layer + '__neg__', 'LINEAR', relu_output_name,
                               neg_output_name, [-1.0, 0])
        # apply threshold
        clip_output_name = relu_output_name + '_clip'
        builder.add_unary(layer + '__clip__', neg_output_name, clip_output_name,
                          'threshold', alpha=-6.0)
        # negate it back
        builder.add_activation(layer + '_neg2', 'LINEAR', clip_output_name,
                               output_name, [-1.0, 0])
        return

    return convert_activation(builder, layer, input_names, output_names, keras_layer)


def hack_coremltools():
    # noinspection PyProtectedMember
    reg = _keras2_converter._KERAS_LAYER_REGISTRY
    reg[BilinearUpSampling2D] = _convert_upsample_bilinear
    reg[Activation] = _convert_activation_custom
