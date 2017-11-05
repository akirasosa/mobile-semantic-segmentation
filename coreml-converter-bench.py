from itertools import product

import coremltools
from keras import Input
from keras.utils import CustomObjectScope

from coreml.hack import hack_coremltools
from nets.MobileUNet import MobileUNet, custom_objects


# def conv_net(inputs):
#     x = Conv2D(128, (3, 3), padding='same', strides=(2, 2))(inputs)
#     return Model(inputs, x)
#
#
# def depthwise_conv_net(inputs):
#     x = DepthwiseConv2D((3, 3), padding='same', strides=(2, 2))(inputs)
#     x = Conv2D(128, (1, 1))(x)
#     return Model(inputs, x)
#
#
# def upsample_net(inputs):
#     x = BilinearUpSampling2D(target_size=(128, 128))(inputs)
#     return Model(inputs, x)


def main():
    """
    Generate CoreML model for benchmark by using non-trained model.
    It's useful if you just want to measure the inference speed
    of your model
    """
    hack_coremltools()

    sizes = [224, 192, 160, 128]
    alphas = [1., .75, .50, .25]
    name_fmt = 'mobile_unet_{0:}_{1:03.0f}_{2:03.0f}'

    experiments = [
        {
            'name': name_fmt.format(s, a * 100, a * 100),
            'model': MobileUNet(input_shape=(s, s, 3),
                                input_tensor=Input(shape=(s, s, 3)),
                                alpha=a,
                                alpha_up=a)
        }
        for s, a in product(sizes, alphas)
    ]

    for e in experiments:
        model = e['model']
        name = e['name']

        model.summary()

        with CustomObjectScope(custom_objects()):
            coreml_model = coremltools.converters.keras.convert(model, input_names='data')
        coreml_model.save('artifacts/{}.mlmodel'.format(name))


if __name__ == '__main__':
    main()
