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
    It's useful if you just want to measure the speed of inference
    of your model
    """
    hack_coremltools()

    shape128 = (128, 128, 3)
    shape224 = (224, 224, 3)

    experiments = [
        {
            'name': 'mobile_unet_128_1_025',
            'model': MobileUNet(input_shape=shape128,
                                input_tensor=Input(shape=shape128),
                                alpha_up=0.25)
        },
        {
            'name': 'mobile_unet_224_1_025',
            'model': MobileUNet(input_shape=shape224,
                                input_tensor=Input(shape=shape224),
                                alpha_up=0.25)
        },
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
