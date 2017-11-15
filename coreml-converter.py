import argparse
import re

import coremltools
from keras.models import load_model
from keras.utils import CustomObjectScope

from coreml.hack import hack_coremltools
from nets.MobileUNet import custom_objects


def main(input_model_path):
    """
    Convert hdf5 file to CoreML model.
    :param input_model_path:
    :return:
    """
    out_path = re.sub(r"h5$", 'mlmodel', input_model_path)

    hack_coremltools()

    with CustomObjectScope(custom_objects()):
        model = load_model(input_model_path)
        # https://github.com/akirasosa/mobile-semantic-segmentation/issues/6#issuecomment-344508193
        coreml_model = coremltools.converters.keras.convert(model,
                                                            input_names='image',
                                                            image_input_names='image',
                                                            red_bias=29.24429131 / 64.881128947,
                                                            green_bias=29.24429131 / 64.881128947,
                                                            blue_bias=29.24429131 / 64.881128947,
                                                            image_scale=1. / 64.881128947)
    coreml_model.save(out_path)

    print('CoreML model is created at %s' % out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_model_path',
        type=str,
        default='artifacts/mu_128_1_025.h5',
    )
    args, _ = parser.parse_known_args()

    main(**vars(args))
