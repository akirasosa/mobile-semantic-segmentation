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
        coreml_model = coremltools.converters.keras.convert(model, input_names='data')
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
