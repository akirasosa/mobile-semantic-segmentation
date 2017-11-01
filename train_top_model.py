import argparse
import os

from data import load_data
from loss import dice_coef_loss, dice_coef, recall, precision
from nets.MobileUNet import MobileUNet

top_model_weights_path = 'artifacts/top_model_weights.h5'
transferred_model_path = 'artifacts/transferred.h5'
nb_train_samples = 2341
nb_validation_samples = 586


def train(img_file, mask_file, mobilenet_weights_path, epochs, batch_size):
    train_gen, validation_gen, img_shape = load_data(img_file, mask_file)

    img_height = img_shape[0]
    img_width = img_shape[1]

    # model = MobileDeepLab(input_shape=(img_height, img_width, 3))
    model = MobileUNet(input_shape=(img_height, img_width, 3), alpha_up=0.25)
    model.load_weights(os.path.expanduser(mobilenet_weights_path.format(img_height)),
                       by_name=True)

    # Freeze mobilenet original weights
    for layer in model.layers[:82]:
        layer.trainable = False

    model.summary()
    model.compile(
        optimizer='rmsprop',
        loss=dice_coef_loss,
        metrics=[
            dice_coef,
            recall,
            precision,
            'binary_crossentropy',
        ],
    )

    model.fit_generator(
        generator=train_gen(),
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_gen(),
        validation_steps=nb_validation_samples // batch_size,
    )

    model.save_weights(top_model_weights_path)
    model.save(transferred_model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--img_file',
        type=str,
        default='data/images.npy',
        help='image file as numpy format'
    )
    parser.add_argument(
        '--mask_file',
        type=str,
        default='data/masks.npy',
        help='mask file as numpy format'
    )
    parser.add_argument(
        '--mobilenet_weights_path',
        type=str,
        default='~/.keras/models/mobilenet_1_0_{}_tf_no_top.h5',
        help='mobilenet weights using imagenet which is available at keras page'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
    )
    args, _ = parser.parse_known_args()

    if not os.path.exists('artifacts'):
        os.makedirs('artifacts')

    train(**vars(args))
