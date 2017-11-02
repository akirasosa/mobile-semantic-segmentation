from __future__ import print_function

import argparse
import os

from keras import callbacks, optimizers

from data import load_data
from learning_rate import create_lr_schedule
from loss import dice_coef_loss, dice_coef, recall, precision
from nets.MobileUNet import MobileUNet

checkpoint_path = 'artifacts/checkpoint_weights.{epoch:02d}-{val_loss:.2f}.h5'
trained_model_path = 'artifacts/model.h5'

nb_train_samples = 2341
nb_validation_samples = 586


def train(img_file, mask_file, epochs, batch_size):
    train_gen, validation_gen, img_shape = load_data(img_file, mask_file)

    img_height = img_shape[0]
    img_width = img_shape[1]
    lr_base = 0.01 * (float(batch_size) / 16)

    model = MobileUNet(input_shape=(img_height, img_width, 3),
                       alpha=1,
                       alpha_up=0.25)

    model.summary()
    model.compile(
        optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
        # optimizer=Adam(lr=0.001),
        # optimizer=optimizers.RMSprop(),
        loss=dice_coef_loss,
        metrics=[
            dice_coef,
            recall,
            precision,
            'binary_crossentropy',
        ],
    )

    # callbacks
    scheduler = callbacks.LearningRateScheduler(
        create_lr_schedule(epochs, lr_base=lr_base, mode='progressive_drops'))
    tensorboard = callbacks.TensorBoard(log_dir='./logs')
    csv_logger = callbacks.CSVLogger('logs/training.csv')
    checkpoint = callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                           save_weights_only=True,
                                           save_best_only=True)

    model.fit_generator(
        generator=train_gen(),
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_gen(),
        validation_steps=nb_validation_samples // batch_size,
        # callbacks=[tensorboard, checkpoint, csv_logger],
        callbacks=[scheduler, tensorboard, checkpoint, csv_logger],
    )

    model.save(trained_model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--img_file',
        type=str,
        default='data/images-128.npy',
        help='image file as numpy format'
    )
    parser.add_argument(
        '--mask_file',
        type=str,
        default='data/masks-128.npy',
        help='mask file as numpy format'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=250,
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
    )
    args, _ = parser.parse_known_args()

    if not os.path.exists('artifacts'):
        os.makedirs('artifacts')

    train(**vars(args))
