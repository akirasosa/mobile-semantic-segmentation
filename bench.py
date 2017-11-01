import time

import numpy as np

from nets.MobileUNet import MobileUNet

img_size = 128
batch_num = 16


def main():
    """
    Benchmark your model in your local pc.
    """
    model = MobileUNet(input_shape=(img_size, img_size, 3))
    inputs = np.random.randn(batch_num, img_size, img_size, 3)

    time_per_batch = []

    for i in range(10):
        start = time.time()
        model.predict(inputs, batch_size=batch_num)
        elapsed = time.time() - start
        time_per_batch.append(elapsed)

    time_per_batch = np.array(time_per_batch)

    # exclude 1st measure
    print(time_per_batch[1:].mean())
    print(time_per_batch[1:].std())


if __name__ == '__main__':
    main()
