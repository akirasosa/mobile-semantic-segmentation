# Real-Time Semantic Segmentation in Mobile device

This project is an example project of semantic segmentation for mobile real-time app.

The architecture is inspired by [MobileNets](https://arxiv.org/abs/1704.04861) and [U-Net](https://arxiv.org/abs/1505.04597).

[LFW, Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/part_labels/), is used as a Dataset.

The goal of this project is to detect hair segments with reasonable **accuracy and speed in mobile device**. Currently, it achieves 0.89 IoU.

About speed vs accuracy, more details are available at [my post](https://medium.com/vitalify-asia/real-time-deep-learning-in-mobile-application-25cf601a8976).

![Example of predicted image.](assets/prediction.png)

## Example application

* [iOS](https://github.com/vfa-tranhv/MobileAILab-HairColor-iOS)
* Android (TODO)

## Requirements

* Keras 2
* TensorFlow as a backend of Keras and for Android app.
* CoreML for iOS app.

## About Model

At this time, there is only one model in this repository, [MobileUNet.py](nets/MobileUNet.py). As a typical U-Net architecture, it has encoder and decoder parts, which consist of depthwise conv blocks proposed by MobileNets.

Input image is encoded to 1/32 size, and then decoded to 1/2. Finally, it scores the results and make it to original size.

Beside the U-Net like model, [PSPNet](https://arxiv.org/abs/1612.01105) like model was also tried. But it did not make a good result. Probably, global context does not have so much importance in the problem of hair recognition.

## Steps to training

### Data Preparation

Data is available at LFW. To get mask images, refer [issue #11](../../issues/11) for more. After you got images and masks, put the images of faces and masks as shown below.
```
data/
  raw/
    images/
      0001.jpg
      0002.jpg
    masks/
      0001.ppm
      0002.ppm
```

Then, convert it to numpy binary format for portability.
```
python data.py --img_size=128
```

Data augmentation will be done on the fly during training phase. I used rotation, shear ,zoom and horizontal flip. 


### Training

This repository contains three kinds of training scripts, transfer learning, fine tuning and full training. MobileNets is so compact that it's possible to try full training in a short time.

```
# Full training
python train_full.py \
  --img_file=/path/to/images.npy \
  --mask_file=/path/to/masks.npy
```

Dice coefficient is used as a loss function. Some other metrics are used such as precision, recall and binary cross entropy. Loss can be decreased soon smoothly even with high learning rate.

I also tried adding aux loss by using the segment of face part. Though, still I have not fully examined the effect of it, there maybe a little improvement of accuracy **without dropping inference speed**.


## Converting

As the purpose of this project is to make model run in mobile device, this repository contains some scripts to convert models for iOS and Android.

* [coreml-converter.py](coreml-converter.py)
  * It converts trained hdf5 model to CoreML model for iOS app.
* [coreml-converter-bench.py](coreml-converter-bench.py)
  * It generates non-trained CoreML model. It's useful to measure the inference speed in iOS device.
* [tf-converter.py](tf-converter.py)
  * It converts trained hdf5 model to protocol buffer format for TensorFlow which is used in Android app.


## TBD

- [x] Report speed vs accuracy in mobile device.
- [ ] Example app for Android
- [ ] Aux loss
- [ ] Some more optimizations??


