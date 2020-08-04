FROM akirasosa/ubuntu:0.2.0

RUN pip install -U pip \
  && pip install -U \
  Pillow \
  albumentations \
  coremltools==4.0b2 \
  dacite \
  matplotlib \
  numpy \
  omegaconf \
  onnx \
  onnx-coreml \
  onnx-simplifier \
  pandas \
  pytorch-lightning \
  scikit-learn \
  scipy \
  timm \
  torch \
  torch_optimizer \
  torchvision \
  tqdm \
  && rm -rf ~/.cache/pip

USER root

