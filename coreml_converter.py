import re

import onnx
import torch
from onnx import onnx_pb
from onnx_coreml import convert
from onnx_tf.backend import prepare

from nets.ImgWrapNet import ImgWrapNet

# %%
IMG_SIZE = 224

TMP_ONNX = 'tmp/tmp.onnx'
WEIGHT_PATH = 'outputs/train_unet/0-best.pth'
ML_MODEL = re.sub('\.pth$', '.mlmodel', WEIGHT_PATH)
TF_MODEL = re.sub('\.pth$', '.pb', WEIGHT_PATH)

# %%
# Convert to ONNX once
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImgWrapNet(torch.load(WEIGHT_PATH, map_location='cpu'))
model.to(device)

torch.onnx.export(model,
                  torch.randn(1, 3, IMG_SIZE, IMG_SIZE),
                  TMP_ONNX)

# %%
# Print out ONNX model to confirm the number of output layer
onnx_model = onnx.load(TMP_ONNX)
print(onnx_model)

# %%
# Convert ONNX to CoreML model
model_file = open(TMP_ONNX, 'rb')
model_proto = onnx_pb.ModelProto()
model_proto.ParseFromString(model_file.read())
# 595 is the identifier of output.
coreml_model = convert(model_proto,
                       image_input_names=['0'],
                       image_output_names=['595'])
coreml_model.save(ML_MODEL)

# %%
# tf_rep = prepare(onnx_model)  # prepare tf representation
# tf_rep.export_graph(TF_MODEL)  # export the model
