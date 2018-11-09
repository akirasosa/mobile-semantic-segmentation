import re

import onnx
import torch
from onnx import onnx_pb
from onnx_coreml import convert

from nets.ImgWrapNet import ImgWrapNet

# %%
TMP_ONNX = 'tmp/tmp.onnx'
WEIGHT_PATH = 'outputs/train_unet/0-best.pth'
ML_MODEL = re.sub('\.pth$', '.mlmodel', WEIGHT_PATH)

# %%
# Convert to ONNX once
img = torch.randn(1, 3, 224, 224)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImgWrapNet(torch.load(WEIGHT_PATH, map_location='cpu'))
model.to(device)

torch.onnx.export(model, img, TMP_ONNX)

# %%
# Print out ONNX model to confirm the number of output layer
onnx_model = onnx.load(TMP_ONNX)
print('The model is:\n{}'.format(onnx_model))

# %%
# Convert ONNX to CoreML model
model_file = open(TMP_ONNX, 'rb')
model_proto = onnx_pb.ModelProto()
model_proto.ParseFromString(model_file.read())
# 590 is the identifier of output.
coreml_model = convert(model_proto,
                       image_input_names=['0'],
                       image_output_names=['590'])
coreml_model.save(ML_MODEL)
