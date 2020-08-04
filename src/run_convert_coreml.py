import coremltools as ct
import onnx
import torch
from onnxsim import simplify

from mobile_seg.const import TMP_DIR, EXP_DIR
from mobile_seg.modules.net import load_trained_model
from mobile_seg.modules.wrapper import Wrapper

# %%
IMG_SIZE = 224

CKPT_PATH = EXP_DIR / 'mobile_seg/1596704750/checkpoints/epoch=194.ckpt'
TMP_ONNX = TMP_DIR / 'tmp.onnx'
TMP_OPT_ONNX = TMP_DIR / 'opt.onnx'
COREML_OUT = TMP_DIR / 'opt.mlmodel'

# %%
unet = load_trained_model(CKPT_PATH)
model = Wrapper(unet=unet).eval()

# %%
inputs = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
torch.onnx.export(
    model, inputs, TMP_ONNX,
    verbose=True,
    input_names=['input0'],
    output_names=['output'],
)

# %%
onnx_opt = simplify(str(TMP_ONNX))
onnx.save_model(onnx_opt[0], str(TMP_OPT_ONNX))

# %%

coreml_model = ct.converters.onnx.convert(
    model=str(TMP_OPT_ONNX),
)
coreml_model.save(COREML_OUT)
