import coremltools as ct
import torch

from mobile_seg.const import TMP_DIR, EXP_DIR
from mobile_seg.modules.net import load_trained_model
from mobile_seg.modules.wrapper import Wrapper

# %%
IMG_SIZE = 224

TMP_ONNX = TMP_DIR / 'tmp.onnx'
CKPT_PATH = EXP_DIR / 'mobile_seg/1603164686/checkpoints/epoch=179.ckpt'

# %%
unet = load_trained_model(CKPT_PATH)
model = Wrapper(unet=unet).eval()

# %%
inputs = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
traced_model = torch.jit.trace(model, inputs)

# %%
model = ct.convert(
    traced_model,
    inputs=[ct.ImageType(name="input_1", shape=inputs.shape)],
)
model.save(TMP_DIR / 'MobileNetV2_unet.mlmodel')
