#!/usr/bin/env python3
"""
Export TransReID-SSL ViT-S (MSMT17) to ONNX for Triton Re-ID.

Architecture: Vision Transformer Small (ViT-S/16) fine-tuned with
TransReID-SSL self-supervised pretraining on LUPerson, then supervised
fine-tuning on MSMT17. Achieves ~88% Rank-1 vs OSNet-AIN's ~78%.

Strategy:
  1. Build ViT-S/16 with timm (already installed in container)
  2. Download TransReID-SSL MSMT17 checkpoint from GitHub releases
  3. Load backbone weights, strip classifier head
  4. Patch forward() to return L2-normalised 512-dim embedding
  5. Export to ONNX opset 18 (same input/output spec as OSNet config)

Input:  [B, 3, 256, 128]  float32
Output: [B, 512]          float32  (L2-normalised)

No Triton config.pbtxt changes needed — same spec as OSNet-AIN.
"""

import os, sys, subprocess, shutil, urllib.request, types
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
OUTDIR   = "/model_repo/osnet/1"
OUTPATH  = os.path.join(OUTDIR, "model.onnx")
WEIGHTS  = "/tmp/transreid_ssl_vits_msmt17.pth"

# TransReID-SSL ViT-S MSMT17 checkpoint from official damo-cv release
WEIGHTS_URL = (
    "https://github.com/damo-cv/TransReID-SSL/releases/download/v0.1.0/"
    "MSMT17_vit_s.pth"
)

os.makedirs(OUTDIR, exist_ok=True)
CONFIG_PATH = os.path.join(os.path.dirname(OUTDIR), "config.pbtxt")
if not os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "w") as f:
        f.write('''name: "osnet"
platform: "onnxruntime_onnx"
max_batch_size: 16
input [
  {
    name: "images"
    data_type: TYPE_FP32
    dims: [ 3, 256, 128 ]
  }
]
output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [ 512 ]
  }
]
instance_group [
  {
    count: 1
    kind: KIND_GPU
  }
]
''')
    print(f"Created config.pbtxt at {CONFIG_PATH}")

# ---------------------------------------------------------------------------

# Step 1 — ensure timm is available (pre-installed in API container)
try:
    import timm  # type: ignore
    print(f"timm version: {timm.__version__}")
except ImportError:
    print("Installing timm...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "timm", "-q"])
    import timm  # type: ignore

# Step 2 — download checkpoint
if not os.path.exists(WEIGHTS) or os.path.getsize(WEIGHTS) < 1_000_000:
    print("=== Downloading TransReID-SSL ViT-S MSMT17 checkpoint ===")
    try:
        import gdown  # type: ignore
        # Google Drive file ID from damo-cv/TransReID-SSL README
        gdown.download(id="1ODxA7mJv17UfzwfXtY9dTWNsYghoNWGB", output=WEIGHTS)
        print(f"Downloaded: {os.path.getsize(WEIGHTS) / 1024 / 1024:.1f} MB")
    except Exception as e:
        raise RuntimeError(
            f"Failed to download weights: {e}\n"
            "Run manually: python3 -c \"import gdown; gdown.download(id='1ODxA7mJv17UfzwfXtY9dTWNsYghoNWGB', output='/tmp/transreid_ssl_vits_msmt17.pth')\""
        )
else:
    print(f"Weights already present: {WEIGHTS} ({os.path.getsize(WEIGHTS)/1024/1024:.1f} MB)")


# Step 3 — build ViT-S backbone
print("=== Building TransReID-SSL ViT-S backbone ===")

class TransReIDEmbedder(nn.Module):
    """
    ViT-S backbone that outputs a 512-dim L2-normalised embedding.
    Matches the TransReID-SSL architecture: ViT-S/16 + BNNeck bottleneck.
    """
    def __init__(self):
        super().__init__()
        # ViT-S/16 with custom image size (256x128 for Re-ID)
        self.backbone = timm.create_model(
            "vit_small_patch16_224",
            pretrained=False,
            img_size=(256, 128),
            num_classes=0,       # remove classifier head -> outputs 384-dim CLS token
        )
        # Bottleneck: BN + Linear 384 -> 512 (TransReID-SSL convention)
        self.bottleneck = nn.BatchNorm1d(384)
        self.bottleneck.bias.requires_grad_(False)
        self.embed = nn.Linear(384, 512, bias=False)

    def forward(self, x):
        feat = self.backbone(x)               # [B, 384]
        feat = self.bottleneck(feat)           # BN
        emb  = self.embed(feat)               # [B, 512]
        return F.normalize(emb, p=2, dim=1)   # L2 norm


model = TransReIDEmbedder()

# Step 4 — load checkpoint
print("Loading TransReID-SSL checkpoint...")
ckpt = torch.load(WEIGHTS, map_location="cpu", weights_only=False)

# TransReID-SSL checkpoints use 'model' or 'state_dict' key
if "model" in ckpt:
    state = ckpt["model"]
elif "state_dict" in ckpt:
    state = ckpt["state_dict"]
else:
    state = ckpt

# Print top-level keys to understand checkpoint structure
top_keys = list(state.keys())[:10]
print(f"  Checkpoint top-level keys (first 10): {top_keys}")

# Strip common prefixes
def _strip_prefix(d, prefix):
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in d.items()}

state = _strip_prefix(state, "base_model.")
state = _strip_prefix(state, "module.")

# Separate backbone, bottleneck and embed keys
backbone_state = {k.replace("backbone.", ""): v for k, v in state.items() if k.startswith("backbone.")}
bn_state       = {k.replace("bottleneck.", ""): v for k, v in state.items() if k.startswith("bottleneck.")}
embed_state    = {k.replace("embed.", ""): v for k, v in state.items() if k.startswith("embed.")}

# Fallback: TransReID-SSL may store backbone without prefix
if not backbone_state:
    skip = {"classifier", "head", "bottleneck", "embed", "ID_classifier"}
    backbone_state = {k: v for k, v in state.items()
                      if not any(k.startswith(s) for s in skip)}

b_miss, b_unex = model.backbone.load_state_dict(backbone_state, strict=False)
print(f"  Backbone: {len(backbone_state)} keys loaded | missing={len(b_miss)} | unexpected={len(b_unex)}")
if b_miss:
    print(f"  Missing keys (first 5): {b_miss[:5]}")

if bn_state:
    model.bottleneck.load_state_dict(bn_state, strict=False)
    print("  BNNeck loaded")
if embed_state:
    model.embed.load_state_dict(embed_state, strict=False)
    print("  Embed head loaded")

model.eval()

# Step 5 — sanity check
dummy = torch.randn(1, 3, 256, 128)
with torch.no_grad():
    out = model(dummy)
print(f"Forward pass OK -> shape={out.shape}, L2-norm={out[0].norm().item():.6f}")
assert out.shape == (1, 512), f"Expected (1, 512), got {out.shape}"
assert abs(out[0].norm().item() - 1.0) < 1e-4, "Output is not L2-normalised!"

# Step 6 — determinism check
with torch.no_grad():
    out2 = model(dummy)
diff = (out - out2).abs().max().item()
print(f"Determinism check: max diff = {diff:.2e} (should be ~0)")
assert diff < 1e-6, "Model is not deterministic in eval mode!"

# Step 7 — export ONNX
print(f"=== Exporting to {OUTPATH} ===")
torch.onnx.export(
    model,
    dummy,
    OUTPATH,
    input_names=["images"],
    output_names=["output"],
    dynamic_axes={"images": {0: "batch"}, "output": {0: "batch"}},
    opset_version=18,
    do_constant_folding=True,
)
mb = os.path.getsize(OUTPATH) / 1024 / 1024
print(f"Saved: {OUTPATH} ({mb:.1f} MB)")

# Step 8 — verify ONNX graph
import onnx          # type: ignore
import onnxruntime   # type: ignore
model_proto = onnx.load(OUTPATH)
onnx.checker.check_model(model_proto)
print("ONNX graph check passed")

sess = onnxruntime.InferenceSession(OUTPATH, providers=["CPUExecutionProvider"])
np_dummy = dummy.numpy()
rt_out = sess.run(["output"], {"images": np_dummy})[0]
rt_norm = float((rt_out[0] ** 2).sum() ** 0.5)
print(f"ONNX Runtime test: shape={rt_out.shape}, L2-norm={rt_norm:.6f}")

print()
print("=== Done! ===")
print(f"Model : {OUTPATH} ({mb:.1f} MB)")
print()
print("IMPORTANT: Clear Qdrant gallery before using (old embeddings are incompatible):")
print("  DELETE http://api:8003/tracked/<each_id>  or reset the collection")
print()
print("Triton reload:")
print("  POST http://triton:8000/v2/repository/models/osnet/unload")
print("  POST http://triton:8000/v2/repository/models/osnet/load")
