"""
reid_client.py — Re-ID embedding extraction via Triton OSNet.

OSNet-x1.0 (Market-1501 pretrained):
  - Input:  [B, 3, 256, 128] — height=256, width=128 (portrait crop)
  - Output: [B, 512] — L2-normalized embedding
  - Platform: onnxruntime_onnx in Triton

ImageNet normalization applied (same as training pipeline).
"""
import cv2
import numpy as np
import tritonclient.grpc.aio as grpcclient
import os

TRITON_URL   = os.getenv("TRITON_GRPC_URL", "localhost:8001")
REID_MODEL   = "osnet"
REID_INPUT   = "images"
REID_OUTPUT  = "output"
REID_H, REID_W = 256, 128   # OSNet portrait input (height × width)

# ImageNet mean/std (same normalization used during OSNet training)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


async def extract_embedding(image: np.ndarray, bbox: list) -> list | None:
    """
    Extract a 512-dim L2-normalized Re-ID embedding for the object crop.

    Args:
        image: BGR frame from cv2 (full frame)
        bbox:  [x1, y1, x2, y2] bounding box in pixel coordinates

    Returns:
        list[float] of length 512, or None on error.
    """
    try:
        x1, y1, x2, y2 = map(int, bbox)
        crop = image[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        if crop.size == 0:
            return None

        # Resize to OSNet input — portrait 256×128 (preserving aspect ratio with padding)
        h, w = crop.shape[:2]
        r = min(REID_W / w, REID_H / h)
        new_w, new_h = int(round(w * r)), int(round(h * r))
        if (w, h) != (new_w, new_h):
            resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            resized = crop
        dw = REID_W - new_w
        dh = REID_H - new_h
        top, left = dh // 2, dw // 2
        bottom, right = dh - top, dw - left
        crop = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(127, 127, 127))
        # BGR → RGB
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        # HWC → CHW, normalize to [0,1] then ImageNet-normalize
        crop = crop.transpose((2, 0, 1)).astype(np.float32) / 255.0
        crop = (crop - _MEAN) / _STD

        input_data = np.expand_dims(crop, axis=0)   # [1, 3, 256, 128]

        client = grpcclient.InferenceServerClient(url=TRITON_URL)
        inputs  = [grpcclient.InferInput(REID_INPUT, input_data.shape, "FP32")]
        inputs[0].set_data_from_numpy(input_data)
        outputs = [grpcclient.InferRequestedOutput(REID_OUTPUT)]

        result    = await client.infer(model_name=REID_MODEL, inputs=inputs, outputs=outputs)
        embedding = result.as_numpy(REID_OUTPUT)[0]   # [512]

        # L2 normalize (OSNet already normalizes internally, but enforce it)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.tolist()

    except Exception as e:
        print(f"[ReID] extract_embedding error: {e}")
        return None
