import cv2
import numpy as np
import torch

def preprocess_image(img):
    """
    Preprocess a CARLA camera image to match training pipeline.
    Crop sky/hood → resize to (66,200) → normalize.
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    top = int(h * 0.35)
    bottom = int(h * 0.9)
    img = img[top:bottom, :, :]

    img = cv2.resize(img, (200, 66))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # CHW
    return img


def to_tensor(img):
    """Convert np array image to PyTorch tensor."""
    return torch.tensor(img, dtype=torch.float32).unsqueeze(0)


def smooth_steering(prev, new, factor=0.2):
    """
    Linear interpolation for smoother steering.
    factor = 0 means no new steering, 1 means instant override.
    """
    return prev + factor * (new - prev)


# ---------- Metrics ----------
def running_average(prev_avg, new_val, count):
    """
    Online running average:
    avg_new = (prev_avg * count + new_val) / (count + 1)
    """
    return (prev_avg * count + new_val) / (count + 1)
