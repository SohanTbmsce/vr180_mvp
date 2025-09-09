# test_depth.py
import torch, cv2, numpy as np
import matplotlib.pyplot as plt

# Load MiDaS small (fast)
model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = transforms.small_transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model.to(device).eval()

img_path = "WIN_20250902_19_49_11_Pro.jpg"   # put a test jpg here
img_bgr = cv2.imread(img_path)
if img_bgr is None:
    raise SystemExit("Place sample.jpg in the folder and re-run")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# transform + infer
input_tensor = transform(img_rgb).to(device)
with torch.no_grad():
    prediction = model(input_tensor)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img_rgb.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

depth = prediction.cpu().numpy()
depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

plt.imsave("depth_norm.png", depth_norm, cmap="inferno")
print("Saved depth_norm.png")
