# warp_test.py
import cv2, numpy as np
from test_depth import transform, model  # or re-load model similarly
import torch
import matplotlib.pyplot as plt

# (For clarity, re-load MiDaS here or import from test_depth)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

def estimate_depth(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
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
    return (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

def warp_frame(frame_bgr, depth_norm, shift_px):
    H,W = depth_norm.shape
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    depth_inv = 1.0 - depth_norm
    disparity = depth_inv * float(shift_px)
    map_x = (xs + disparity).astype(np.float32)
    map_y = ys.astype(np.float32)
    warped = cv2.remap(frame_bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped

img = cv2.imread("WIN_20250902_19_49_11_Pro.jpg")
d = estimate_depth(img)
left = warp_frame(img, d, -30)   # left eye
right = warp_frame(img, d, 30)   # right eye

cv2.imwrite("left.png", left)
cv2.imwrite("right.png", right)

# side-by-side
left_rgb = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
right_rgb = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)
sbs = np.concatenate((left_rgb, right_rgb), axis=1)
plt.imsave("sbs.png", sbs)
print("Wrote left.png right.png sbs.png")

def inpaint_image(img_bgr):
    mask = np.all(img_bgr == 0, axis=2).astype('uint8') * 255
    if mask.sum() == 0:
        return img_bgr
    return cv2.inpaint(img_bgr, mask, 3, cv2.INPAINT_TELEA)

left_in = inpaint_image(left)
right_in = inpaint_image(right)
cv2.imwrite("left_inpaint.png", left_in)
cv2.imwrite("right_inpaint.png", right_in)
