import argparse, imageio, cv2, numpy as np, torch, subprocess, sys
from tqdm import tqdm
from simple_lama_inpainting import SimpleLama
from PIL import Image

# -------------------------
# MiDaS loader + helpers
# -------------------------
def load_midas(device_str="cuda"):
    device = torch.device(device_str if torch.cuda.is_available() and device_str=="cuda" else "cpu")
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = transforms.small_transform
    model.to(device).eval()
    return model, transform, device

def estimate_depth(frame_bgr, model, transform, device, max_res=480):
    h, w = frame_bgr.shape[:2]
    scale = max_res / max(h, w)
    frame_small = (cv2.resize(frame_bgr, (int(w*scale), int(h*scale))) if scale < 1.0 else frame_bgr.copy())
    img_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
    input_tensor = transform(img_rgb).to(device)
    with torch.no_grad():
        prediction = model(input_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1), size=img_rgb.shape[:2],
            mode="bicubic", align_corners=False
        ).squeeze()
    depth = prediction.cpu().numpy()
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth_up = cv2.resize(depth_norm, (w, h), interpolation=cv2.INTER_CUBIC)
    return depth_up

def warp_frame(frame_bgr, depth_norm, shift_px):
    H, W = depth_norm.shape
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    depth_inv = 1.0 - depth_norm
    disparity = depth_inv * float(shift_px)
    map_x = (xs + disparity).astype(np.float32)
    map_y = ys.astype(np.float32)
    return cv2.remap(frame_bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)

# -------------------------
# LaMa inpainting
# -------------------------
lama = SimpleLama()

def inpaint_image(img_bgr):
    mask = np.all(img_bgr == 0, axis=2).astype('uint8') * 255
    if mask.sum() == 0:
        return img_bgr
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    mask_pil = Image.fromarray(mask).convert("L")
    result_pil = lama(img_pil, mask_pil)
    return cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)

# -------------------------
# Main video pipeline
# -------------------------
def main(args):
    model, transform, device = load_midas(args.device)

    cap = cv2.VideoCapture(args.input)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    tmp_output = args.output.replace(".mp4", "_raw.mp4")

    writer = imageio.get_writer(
        tmp_output, fps=fps, codec='libx264',
        ffmpeg_params=['-pix_fmt', 'yuv420p']
    )

    prev_depth = None
    pbar = tqdm(total=total_frames, desc="Processing frames")

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        depth = estimate_depth(frame_bgr, model, transform, device)
        if prev_depth is not None:
            depth = args.smoothing * prev_depth + (1.0 - args.smoothing) * depth
        prev_depth = depth.copy()

        left = warp_frame(frame_bgr, depth, -args.baseline)
        right = warp_frame(frame_bgr, depth, args.baseline)

        left = inpaint_image(left)
        right = inpaint_image(right)

        stereo = np.concatenate((
            cv2.cvtColor(left, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(right, cv2.COLOR_BGR2RGB)
        ), axis=1)

        writer.append_data(stereo)
        pbar.update(1)

    cap.release()
    writer.close()
    pbar.close()

    # Inject VR180 metadata (overwrite final output)
    try:
        subprocess.run([
            sys.executable, "-m", "spatialmedia",
            "-i", "--stereo=left-right",
            tmp_output, args.output
        ], check=True)
        print("✅ VR180 video saved:", args.output)
    except Exception as e:
        print("⚠️ Metadata injection failed. Install spatial-media first:")
        print("    pip install git+https://github.com/google/spatial-media.git")
        print("Error:", e)

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline", type=int, default=30)
    parser.add_argument("--smoothing", type=float, default=0.6)
    parser.add_argument("--device", choices=["cuda", "cpu"],
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    main(args)
