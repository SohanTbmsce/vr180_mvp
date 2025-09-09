# streamlit_vr180_mvp.py
"""
Streamlit app: 2D -> VR180 stereo (side-by-side) converter
Features:
 - MiDaS_small depth with optional fp16 on CUDA
 - simple depth-based stereo synthesis
 - optional LaMa inpainting (fast fallback to OpenCV)
 - smart inpainting (skip LaMa if hole area small)
 - resolution/FPS downsample + frame-skip
 - automatic 10s sample export, GIF preview, inline video player
 - VR180 metadata injection (spatial-media) with verification
 - Progress bar with ETA/frame counts and demo mode
"""

import os, sys, tempfile, time, subprocess
from pathlib import Path
import streamlit as st
import numpy as np
from PIL import Image
import cv2
import imageio
import torch

# try to import simple-lama-inpainting wrapper
try:
    from simple_lama_inpainting import SimpleLama
    LAMA_AVAILABLE = True
    LAMA = SimpleLama()
except Exception:
    LAMA_AVAILABLE = False
    LAMA = None

# Constants
DEFAULT_DEPTH_RES = 256
SAMPLE_SECONDS = 10
SMART_INPAINT_THRESHOLD = 0.02  # fraction of pixels missing to trigger heavy inpainting

# ---------------------
# Utilities & model
# ---------------------
@st.cache_resource
def load_midas(device_str="cuda", use_fp16=True):
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    # Load MiDaS small for speed
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = transforms.small_transform
    model.to(device).eval()
    # try switch to half precision for speed when on CUDA
    if device.type == "cuda" and use_fp16:
        try:
            model.half()
        except Exception:
            pass
    return model, transform, device

def estimate_depth(frame_bgr, model, transform, device, max_res=DEFAULT_DEPTH_RES):
    # Resize for speed
    h, w = frame_bgr.shape[:2]
    scale = max_res / max(h, w)
    if scale < 1.0:
        frame_small = cv2.resize(frame_bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    else:
        frame_small = frame_bgr.copy()
    img_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
    input_tensor = transform(img_rgb).to(device)
    # convert to half if device & model support it
    if device.type == "cuda":
        try:
            input_tensor = input_tensor.half()
        except Exception:
            pass
    with torch.no_grad():
        prediction = model(input_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()
    depth = prediction.float().cpu().numpy()
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    # upsample back to original resolution
    depth_up = cv2.resize(depth_norm, (w, h), interpolation=cv2.INTER_CUBIC)
    return depth_up

def warp_frame(frame_bgr, depth_norm, shift_px):
    H, W = depth_norm.shape
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    disparity = (1.0 - depth_norm) * float(shift_px)
    map_x = (xs + disparity).astype(np.float32)
    map_y = ys.astype(np.float32)
    warped = cv2.remap(frame_bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped

# Inpainting helpers
def opencv_inpaint(img_bgr):
    mask = np.all(img_bgr == 0, axis=2).astype('uint8') * 255
    if mask.sum() == 0:
        return img_bgr
    return cv2.inpaint(img_bgr, mask, 3, cv2.INPAINT_TELEA)

def lama_inpaint(img_bgr):
    # expects BGR uint8, returns BGR uint8
    mask = np.all(img_bgr == 0, axis=2).astype('uint8') * 255
    if mask.sum() == 0:
        return img_bgr
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    mask_pil = Image.fromarray(mask).convert("L")
    out_pil = LAMA(img_pil, mask_pil)
    out = cv2.cvtColor(np.array(out_pil), cv2.COLOR_RGB2BGR)
    return out

def smart_inpaint(img_bgr, use_lama):
    mask = np.all(img_bgr == 0, axis=2)
    hole_frac = float(mask.mean())
    if hole_frac <= 0:
        return img_bgr, hole_frac, "none"
    # if LaMa not available or user disabled it, use OpenCV
    if not use_lama or not LAMA_AVAILABLE:
        return opencv_inpaint(img_bgr), hole_frac, "opencv"
    # if hole area small, skip slow LaMa and use OpenCV
    if hole_frac < SMART_INPAINT_THRESHOLD:
        return opencv_inpaint(img_bgr), hole_frac, "opencv_small"
    # otherwise use LaMa
    try:
        out = lama_inpaint(img_bgr)
        return out, hole_frac, "lama"
    except Exception:
        # fallback
        return opencv_inpaint(img_bgr), hole_frac, "opencv_failed"

# ---------------------
# Main processing
# ---------------------
def process_video_streamlit(
    input_path, output_path,
    baseline, smoothing,
    depth_res, skip, use_lama, gif_frames,
    max_frames=None, demo_mode=False, fps_override=None,
    sample_seconds=SAMPLE_SECONDS, progress_updater=None
):
    model, transform, device = load_midas("cuda", use_fp16=True)
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video file")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    fps = orig_fps if not fps_override else fps_override

    tmp_raw = output_path.replace(".mp4", "_raw.mp4")
    # main writer only if not demo_mode
    writer = None
    if not demo_mode:
        writer = imageio.get_writer(tmp_raw, fps=fps/skip, codec='libx264', ffmpeg_params=['-pix_fmt', 'yuv420p'])

    # sample writer for quick demo sample (first sample_seconds)
    sample_writer = None
    sample_limit = 0
    if (not demo_mode) and sample_seconds > 0:
        sample_path = output_path.replace(".mp4", f"_sample{sample_seconds}s.mp4")
        sample_writer = imageio.get_writer(sample_path, fps=fps/skip, codec='libx264', ffmpeg_params=['-pix_fmt', 'yuv420p'])
        sample_limit = int(sample_seconds * fps)

    prev_depth = None
    max_frames = max_frames if (max_frames and max_frames > 0) else total_frames
    max_frames = max_frames if max_frames > 0 else total_frames

    preview_stack = None
    gif_list = []

    processed = 0
    frame_idx = 0
    start = time.time()
    # compute steps for progress
    total_steps = (max_frames + skip - 1) // skip if max_frames else max(1, (total_frames + skip -1)//skip)
    if total_steps <= 0: total_steps = 1

    if progress_updater:
        progress_updater(0, total_steps, "Starting...")

    while frame_idx < max_frames:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        # skip frames for speed
        if frame_idx % skip != 0:
            frame_idx += 1
            continue

        # depth
        depth = estimate_depth(frame_bgr, model, transform, device, max_res=depth_res)
        if prev_depth is not None:
            depth = smoothing * prev_depth + (1.0 - smoothing) * depth
        prev_depth = depth.copy()

        # synthesize stereo
        left = warp_frame(frame_bgr, depth, -baseline)
        right = warp_frame(frame_bgr, depth, baseline)

        # inpainting (smart)
        if use_lama and LAMA_AVAILABLE:
            left, lf, left_method = smart_inpaint(left, use_lama)
            right, rf, right_method = smart_inpaint(right, use_lama)
        else:
            left = opencv_inpaint(left)
            right = opencv_inpaint(right)

        # stereo RGB frame for writer (imageio expects RGB)
        stereo_rgb = np.concatenate((cv2.cvtColor(left, cv2.COLOR_BGR2RGB), cv2.cvtColor(right, cv2.COLOR_BGR2RGB)), axis=1)

        # write out
        if writer is not None:
            writer.append_data(stereo_rgb)
        if sample_writer is not None and processed < sample_limit:
            sample_writer.append_data(stereo_rgb)

        # collect GIF frames if requested
        if gif_frames and processed < gif_frames:
            gif_list.append(stereo_rgb)

        # build preview stack for first processed frame
        if processed == 0:
            orig_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            target_w = stereo_rgb.shape[1]
            # resize original to match stereo width
            h_o, w_o = orig_rgb.shape[:2]
            new_h = int(h_o * (target_w / w_o))
            orig_resized = cv2.resize(orig_rgb, (target_w, new_h), interpolation=cv2.INTER_AREA)
            # build stacked image: original on top, stereo below
            stacked_h = orig_resized.shape[0] + stereo_rgb.shape[0]
            stacked = np.zeros((stacked_h, target_w, 3), dtype=np.uint8)
            stacked[0:orig_resized.shape[0], :, :] = orig_resized
            stacked[orig_resized.shape[0]:orig_resized.shape[0]+stereo_rgb.shape[0], :, :] = stereo_rgb
            preview_stack = stacked

        processed += 1
        frame_idx += 1

        # progress update
        elapsed = time.time() - start
        if progress_updater:
            progress_updater(processed, total_steps, f"Processed {processed}/{total_steps} frames — {elapsed:.1f}s elapsed")

    cap.release()
    if writer is not None:
        writer.close()
    if sample_writer is not None:
        sample_writer.close()

    total_elapsed = time.time() - start

    # save gif if requested
    gif_path = None
    if gif_list:
        gif_path = output_path.replace(".mp4", "_preview.gif")
        imageio.mimsave(gif_path, gif_list, fps=min(10, orig_fps))

    # VR metadata injection if we produced main file and not demo_mode
    final_output = output_path
    if (writer is not None) and (not demo_mode):
        try:
            # inject metadata using spatialmedia
            subprocess.run([sys.executable, "-m", "spatialmedia", "-i", "--stereo=left-right", tmp_raw, final_output], check=True)
            injected = True
        except Exception:
            # fallback: move raw to final
            try:
                os.replace(tmp_raw, final_output)
            except Exception:
                pass
            injected = False
    else:
        # demo mode: no final video written
        final_output = output_path  # likely doesn't exist
        injected = False

    return preview_stack, final_output, processed, total_elapsed, gif_path, injected

# ---------------------
# Streamlit UI layout
# ---------------------
st.set_page_config(page_title="VR180 Immersive Generator", layout="wide")
st.title("🤖 VR180 Immersive Video Generator — Hackathon MVP")

# Top instructions
with st.expander("How to use (quick) — follow these steps"):
    st.markdown("""
    1. Upload a short clip (3–15s) — smaller resolution is faster.
    2. Adjust **Stereo baseline** and **Depth smoothing** (defaults work well).
    3. Use **Demo Mode** for a very fast GIF preview for presentation.
    4. Click **Process** — wait for progress/ETA.
    5. Preview and **Download** the final VR180 video (or GIF).
    
    **Notes:**  
    - If you want highest quality, enable LaMa inpainting and a larger depth resolution (but it will be slower).  
    - For demo, keep depth resolution low (128–256) and skip frames (2) to speed up.
    """)

# show GPU/LaMa status
col_status1, col_status2 = st.columns(2)
with col_status1:
    cuda_ok = torch.cuda.is_available()
    st.write("**GPU (CUDA)**:", "✅ Available" if cuda_ok else "❌ Not available")
    if cuda_ok:
        try:
            st.caption(f"CUDA device: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass
with col_status2:
    st.write("**LaMa available**:", "✅ Yes" if LAMA_AVAILABLE else "❌ No")
    if not LAMA_AVAILABLE:
        st.caption("Install `simple-lama-inpainting` to enable high-quality inpainting.")

st.markdown("---")

# settings + upload panel
left, right = st.columns([1, 1])
with left:
    uploaded = st.file_uploader("Upload video (.mp4/.mov/.avi) — small clips recommended", type=["mp4", "mov", "avi"])
    demo_mode = st.checkbox("⚡ Demo Mode (fast GIF only)", value=False)
    use_lama = st.checkbox("Use LaMa inpainting (slower, cleaner)", value=False)
    if use_lama and not LAMA_AVAILABLE:
        st.error("LaMa not available — install `simple-lama-inpainting`.")
with right:
    # --- UI Controls (Optimized for Speed Demo) ---
    st.sidebar.header("⚡ Processing Settings (Demo-Optimized)")

    baseline = st.sidebar.slider( "Stereo baseline (px)", min_value=8, max_value=80, value=30)

    smoothing = st.sidebar.slider("Depth smoothing", min_value=0.0, max_value=1.0, value=0.6)

# ⚡ Default lower depth resolution = much faster
    depth_res = st.sidebar.selectbox("Depth resolution (lower=faster)", [128, 256, 384, 512], index=0)

# ⚡ Default skip=2 → halves processing time
    skip = st.sidebar.slider("Frame skip (1=every frame)", min_value=1, max_value=5, value=2)

# Limit frames for GIF preview
    gif_frames = st.sidebar.slider("GIF preview frames (0=off)", min_value=0, max_value=40, value=12)

# Toggle for LaMa inpainting (default: off, since it's slow)
    use_lama = st.sidebar.checkbox("Use LaMa inpainting (slower but cleaner)", value=False)

# Max frames (0 = full video). Default capped at 150 frames for speed.
    max_frames = st.sidebar.number_input("Max frames to process (0=all)", min_value=0, value=150, step=50)


if demo_mode:
    st.info("Demo Mode: 30 frames, skip=2, inpainting OFF, GIF preview enabled.")
    max_frames = 30
    skip = 2
    use_lama = False
    gif_frames = max(8, gif_frames)
else:
    max_frames = st.number_input("Max frames to process (0 = all)", min_value=0, value=0)

# Process button & progress placeholders
process_btn = st.button("🚀 Process Video")
progress_bar = st.progress(0)
progress_text = st.empty()
preview_slot = st.empty()
downloads_col = st.empty()

# helper to update progress from processing
def update_progress(done, total, msg):
    frac = min(1.0, done / max(1, total))
    progress_bar.progress(frac)
    progress_text.info(msg)

# ----------------------------
# 🔑 Guard: Require video upload
# ----------------------------
if process_btn and uploaded is None:
    progress_text.error("❌ Please upload a video before clicking 'Process Video'.")
    st.stop()

# ----------------------------
# Run processing when clicked
# ----------------------------
if process_btn and uploaded is not None:
    # save temp input
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpf:
        tmpf.write(uploaded.read())
        tmp_input = tmpf.name

    out_dir = tempfile.gettempdir()
    out_name = f"out_vr180_{int(time.time())}.mp4"
    out_path = os.path.join(out_dir, out_name)

    # convert max_frames param
    max_frames_arg = None if (not demo_mode and (max_frames == 0)) else (max_frames if not demo_mode else 30)

    progress_text.info("⏳ Starting processing...")
    try:
        preview_img, final_path, processed, elapsed, gif_path, injected = process_video_streamlit(
            tmp_input, out_path,
            baseline=baseline, smoothing=smoothing,
            depth_res=int(depth_res), skip=int(skip),
            use_lama=use_lama, gif_frames=int(gif_frames),
            max_frames=(None if max_frames_arg is None else int(max_frames_arg)),
            demo_mode=demo_mode, fps_override=None,
            sample_seconds=SAMPLE_SECONDS,
            progress_updater=update_progress
        )
    except Exception as e:
        # 🔑 Show error in status bar AND as error message
        progress_text.error(f"❌ Processing failed: {e}")
        st.error(f"An error occurred: {e}")
        try:
            os.remove(tmp_input)
        except Exception:
            pass
        st.stop()

    # show preview stacked (original top, stereo bottom)
    if preview_img is not None:
        preview_slot.subheader("Preview (Top: original resized • Bottom: stereo)")
        preview_slot.image(preview_img, use_column_width=True)
    else:
        preview_slot.info("No preview available.")

    # summary & ETA
    if processed > 0:
        st.success(f"Done — processed {processed} frames in {elapsed:.1f}s ({elapsed/max(1,processed):.2f}s/frame).")

    # final video (only if not demo)
    if (not demo_mode) and os.path.exists(final_path):
        st.subheader("▶️ Final VR180 Video (Playable & Downloadable)")
        try:
            with open(final_path, "rb") as f:
                video_bytes = f.read()
                st.video(video_bytes)
                downloads_col.download_button("⬇️ Download VR180 Video", data=video_bytes, file_name=os.path.basename(final_path))
        except Exception:
            st.warning("Could not load final video into player — download is available below.")
            downloads_col.download_button("⬇️ Download VR180 Video", data=open(final_path, "rb"), file_name=os.path.basename(final_path))

        if injected:
            st.success("VR180 metadata injected (left-right).")
        else:
            st.warning("VR180 metadata injection either failed or was skipped; the output may still play but may not be recognized as VR180 by all players.")
    else:
        if demo_mode:
            st.info("Demo Mode: full VR180 export skipped (GIF preview available).")
        else:
            st.warning("Final VR180 output not found.")

    # GIF preview
    if gif_path and os.path.exists(gif_path):
        st.subheader("✨ GIF Preview (Quick demo)")
        st.image(gif_path, use_column_width=True)
        st.download_button("⬇️ Download GIF Preview", data=open(gif_path, "rb"), file_name=os.path.basename(gif_path))
    else:
        if gif_frames > 0:
            st.info("No GIF preview created (maybe gif_frames was 0 or not enough frames).")

    # cleanup temp input
    try:
        os.remove(tmp_input)
    except Exception:
        pass

    progress_text.text("Finished.")

# show helpful tips
st.markdown("---")
st.markdown("**Tips for demo**: use Demo Mode, low depth resolution (128–256), skip=2, small clip (5–10s).")
st.markdown("**Further improvements (optional for production):** move heavy processing to a GPU worker (FastAPI + Celery), replace simple warp with MPI or Softsplat for higher quality, add user accounts + queue for multi-user upload.")
