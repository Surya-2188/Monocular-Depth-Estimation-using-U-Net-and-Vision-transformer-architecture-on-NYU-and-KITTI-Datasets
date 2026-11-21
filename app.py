from hashlib import sha256
from io import BytesIO
import math
from pathlib import Path
from zipfile import ZipFile

from matplotlib import colormaps
import numpy as np
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
import torch
from depth_models import UNetDepth, ViTDepth

ROOT = Path(__file__).resolve().parent
PROFILES = {
    "Indoor · NYU U-Net": (
        "stage1/nyu/results/20261121T211823767832Z.zip", "unet_best.pt", 192, 256, 10.0,
    ),
    "Outdoor · KITTI ViT": (
        "stage2/kitti/results/20261121T222116815973Z.zip", "vit_best.pt", 192, 640, 80.0,
    ),
}


@st.cache_resource
def load_model(profile, modified_at):
    archive, member, height, width, _ = PROFILES[profile]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetDepth() if member.startswith("unet") else ViTDepth(height, width)
    with ZipFile(ROOT / archive) as bundle:
        weights = torch.load(BytesIO(bundle.read(member)), map_location="cpu", weights_only=True)
    model.load_state_dict(weights)
    return model.to(device).eval(), device


@torch.inference_mode()
def estimate_depth(rgb, model, device, height, width):
    model_image = rgb.resize((width, height), Image.Resampling.BILINEAR)
    pixels = np.asarray(model_image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)
    return model(tensor.to(device))[0, 0].cpu().numpy()


@st.cache_data(max_entries=4)
def predict(image_bytes, profile, modified_at):
    with Image.open(BytesIO(image_bytes)) as source:
        rgb = ImageOps.exif_transpose(source).convert("RGB")
    _, _, height, width, _ = PROFILES[profile]
    model, device = load_model(profile, modified_at)
    return rgb, estimate_depth(rgb, model, device, height, width)


def color_depth(depth, maximum, size):
    normalized = np.clip(depth / maximum, 0, 1)
    colors = colormaps["plasma"](normalized, bytes=True)[..., :3]
    return Image.fromarray(colors).resize(size, Image.Resampling.BILINEAR)


def show_live_camera(model, device, height, width, maximum, profile):
    import av
    from streamlit_webrtc import WebRtcMode, webrtc_streamer

    def process_frame(frame):
        rgb = Image.fromarray(frame.to_ndarray(format="rgb24"))
        rgb.thumbnail((640, 480))
        depth = estimate_depth(rgb, model, device, height, width)
        heatmap = color_depth(depth, maximum, rgb.size)
        result = Image.new("RGB", (2 * rgb.width, rgb.height))
        result.paste(rgb, (0, 0))
        result.paste(heatmap, (rgb.width, 0))
        x, y = rgb.width + rgb.width // 2, rgb.height // 2
        draw = ImageDraw.Draw(result)
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="white", outline="black")
        center_depth = float(depth[height // 2, width // 2])
        draw.text((rgb.width + 8, 10), f"Center depth: {center_depth:.1f} m",
                  fill="white", stroke_fill="black", stroke_width=2)
        return av.VideoFrame.from_ndarray(np.asarray(result), format="rgb24")

    st.caption(f"Camera on the left, estimated depth on the right. Colour scale: 0–{maximum:g} m.")
    webrtc_streamer(
        key=f"camera-{profile}-{maximum}",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video": {"width": {"ideal": 640}, "frameRate": {"ideal": 10}},
                                  "audio": False},
        video_frame_callback=process_frame,
        async_processing=True,
    )


st.set_page_config(page_title="Depth Estimation", layout="wide")
st.title("Depth Estimation")
profile = st.radio("Scene", tuple(PROFILES), horizontal=True)
archive, _, HEIGHT, WIDTH, depth_limit = PROFILES[profile]
checkpoint = ROOT / archive
if not checkpoint.is_file():
    st.error(f"Missing checkpoint archive: {archive}")
    st.stop()
modified_at = checkpoint.stat().st_mtime_ns

mode = st.radio("Input", ("Upload", "Camera snapshot", "Live camera"), horizontal=True)
maximum = st.slider("Depth colour scale maximum (m)", 1.0, depth_limit, depth_limit, 1.0,
                    key=f"scale-{profile}")
st.caption("Estimated depth is along the camera's viewing axis. Images unlike the training scenes may give less accurate estimates.")

if mode == "Live camera":
    try:
        model, device = load_model(profile, modified_at)
        show_live_camera(model, device, HEIGHT, WIDTH, maximum, profile)
    except ImportError:
        st.error("Install the live camera dependencies with python -m pip install -r requirements.txt.")
    except (OSError, RuntimeError, KeyError) as exc:
        st.error(f"Could not start live depth: {exc}")
    st.stop()

if mode == "Upload":
    photo = st.file_uploader("Upload a photo", type=("png", "jpg", "jpeg", "webp"))
else:
    photo = st.camera_input("Take a photo")

if photo is None:
    st.stop()

image_bytes = photo.getvalue()
try:
    with st.spinner("Estimating depth…"):
        rgb, depth = predict(image_bytes, profile, modified_at)
except (UnidentifiedImageError, OSError, RuntimeError, ValueError) as exc:
    st.error(f"Could not process this photo. {exc}")
    st.stop()

preview = rgb.copy()
preview.thumbnail((640, 640))
heatmap = color_depth(depth, maximum, preview.size)

measurement = st.radio("Measure", ("Point", "Object"), horizontal=True)
measure_object = measurement == "Object"
photo_column, depth_column = st.columns(2)

with photo_column:
    st.subheader("Select an object" if measure_object else "Select a point")
    if measure_object:
        st.caption("Drag a tight box around one object.")
    click = streamlit_image_coordinates(
        preview,
        width="content",
        click_and_drag=measure_object,
        cursor="crosshair",
        key=f"{profile}-{mode}-{measurement}-{sha256(image_bytes).hexdigest()}",
    )

with depth_column:
    st.subheader("Estimated depth")
    caption = f"Purple 0 m · Yellow {maximum:.1f} m"
    if click is not None:
        display_width = max(click.get("width", preview.width), 1)
        display_height = max(click.get("height", preview.height), 1)
        draw = ImageDraw.Draw(heatmap)

        if measure_object:
            left, right = sorted((click["x1"] / display_width, click["x2"] / display_width))
            top, bottom = sorted((click["y1"] / display_height, click["y2"] / display_height))
            x0 = max(0, min(WIDTH, math.floor(left * WIDTH)))
            x1 = max(0, min(WIDTH, math.ceil(right * WIDTH)))
            y0 = max(0, min(HEIGHT, math.floor(top * HEIGHT)))
            y1 = max(0, min(HEIGHT, math.ceil(bottom * HEIGHT)))

            if left == right or top == bottom or x0 >= x1 or y0 >= y1:
                st.info("Drag across the photo to select an area.")
            else:
                estimate = float(np.median(depth[y0:y1, x0:x1]))
                draw.rectangle(
                    (
                        x0 * preview.width // WIDTH,
                        y0 * preview.height // HEIGHT,
                        min(preview.width - 1, math.ceil(x1 * preview.width / WIDTH) - 1),
                        min(preview.height - 1, math.ceil(y1 * preview.height / HEIGHT) - 1),
                    ),
                    outline="white",
                    width=2,
                )
                st.metric("Estimated object depth", f"{estimate:.1f} m")
                st.caption("Median depth inside the box. Background in the box can affect the estimate.")
        else:
            fraction_x = click["x"] / display_width
            fraction_y = click["y"] / display_height
            x = min(max(int(fraction_x * WIDTH), 0), WIDTH - 1)
            y = min(max(int(fraction_y * HEIGHT), 0), HEIGHT - 1)
            marker_x = min(max(int(fraction_x * preview.width), 0), preview.width - 1)
            marker_y = min(max(int(fraction_y * preview.height), 0), preview.height - 1)
            bounds = (marker_x - 5, marker_y - 5, marker_x + 5, marker_y + 5)
            draw.ellipse(bounds, fill="white", outline="black", width=2)
            st.metric("Selected point", f"{float(depth[y, x]):.1f} m")

    st.image(heatmap, caption=caption, width="content")
