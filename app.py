import pickle

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from scipy import ndimage as ndi
from tensorflow.keras import Model, layers

MODEL_PATH = "lung_unet.pkl"      # keep this file in the same folder as app.py


# ---------------- model (same as in the notebook) ----------------
def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    return x


def build_unet(input_shape=(256, 256, 1)):
    inputs = layers.Input(shape=input_shape)

    c1 = conv_block(inputs, 32);  p1 = layers.MaxPooling2D()(c1)
    c2 = conv_block(p1, 64);      p2 = layers.MaxPooling2D()(c2)
    c3 = conv_block(p2, 128);     p3 = layers.MaxPooling2D()(c3)
    c4 = conv_block(p3, 256);     p4 = layers.MaxPooling2D()(c4)

    c5 = conv_block(p4, 512)

    u6 = layers.Conv2DTranspose(256, 2, strides=2, padding="same")(c5)
    c6 = conv_block(layers.Concatenate()([u6, c4]), 256)
    u7 = layers.Conv2DTranspose(128, 2, strides=2, padding="same")(c6)
    c7 = conv_block(layers.Concatenate()([u7, c3]), 128)
    u8 = layers.Conv2DTranspose(64, 2, strides=2, padding="same")(c7)
    c8 = conv_block(layers.Concatenate()([u8, c2]), 64)
    u9 = layers.Conv2DTranspose(32, 2, strides=2, padding="same")(c8)
    c9 = conv_block(layers.Concatenate()([u9, c1]), 32)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(c9)
    return Model(inputs, outputs, name="Lung_UNet")


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:          # only load pickle files you trust
        data = pickle.load(f)
    model = build_unet((data["img_size"], data["img_size"], 1))
    model.set_weights(data["weights"])
    return model, data


# ---------------- preprocessing and prediction ----------------
def prep(gray_u8, size, clahe):
    x = cv2.resize(gray_u8, (size, size), interpolation=cv2.INTER_AREA)
    x = clahe.apply(x).astype(np.float32)
    return (x - x.mean()) / (x.std() + 1e-6)


def predict_prob(model, x, tta=True):             # x: (1, size, size, 1)
    p1 = model.predict(x, verbose=0)
    if not tta:
        return p1[0, ..., 0]
    p2 = model.predict(x[:, :, ::-1], verbose=0)[:, :, ::-1]   # flip and average
    return ((p1 + p2)[..., 0] / 2)[0]


@st.cache_data(show_spinner="Running model...")
def get_prob(_model, file_bytes, size, clip, tile, tta):
    """Model output is cached, so moving the sliders does not re-run the model."""
    gray = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None, None
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    x = prep(gray, size, clahe)[None, ..., None]
    return gray, predict_prob(_model, x, tta)


# ---------------- postprocessing ----------------
def split_merged(m, min_area, max_erosions=15):
    """If both lungs are joined in one blob, erode until two pieces appear,
    then give every pixel of the blob to its nearest piece."""
    for i in range(1, max_erosions + 1):
        er = ndi.binary_erosion(m, iterations=i)
        lab, n = ndi.label(er)
        if n < 2:
            if n == 0:
                return None
            continue
        sizes = ndi.sum(er, lab, range(1, n + 1))
        top = np.argsort(sizes)[-2:] + 1
        if sizes[top].min() < min_area * 0.2:
            continue
        seeds = np.where(np.isin(lab, top), lab, 0)
        idx = ndi.distance_transform_edt(seeds == 0, return_indices=True)[1]
        nearest = seeds[idx[0], idx[1]]
        out = np.where(m, nearest, 0)
        return np.where(out == top[0], 1, np.where(out == top[1], 2, 0))
    return None


def order_lungs(labels):
    """Relabel so 1 = patient's right lung (left side of image) and
    2 = patient's left lung (right side of image), using the x position."""
    ids = [i for i in np.unique(labels) if i != 0]
    if not ids:
        return labels
    xs = {i: ndi.center_of_mass(labels == i)[1] for i in ids}
    out = np.zeros_like(labels)
    if len(ids) == 1:
        side = 1 if xs[ids[0]] < labels.shape[1] / 2 else 2
        out[labels == ids[0]] = side
        return out
    ids = sorted(ids, key=lambda i: xs[i])[:2]
    out[labels == ids[0]] = 1
    out[labels == ids[1]] = 2
    return out


def postprocess(prob, thr=0.5, min_area_frac=0.01, split=True):
    """Returns a label map: 0 = background, 1 = right lung, 2 = left lung."""
    m = prob >= thr
    m = ndi.binary_opening(m, iterations=2)       # remove small specks
    m = ndi.binary_closing(m, iterations=3)       # join broken pieces
    m = ndi.binary_fill_holes(m)                  # fill holes inside the lungs

    lab, n = ndi.label(m)
    if n == 0:
        return np.zeros(m.shape, np.uint8)

    min_area = min_area_frac * m.size
    sizes = ndi.sum(m, lab, range(1, n + 1))
    order = np.argsort(sizes)[::-1]
    keep = [i + 1 for i in order if sizes[i] >= min_area][:2]   # up to 2 real lungs
    if not keep:
        return np.zeros(m.shape, np.uint8)

    labels = np.zeros(m.shape, np.uint8)
    for new_id, old_id in enumerate(keep, start=1):
        labels[lab == old_id] = new_id

    if len(keep) == 1 and split:                  # both lungs may be merged
        merged = split_merged(labels > 0, min_area)
        if merged is not None:
            labels = merged.astype(np.uint8)

    return order_lungs(labels).astype(np.uint8)


# ---------------- output helpers ----------------
def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def make_overlay(gray, labels, colors, opacity):
    h, w = gray.shape
    full = cv2.resize(labels, (w, h), interpolation=cv2.INTER_NEAREST)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB).astype(np.float32)
    out = rgb.copy()
    for lid, color in colors.items():
        region = full == lid
        out[region] = (1 - opacity) * rgb[region] + opacity * np.array(color, np.float32)
    return out.astype(np.uint8), full


def to_png_bytes(img):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()


# ---------------- app ----------------
def main():
    st.set_page_config(page_title="Lung Segmentation", layout="wide")
    st.title("")
    st.caption("For learning and research only. Not for medical diagnosis.")

    model, data = load_model()

    # ----- sidebar controls -----
    with st.sidebar:
        st.header("Settings")
        thr = st.slider("Threshold", 0.05, 0.95, float(data["threshold"]), 0.01,
                        help="Higher = stricter mask, lower = larger mask.")
        min_area = st.slider("Ignore pieces smaller than (% of image)", 0.0, 5.0, 1.0, 0.25) / 100
        split = st.checkbox("Try to split merged lungs", value=True)
        tta = st.checkbox("Flip averaging (slower, a bit more stable)", value=True)

        st.header("Overlay")
        right_color = st.color_picker("Right lung color", "#ff3b30")
        left_color = st.color_picker("Left lung color", "#0a84ff")
        opacity = st.slider("Opacity", 0.0, 1.0, 0.4, 0.05)
        st.caption("Right lung = patient's right, which appears on the left side of the image.")

    colors = {1: hex_to_rgb(right_color), 2: hex_to_rgb(left_color)}

    files = st.file_uploader("Upload one or more chest X-ray images",
                             type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    if not files:
        st.info("Upload an X-ray to start.")
        return

    size = data["img_size"]
    for f in files:
        gray, prob = get_prob(model, f.getvalue(), size, data["clahe_clip"],
                              tuple(data["clahe_tile"]), tta)
        if gray is None:
            st.error(f"Could not read {f.name}")
            continue

        labels = postprocess(prob, thr, min_area, split)
        overlay, full = make_overlay(gray, labels, colors, opacity)
        binary = ((full > 0) * 255).astype(np.uint8)

        st.subheader(f.name)
        c1, c2, c3 = st.columns(3)
        c1.image(gray, caption="X-ray", clamp=True)
        c2.image(binary, caption="Predicted mask")
        c3.image(overlay, caption="Overlay")

        # ----- stats -----
        total = (labels > 0).mean() * 100
        right = (labels == 1).mean() * 100
        left = (labels == 2).mean() * 100
        m1, m2, m3 = st.columns(3)
        m1.metric("Total lung area", f"{total:.1f}%")
        m2.metric("Right lung", f"{right:.1f}%")
        m3.metric("Left lung", f"{left:.1f}%")

        present = set(np.unique(labels)) - {0}
        if present != {1, 2}:
            st.warning("The result has no clear left and right lung. "
                       "Try changing the threshold or check this image.")
        elif min(right, left) / max(right, left) < 0.5:
            st.warning("One lung is much smaller than the other. This can be real, "
                       "but please check the mask.")

        # ----- downloads -----
        stem = f.name.rsplit(".", 1)[0]
        d1, d2, _ = st.columns([1, 1, 3])
        d1.download_button("Download mask", to_png_bytes(binary),
                           file_name=f"{stem}_mask.png", mime="image/png",
                           key=f"mask_{f.name}")
        d2.download_button("Download overlay", to_png_bytes(overlay),
                           file_name=f"{stem}_overlay.png", mime="image/png",
                           key=f"ov_{f.name}")
        st.divider()


if __name__ == "__main__":
    main()