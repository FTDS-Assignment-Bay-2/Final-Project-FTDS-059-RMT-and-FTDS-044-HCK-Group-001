from __future__ import annotations

import hashlib
import html
import io
import shutil
import tempfile
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, ImageColor
from skimage.color import rgb2lab
from pathlib import Path
import base64
from pathlib import Path

# ============================================================
# PAGE CONFIGURATION
# ============================================================

logo_path = Path(__file__).parent / "logo.png"

if logo_path.exists():
    logo_base64 = base64.b64encode(logo_path.read_bytes()).decode("utf-8")

    logo_html = (
        f'<img src="data:image/png;base64,{logo_base64}"'
        'alt="LipMatch AI Logo" class="lm-logo">'
    )
else:
    logo_html = ""
    st.warning("File logo.png tidak ditemukan.")

st.set_page_config(
    page_title="LipMatch AI | Lipstick Shade Finder",
    page_icon="💄",
    layout="wide",
    initial_sidebar_state="collapsed",
)


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset.csv"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

# MediaPipe Face Landmarker indices used by the accompanying notebook.
SKIN_SAMPLE_INDICES = {
    "Pipi kiri": 205,
    "Pipi kanan": 425,
    "Dahi": 151,
}
FACE_LEFT_INDEX = 234
FACE_RIGHT_INDEX = 454

# Titik MediaPipe untuk mendeteksi kontur bibir
LIP_OUTER_INDICES = [
    61,
    185,
    40,
    39,
    37,
    0,
    267,
    269,
    270,
    409,
    291,
    375,
    321,
    405,
    314,
    17,
    84,
    181,
    91,
    146,
]

LIP_INNER_INDICES = [
    78,
    191,
    80,
    81,
    82,
    13,
    312,
    311,
    310,
    415,
    308,
    324,
    318,
    402,
    317,
    14,
    87,
    178,
    88,
    95,
]


# ============================================================
# UI STYLE
# ============================================================
st.markdown(
    """
    <style>
    :root {
      --lm-ink: #38272e;
      --lm-muted: #806d73;
      --lm-rose: #8f405b;
      --lm-rose-dark: #713047;
      --lm-pink: #f7e7e9;
      --lm-cream: #fff8f1;
      --lm-line: #f0dfe0;
      --lm-white: #ffffff;
    }
    [data-testid="stAppViewContainer"] {
      background: linear-gradient(180deg, #fff9f5 0%, #ffffff 45%, #fff7f7 100%);
      color: var(--lm-ink);
    }
    [data-testid="stHeader"] { background: rgba(255,255,255,0); }
    [data-testid="stMainBlockContainer"] {
      max-width: 1180px;
      padding-top: 1.25rem;
      padding-bottom: 3rem;
    }
    h1, h2, h3, h4, p, label { color: var(--lm-ink); }
    .lm-brand-row { display:flex; align-items:center; gap:14px; margin: 0 0 1rem 0; }
    
    .lm-logo {
        width: 300px;
        height: 300px;
        object-fit: contain;
        flex-shrink: 0;
    }

    .lm-brand-name { font-size:2.25rem; line-height:1.1; font-weight:850; letter-spacing:-1.2px; color:#38272e; }
    .lm-brand-name span { color:#a64e6c; }
    .lm-brand-kicker { color:#92767f; font-size:.78rem; letter-spacing:2px; font-weight:750; margin-top:5px; }
    .lm-hero {
      border:1px solid #f0dfe0; border-radius:24px; padding:1.5rem 1.65rem;
      background:linear-gradient(115deg,rgba(255,247,241,.98),rgba(255,238,242,.94));
      margin-bottom:1.2rem;
    }
    .lm-hero h2 { font-size:1.65rem; line-height:1.25; margin:0 0 .5rem 0; color:#432a34; }
    .lm-hero p { color:#79636b; margin:0; font-size:1rem; line-height:1.7; }
    .lm-eyebrow { color:#9a4b66; font-weight:800; font-size:.75rem; letter-spacing:1.6px; margin-bottom:.5rem; }
    .lm-section-title { font-size:1.3rem; font-weight:800; margin:.3rem 0 .2rem 0; color:#402b33; }
    .lm-section-sub { color:#8a747b; font-size:.93rem; margin-bottom:.9rem; }
    .lm-step {
      background:rgba(255,255,255,.9); border:1px solid #f0dfe0; border-radius:17px;
      padding:1rem; min-height:112px;
    }
    .lm-step-num { color:#a24b68; font-size:.75rem; font-weight:850; letter-spacing:1.2px; margin-bottom:.35rem; }
    .lm-step-title { color:#422b34; font-size:1rem; font-weight:800; margin-bottom:.3rem; }
    .lm-step-copy { color:#806c72; font-size:.88rem; line-height:1.5; }
    .lm-panel {
      background:#fff; border:1px solid #f0dfe0; border-radius:20px; padding:1.1rem 1.2rem;
      box-shadow:0 5px 20px rgba(92,43,61,.035);
    }
    .lm-note {
      background:#fff8f0; border:1px solid #f1e2cf; border-radius:14px;
      padding:.85rem 1rem; color:#785f4a; font-size:.88rem; line-height:1.55;
    }
    .lm-result-hero {
      border-radius:20px; padding:1.15rem 1.25rem; border:1px solid #f0dfe0;
      background:linear-gradient(110deg,#fff8f2,#fff0f4); margin:.3rem 0 1rem 0;
    }
    .lm-result-label { color:#916b77; text-transform:uppercase; letter-spacing:1.4px; font-size:.73rem; font-weight:850; }
    .lm-result-value { color:#78384f; font-size:1.65rem; font-weight:850; margin:.15rem 0 .2rem 0; }
    .lm-result-copy { color:#7b666e; font-size:.9rem; line-height:1.5; }
    .lm-color-chip { width:100%; height:16px; border-radius:99px; border:1px solid rgba(0,0,0,.06); margin:.4rem 0; }
    .lm-shade-card {
      border:1px solid #f0dfe0; background:#fff; border-radius:17px; overflow:hidden;
      height:100%; box-shadow:0 5px 18px rgba(92,43,61,.035);
    }
    .lm-swatch {
      height:78px; padding:12px; display:flex; align-items:flex-end; justify-content:flex-start;
      position:relative;
    }
    .lm-swatch span {
      font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.74rem;
      color:white; background:rgba(35,22,27,.45); padding:4px 7px; border-radius:7px;
    }
    .lm-shade-content { padding:1rem; }
    .lm-shade-name { font-size:1rem; line-height:1.35; font-weight:850; color:#3d2931; margin-bottom:.35rem; }
    .lm-product-name { font-size:.82rem; color:#806b72; line-height:1.45; margin-bottom:.55rem; min-height:2.35rem; }
    .lm-price { font-size:.9rem; font-weight:800; color:#8d405c; margin-bottom:.55rem; }
    .lm-description { color:#806d73; font-size:.82rem; line-height:1.5; min-height:3.8rem; }
    .lm-score { border-top:1px solid #f3e7e8; margin-top:.75rem; padding-top:.65rem; display:flex; justify-content:space-between; gap:.5rem; font-size:.77rem; color:#927d83; }
    .lm-score strong { color:#6f344a; font-size:.83rem; }
    .lm-product-link { display:inline-block; margin-top:.75rem; color:#8f405b !important; font-size:.84rem; font-weight:800; text-decoration:none !important; }
    .lm-product-link:hover { color:#713047 !important; text-decoration:underline !important; }
    .lm-category-pill { display:inline-block; font-size:.72rem; letter-spacing:.5px; font-weight:850; text-transform:uppercase; padding:5px 9px; border-radius:99px; margin-bottom:.6rem; }
    .lm-pill-warm { background:#fff0e5; color:#a45b2b; }
    .lm-pill-cool { background:#f1eafb; color:#75519b; }
    .lm-pill-neutral { background:#f3eee8; color:#786452; }
    .lm-footer { color:#9b878d; font-size:.8rem; text-align:center; padding:1.5rem .5rem .2rem .5rem; }
    div.stButton > button[kind="primary"] {
      background:linear-gradient(90deg,#91415d,#b65f7c); border:0; border-radius:12px;
      min-height:3rem; font-weight:800; color:white; width:100%;
    }
    div.stButton > button[kind="primary"]:hover { background:linear-gradient(90deg,#78344c,#a84f6e); color:white; border:0; }
    div[data-testid="stFileUploader"] section { background:#fffaf8; border:1px dashed #ddb9c3; border-radius:14px; }
    div[data-testid="stCameraInput"] { border-radius:14px; }
    div[data-testid="stAlert"] { border-radius:12px; }
    @media (max-width: 700px) {
      .lm-brand-name { font-size:1.8rem; }
      .lm-hero { padding:1.1rem; }
      .lm-hero h2 { font-size:1.35rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATASET AND COLOR UTILITIES
# ============================================================
def hex_to_rgb(hex_value: str) -> tuple[int, int, int]:
    """Convert a valid #RRGGBB value to an RGB tuple."""
    return tuple(ImageColor.getcolor(str(hex_value).strip(), "RGB"))


def rgb_to_lab(rgb: tuple[float, float, float] | np.ndarray) -> np.ndarray:
    rgb_array = np.asarray(rgb, dtype=np.float32)
    return rgb2lab((rgb_array / 255.0).reshape(1, 1, 3))[0, 0]


def estimate_skin_category(lab: np.ndarray) -> str:
    """Prototype heuristic inherited from the notebook, not a validated diagnosis."""
    a_star = float(lab[1])
    b_star = float(lab[2])
    difference = b_star - a_star
    if difference >= 11:
        return "Warm"
    if difference <= 4:
        return "Cool"
    return "Neutral"


def classify_shade_category(row: pd.Series) -> str:
    """Heuristically group catalog shades from their names and descriptions."""
    text = " ".join(
        [
            str(row.get("Varian Warna", "")),
            str(row.get("Nama Produk", "")),
            str(row.get("Deskripsi", "")),
        ]
    ).casefold()

    terms = {
        "Warm": [
            "hangat",
            "warm",
            "oranye",
            "orange",
            "peach",
            "persik",
            "terracotta",
            "terakota",
            "coral",
            "koral",
            "cokelat",
            "coklat",
            "brown",
            "caramel",
            "karamel",
            "brick",
            "bata",
            "cinnamon",
            "ginger",
            "burnt",
            "pumpkin",
            "mango",
            "apricot",
            "golden",
            "sunset",
            "toast",
            "latte",
            "mocha",
            "mocca",
            "coffee",
            "toffee",
            "amber",
            "copper",
            "bronze",
            "autumn",
            "rust",
            "tan",
            "milk brown",
            "chocolate",
            "nude caramel",
            "nude peach",
        ],
        "Cool": [
            "dingin",
            "cool",
            "mauve",
            "berry",
            "beri",
            "plum",
            "ungu",
            "purple",
            "fuchsia",
            "magenta",
            "lavender",
            "wine",
            "burgundy",
            "marun",
            "raspberry",
            "cranberry",
            "orchid",
            "violet",
            "keunguan",
            "mulberry",
            "grape",
            "sangria",
            "cerise",
            "lilac",
            "blue-based",
            "blue based",
            "winter berry",
            "pink ungu",
        ],
        "Neutral": [
            "neutral",
            "netral",
            "beige",
            "rose brown",
            "rosewood",
            "nude pink",
            "pink beige",
            "classic red",
            "soft pink",
            "dusty rose",
            "rose",
            "nude",
            "taupe",
            "muted rose",
            "pink kecokelatan",
            "pink kecoklatan",
            "natural pink",
            "true red",
            "true nude",
            "warna bibir alami",
            "natural red",
        ],
    }
    scores = {
        category: sum(text.count(term) for term in words)
        for category, words in terms.items()
    }

    # Explicit undertone phrases carry more weight than general shade names.
    if any(
        phrase in text
        for phrase in ["undertone dingin", "undertone cool", "nuansa dingin"]
    ):
        scores["Cool"] += 5
    if any(
        phrase in text
        for phrase in ["undertone hangat", "undertone warm", "nuansa hangat"]
    ):
        scores["Warm"] += 5
    if any(
        phrase in text
        for phrase in ["undertone netral", "undertone neutral", "nuansa netral"]
    ):
        scores["Neutral"] += 5

    top_score = max(scores.values())
    tied = [name for name, score in scores.items() if score == top_score]
    if top_score > 0 and len(tied) == 1:
        return tied[0]

    # Deterministic fallback for names/descriptions that do not give a clear cue.
    rgb = np.asarray(hex_to_rgb(row["Hex Code"]), dtype=np.float32) / 255.0
    hue, saturation, _ = __import__("colorsys").rgb_to_hsv(*rgb)
    hue *= 360
    if 15 <= hue < 70:
        return "Warm"
    if 260 <= hue < 340:
        return "Cool"
    return "Neutral"


def cosine_similarity_rgb(
    color_a: tuple[int, int, int] | np.ndarray,
    color_b: tuple[int, int, int] | np.ndarray,
) -> float:
    """Cosine similarity between two RGB color vectors."""
    a = np.asarray(color_a, dtype=np.float64)
    b = np.asarray(color_b, dtype=np.float64)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0:
        return 0.0
    return float(np.dot(a, b) / denominator)


@st.cache_data(show_spinner=False)
def load_catalog() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"File dataset.csv tidak ditemukan di {DATASET_PATH}. "
            "Letakkan dataset.csv di folder yang sama dengan app.py."
        )

    catalog = pd.read_csv(DATASET_PATH)
    required_columns = [
        "Nama Produk",
        "Varian Warna",
        "Hex Code",
        "Harga",
        "Deskripsi",
        "Link Produk",
    ]
    missing = [column for column in required_columns if column not in catalog.columns]
    if missing:
        raise ValueError("Kolom dataset tidak lengkap: " + ", ".join(missing))

    catalog = catalog.dropna(subset=required_columns).copy()
    catalog["Hex Code"] = catalog["Hex Code"].astype(str).str.strip()
    catalog["_RGB"] = catalog["Hex Code"].apply(hex_to_rgb)
    catalog["_Kategori"] = catalog.apply(classify_shade_category, axis=1)
    return catalog


# ============================================================
# FACE MODEL AND SKIN COLOR EXTRACTION
# ============================================================
@st.cache_resource(show_spinner=False)
def ensure_face_model() -> str:
    """Use a local MediaPipe model if available; otherwise download it once."""
    local_model = BASE_DIR / "models" / "face_landmarker.task"
    if local_model.exists() and local_model.stat().st_size > 100_000:
        return str(local_model)

    cached_model = Path(tempfile.gettempdir()) / "lipmatch_face_landmarker.task"
    if cached_model.exists() and cached_model.stat().st_size > 100_000:
        return str(cached_model)

    temp_download = cached_model.with_suffix(".download")
    try:
        request = urllib.request.Request(
            MODEL_URL, headers={"User-Agent": "LipMatchAI/1.0"}
        )
        with urllib.request.urlopen(request, timeout=120) as response, open(
            temp_download, "wb"
        ) as output:
            shutil.copyfileobj(response, output)
        if not temp_download.exists() or temp_download.stat().st_size < 100_000:
            raise RuntimeError("File model terlalu kecil atau unduhan tidak lengkap.")
        temp_download.replace(cached_model)
        return str(cached_model)
    except Exception as exc:
        if temp_download.exists():
            temp_download.unlink(missing_ok=True)
        raise RuntimeError(
            "Model deteksi wajah belum tersedia. Periksa koneksi internet saat pertama kali "
            "menjalankan aplikasi. Detail teknis: " + str(exc)
        ) from exc


def decode_uploaded_image(image_bytes: bytes, max_side: int = 1600) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes))
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    longest = max(width, height)
    if longest > max_side:
        scale = max_side / float(longest)
        image = image.resize(
            (max(1, int(width * scale)), max(1, int(height * scale))),
            Image.Resampling.LANCZOS,
        )
    return np.asarray(image, dtype=np.uint8)


def landmark_pixel(
    landmarks: list, index: int, image_shape: tuple[int, ...]
) -> np.ndarray:
    height, width = image_shape[:2]
    point = landmarks[index]
    x = int(np.clip(round(float(point.x) * width), 0, width - 1))
    y = int(np.clip(round(float(point.y) * height), 0, height - 1))
    return np.array([x, y], dtype=np.int32)


def analyze_skin_color(image_rgb: np.ndarray) -> dict:
    model_path = ensure_face_model()
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=np.ascontiguousarray(image_rgb),
    )
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
    )

    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        raise ValueError(
            "Wajah belum terdeteksi. Gunakan foto wajah yang menghadap ke depan, "
            "cukup terang, dan tidak tertutup rambut atau benda lain."
        )

    landmarks = result.face_landmarks[0]
    if len(landmarks) <= FACE_RIGHT_INDEX:
        raise ValueError(
            "Titik wajah tidak lengkap. Coba foto lain dengan wajah menghadap ke depan."
        )

    # Simpan koordinat landmark wajah dalam bentuk array numerik.
    # Koordinat ini digunakan kembali untuk virtual try-on.
    landmarks_xy = np.asarray(
        [[float(point.x), float(point.y)] for point in landmarks],
        dtype=np.float32,
    )

    height, width = image_rgb.shape[:2]
    centers = {
        name: landmark_pixel(landmarks, index, image_rgb.shape)
        for name, index in SKIN_SAMPLE_INDICES.items()
    }
    face_width = float(
        np.linalg.norm(
            landmark_pixel(landmarks, FACE_LEFT_INDEX, image_rgb.shape)
            - landmark_pixel(landmarks, FACE_RIGHT_INDEX, image_rgb.shape)
        )
    )
    radius = max(3, int(face_width * 0.035))
    yy, xx = np.ogrid[:height, :width]
    sampled_regions = []
    region_rgb = {}

    for name, center in centers.items():
        cx, cy = int(center[0]), int(center[1])
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
        pixels = image_rgb[mask]
        if pixels.size == 0:
            raise ValueError(f"Sampel warna {name.lower()} tidak berhasil diambil.")
        region_rgb[name] = np.median(pixels, axis=0)
        sampled_regions.append(pixels)

    all_pixels = np.concatenate(sampled_regions, axis=0)
    median_rgb = np.median(all_pixels, axis=0)
    median_rgb_int = np.clip(np.rint(median_rgb), 0, 255).astype(np.uint8)
    skin_hex = "#{:02X}{:02X}{:02X}".format(*median_rgb_int)
    lab = rgb_to_lab(median_rgb)
    category = estimate_skin_category(lab)

    preview = image_rgb.copy()
    for center in centers.values():
        cv2.circle(
            preview,
            tuple(map(int, center)),
            radius,
            (40, 220, 90),
            max(2, int(max(height, width) / 500)),
        )

    return {
        "skin_rgb": tuple(int(value) for value in median_rgb_int),
        "skin_hex": skin_hex,
        "lab": tuple(float(value) for value in lab),
        "category": category,
        "sample_preview": preview,
        "landmarks_xy": landmarks_xy,
        "region_rgb": {
            key: tuple(float(v) for v in value) for key, value in region_rgb.items()
        },
    }


def lip_landmark_pixel(
    landmarks_xy: np.ndarray,
    index: int,
    image_shape: tuple[int, ...],
) -> np.ndarray:
    """Mengubah koordinat landmark menjadi koordinat piksel."""
    height, width = image_shape[:2]

    x = int(np.clip(round(float(landmarks_xy[index, 0]) * width), 0, width - 1))
    y = int(np.clip(round(float(landmarks_xy[index, 1]) * height), 0, height - 1))

    return np.array([x, y], dtype=np.int32)


def lip_points_from_indices(
    indices,
    landmarks_xy: np.ndarray,
    image_shape: tuple[int, ...],
) -> np.ndarray:
    return np.asarray(
        [lip_landmark_pixel(landmarks_xy, index, image_shape) for index in indices],
        dtype=np.int32,
    )


def create_lip_mask(
    landmarks_xy: np.ndarray,
    image_shape: tuple[int, ...],
    feather_kernel: int = 7,
):
    """Membuat mask bibir dan mengecualikan bagian dalam mulut."""
    height, width = image_shape[:2]

    if len(landmarks_xy) <= max(LIP_OUTER_INDICES + LIP_INNER_INDICES):
        raise ValueError("Landmark bibir tidak lengkap. Coba foto lain.")

    outer_mask = np.zeros((height, width), dtype=np.uint8)
    inner_mask = np.zeros((height, width), dtype=np.uint8)

    outer_points = lip_points_from_indices(LIP_OUTER_INDICES, landmarks_xy, image_shape)
    inner_points = lip_points_from_indices(LIP_INNER_INDICES, landmarks_xy, image_shape)

    cv2.fillPoly(outer_mask, [outer_points], 255)
    cv2.fillPoly(inner_mask, [inner_points], 255)

    # Pisahkan area bibir dari bagian dalam mulut.
    hard_mask = cv2.subtract(outer_mask, inner_mask)

    # Menghaluskan tepi warna agar hasil lebih natural.
    kernel = max(1, int(feather_kernel))
    if kernel % 2 == 0:
        kernel += 1

    if kernel > 1:
        soft_alpha = (
            cv2.GaussianBlur(hard_mask, (kernel, kernel), 0).astype(np.float32) / 255.0
        )
    else:
        soft_alpha = hard_mask.astype(np.float32) / 255.0

    soft_alpha[hard_mask == 0] = 0.0

    return hard_mask, soft_alpha


def apply_lipstick_color(
    image_rgb: np.ndarray,
    landmarks_xy: np.ndarray,
    hex_color: str,
    strength: float = 0.62,
):
    """Menerapkan warna lipstik pada bibir tanpa mewarnai seluruh wajah."""
    if not 0.0 <= strength <= 1.0:
        raise ValueError("Intensitas warna harus antara 0 dan 1.")

    source = np.asarray(image_rgb, dtype=np.uint8)

    hard_mask, soft_alpha = create_lip_mask(landmarks_xy, source.shape)

    if np.count_nonzero(hard_mask) < 30:
        raise ValueError(
            "Area bibir tidak berhasil dideteksi. "
            "Coba foto yang lebih frontal dan jelas."
        )

    # Konversi warna shade dari HEX ke RGB.
    target_rgb = np.asarray(hex_to_rgb(hex_color), dtype=np.uint8).reshape(1, 1, 3)

    # Ruang LAB membantu mempertahankan pencahayaan bibir.
    source_lab = cv2.cvtColor(source, cv2.COLOR_RGB2LAB).astype(np.float32)

    target_lab = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)[0, 0]

    recolored_lab = source_lab.copy()

    # Campurkan warna shade dengan warna asli bibir.
    recolored_lab[:, :, 1] = (
        source_lab[:, :, 1] * (1.0 - strength) + target_lab[1] * strength
    )
    recolored_lab[:, :, 2] = (
        source_lab[:, :, 2] * (1.0 - strength) + target_lab[2] * strength
    )

    recolored_rgb = cv2.cvtColor(
        np.clip(recolored_lab, 0, 255).astype(np.uint8),
        cv2.COLOR_LAB2RGB,
    )

    # Terapkan warna hanya pada area bibir.
    alpha = soft_alpha[..., None]

    result = (
        source.astype(np.float32) * (1.0 - alpha)
        + recolored_rgb.astype(np.float32) * alpha
    )

    result_rgb = np.clip(result, 0, 255).astype(np.uint8)

    # Piksel di luar bibir tetap sama dengan foto asli.
    result_rgb[hard_mask == 0] = source[hard_mask == 0]

    return result_rgb, hard_mask


def recommend_shades(
    catalog: pd.DataFrame, skin_rgb: tuple[int, int, int]
) -> dict[str, pd.DataFrame]:
    ranked = catalog.copy()
    ranked["_Cosine"] = ranked["_RGB"].apply(
        lambda rgb: cosine_similarity_rgb(skin_rgb, rgb)
    )
    recommendations = {}
    for category in ["Warm", "Cool", "Neutral"]:
        subset = ranked[ranked["_Kategori"] == category].sort_values(
            ["_Cosine", "Nama Produk", "Varian Warna"],
            ascending=[False, True, True],
        )
        recommendations[category] = subset.head(3).copy()
    return recommendations


# ============================================================
# HTML COMPONENTS
# ============================================================
def safe(value: object) -> str:
    return html.escape(str(value))


def render_shade_card(row: pd.Series, category: str) -> str:
    hex_color = safe(row["Hex Code"])
    variant = safe(row["Varian Warna"])
    product = safe(row["Nama Produk"])
    price = safe(row["Harga"])
    description = safe(row["Deskripsi"])
    url = safe(row["Link Produk"])
    score = float(row["_Cosine"])
    category_class = {"Warm": "warm", "Cool": "cool", "Neutral": "neutral"}[category]
    return f"""
    <div class="lm-shade-card">
      <div class="lm-swatch" style="background:{hex_color};"><span>{hex_color}</span></div>
      <div class="lm-shade-content">
        <div class="lm-category-pill lm-pill-{category_class}">{safe(category)}</div>
        <div class="lm-shade-name">{variant}</div>
        <div class="lm-product-name">{product}</div>
        <div class="lm-price">{price}</div>
        <div class="lm-description">{description}</div>
        <div class="lm-score"><span>Cosine similarity (RGB)</span><strong>{score:.4f}</strong></div>
        <a class="lm-product-link" href="{url}" target="_blank" rel="noopener noreferrer">Lihat produk ↗</a>
      </div>
    </div>
    """


def render_recommendation_group(
    category: str,
    rows: pd.DataFrame,
    image_rgb: np.ndarray,
    landmarks_xy: np.ndarray,
    strength: float,
) -> None:

    subtitles = {
        "Warm": (
            "Pilihan hangat seperti peach, coral, " "terracotta, dan cokelat kemerahan."
        ),
        "Cool": ("Pilihan seperti mauve, berry, plum, " "dan pink keunguan."),
        "Neutral": (
            "Pilihan seimbang seperti rose, beige, " "nude, dan warna bibir natural."
        ),
    }

    st.markdown(
        f"<div class='lm-section-title'>{category} shades</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='lm-section-sub'>{subtitles[category]}</div>",
        unsafe_allow_html=True,
    )

    columns = st.columns(3, gap="medium")

    for index, column in enumerate(columns):
        with column:
            if index >= len(rows):
                st.info("Belum ada shade yang tersedia.")
                continue

            row = rows.iloc[index]

            try:
                # Simulasikan shade produk pada foto wajah pengguna.
                preview, _ = apply_lipstick_color(
                    image_rgb=image_rgb,
                    landmarks_xy=landmarks_xy,
                    hex_color=str(row["Hex Code"]),
                    strength=strength,
                )

                st.image(
                    preview,
                    caption=(f"Simulasi bibir · " f"{row['Varian Warna']}"),
                    use_container_width=True,
                )

            except Exception as exc:
                st.warning(f"Preview shade gagal dibuat: {exc}")

            # Pertahankan kartu produk dan informasi dataset.
            st.markdown(
                render_shade_card(row, category),
                unsafe_allow_html=True,
            )


# ============================================================
# MAIN PAGE
# ============================================================
st.markdown(
    f"""
    <div class="lm-brand-row">
      {logo_html}
      <div>
        <div class="lm-brand-name">LipMatch<span>AI</span></div>
        <div class="lm-brand-kicker">YOUR SKIN TONE, YOUR LIPSTICK MATCH</div>
      </div>
    </div>
    <div class="lm-hero">
      <div class="lm-eyebrow">PERSONALIZED LIPSTICK SHADE FINDER</div>
      <h2>Temukan warna lipstik yang selaras dengan warna kulitmu.</h2>
      <p>Unggah foto wajah, analisis warna kulit yang tertangkap kamera, lalu lihat rekomendasi shade dari katalog lipstik merek Indonesia. Semua proses tersedia dalam satu halaman.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='lm-section-title'>Cara menggunakan LipMatch AI</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='lm-section-sub'>Ikuti tiga langkah singkat untuk melihat rekomendasi warna.</div>",
    unsafe_allow_html=True,
)
step_columns = st.columns(3, gap="medium")
steps = [
    (
        "01 / UPLOAD",
        "Masukkan foto wajah",
        "Gunakan foto frontal dengan cahaya merata. Hindari filter dan pencahayaan berwarna.",
    ),
    (
        "02 / ANALYZE",
        "Analisis warna kulit",
        "Sistem mengambil sampel warna dari area pipi dan dahi untuk menghitung warna RGB dan CIELAB.",
    ),
    (
        "03 / DISCOVER",
        "Lihat 9 rekomendasi",
        "Tiga shade ditampilkan untuk setiap kelompok Warm, Cool, dan Neutral berdasarkan skor Cosine Similarity.",
    ),
]
for column, (number, title, copy) in zip(step_columns, steps):
    with column:
        st.markdown(
            f"<div class='lm-step'><div class='lm-step-num'>{number}</div><div class='lm-step-title'>{title}</div><div class='lm-step-copy'>{copy}</div></div>",
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    "<div class='lm-section-title'>Upload your photo here</div>", unsafe_allow_html=True
)
st.markdown(
    "<div class='lm-section-sub'>Pilih foto dari galeri atau ambil foto langsung melalui kamera perangkat.</div>",
    unsafe_allow_html=True,
)

source = st.radio(
    "Pilih sumber foto",
    ["Upload dari galeri", "Gunakan kamera"],
    horizontal=True,
    label_visibility="collapsed",
    key="photo_source",
)

uploaded_file = None
if source == "Upload dari galeri":
    uploaded_file = st.file_uploader(
        "Pilih foto wajah",
        type=["jpg", "jpeg", "png", "webp"],
        help="Gunakan foto wajah yang menghadap ke depan dan tidak memakai filter warna.",
        key="gallery_photo",
    )
else:
    uploaded_file = st.camera_input(
        "Ambil foto wajah",
        help="Posisikan wajah di tengah dan pastikan pipi serta dahi terlihat jelas.",
        key="camera_photo",
    )

image_bytes = uploaded_file.getvalue() if uploaded_file is not None else None
image_hash = hashlib.sha256(image_bytes).hexdigest() if image_bytes else None

if image_bytes:
    try:
        input_rgb = decode_uploaded_image(image_bytes)
        preview_col, tip_col = st.columns([1.05, 0.95], gap="large")
        with preview_col:
            st.markdown(
                "<div class='lm-section-title'>Preview foto</div>",
                unsafe_allow_html=True,
            )
            st.image(
                input_rgb, caption="Foto yang akan dianalisis", use_container_width=True
            )
        with tip_col:
            st.markdown(
                "<div class='lm-panel'><div class='lm-section-title'>Agar hasil lebih konsisten</div><ul style='color:#806d73;line-height:1.7;padding-left:1.1rem;'><li>Hadap kamera secara langsung.</li><li>Gunakan cahaya alami atau lampu putih yang merata.</li><li>Hindari filter, mode beauty, dan makeup tebal pada area sampel.</li><li>Pastikan dahi serta kedua pipi terlihat.</li></ul></div>",
                unsafe_allow_html=True,
            )
        if st.button("✨  Analyze & Predict", type="primary", use_container_width=True):
            with st.spinner(
                "Menganalisis warna kulit dan mencocokkan katalog lipstik..."
            ):
                try:
                    analysis = analyze_skin_color(input_rgb)
                    analysis["image_hash"] = image_hash
                    st.session_state["lipmatch_analysis"] = analysis
                except Exception as exc:
                    st.session_state.pop("lipmatch_analysis", None)
                    st.error(str(exc))
    except Exception as exc:
        st.error(f"Foto tidak dapat dibaca. Coba file JPG atau PNG lain. Detail: {exc}")
else:
    st.markdown(
        "<div class='lm-note'>💡 <b>Tips:</b> Untuk hasil awal yang lebih stabil, pilih foto tanpa filter, menghadap depan, dan menggunakan pencahayaan yang merata.</div>",
        unsafe_allow_html=True,
    )

# Keep output on this same page and only show it for the currently selected image.
analysis = st.session_state.get("lipmatch_analysis")
if analysis and image_hash and analysis.get("image_hash") == image_hash:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<div class='lm-section-title'>This is your lipstick shade prediction</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='lm-section-sub'>Hasil ekstraksi warna kulit dan sembilan pilihan shade dari dataset produk.</div>",
        unsafe_allow_html=True,
    )

    result_image_col, result_info_col = st.columns([1.05, 1], gap="large")
    with result_image_col:
        st.image(
            analysis["sample_preview"],
            caption="Titik hijau menunjukkan area sampel warna yang digunakan sistem.",
            use_container_width=True,
        )
    with result_info_col:
        category = analysis["category"]
        skin_hex = analysis["skin_hex"]
        lab = analysis["lab"]
        st.markdown(
            f"""
            <div class="lm-result-hero">
              <div class="lm-result-label">Kategori warna kulit terdeteksi</div>
              <div class="lm-result-value">{safe(category)}</div>
              <div class="lm-result-copy">Kategori ini merupakan estimasi awal dari warna yang tertangkap kamera. Hasil dapat berubah akibat pencahayaan, kamera, filter, atau makeup.</div>
              <div class="lm-color-chip" style="background:{safe(skin_hex)}"></div>
              <div style="display:flex;justify-content:space-between;gap:1rem;align-items:center;">
                <span style="color:#806d73;font-size:.85rem;">Warna kulit terukur</span>
                <strong style="font-family:ui-monospace,monospace;color:#633247;">{safe(skin_hex)}</strong>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='lm-panel'><b>Langkah berikutnya</b><p style='color:#806d73;line-height:1.6;margin:.45rem 0 0 0;'>LipMatch AI membandingkan vektor RGB warna kulit dengan warna HEX dari setiap shade di dataset. Hasilnya diurutkan dengan Cosine Similarity, lalu ditampilkan dalam tiga kelompok warna.</p></div>",
            unsafe_allow_html=True,
        )

    with st.expander("Lihat detail teknis analisis warna"):
        st.write(f"**RGB median:** {analysis['skin_rgb']}")
        st.write(f"**CIELAB (L*, a*, b*):** ({lab[0]:.2f}, {lab[1]:.2f}, {lab[2]:.2f})")
        st.write(
            "**Metode kategori:** aturan heuristik awal dari notebook, bukan model undertone yang tervalidasi secara klinis atau profesional."
        )

    try:
        catalog = load_catalog()
        recs = recommend_shades(catalog, analysis["skin_rgb"])
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<div class='lm-note'>Skor Cosine Similarity dipakai untuk mengurutkan warna dalam tiap kelompok. Skor ini bukan probabilitas bahwa shade pasti cocok, dan kategori Warm/Cool/Neutral pada katalog merupakan pengelompokan heuristik dari nama serta deskripsi shade.</div>",
            unsafe_allow_html=True,
        )

        try_on_strength = st.slider(
            "Intensitas warna lipstik pada foto",
            min_value=0.30,
            max_value=0.90,
            value=0.62,
            step=0.05,
            help="Atur seberapa kuat warna lipstik diterapkan pada bibir.",
            key="lipmatch_try_on_strength",
        )

        for category_name in ["Warm", "Cool", "Neutral"]:
            st.markdown("<br>", unsafe_allow_html=True)

            render_recommendation_group(
                category_name,
                recs[category_name],
                input_rgb,
                analysis["landmarks_xy"],
                try_on_strength,
            )

    except Exception as exc:
        st.error(f"Dataset rekomendasi tidak dapat diproses. Detail: {exc}")

st.markdown(
    "<div class='lm-footer'>LipMatch AI · Prototype rekomendasi shade lipstik · Foto diproses di sesi aplikasi dan tidak disimpan sebagai file oleh kode ini.</div>",
    unsafe_allow_html=True,
)
