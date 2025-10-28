import os, json, numpy as np, torch
from pathlib import Path
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
import gradio as gr
import plotly.graph_objects as go
import gradio_client.utils as gu
from audiotools import AudioSignal

# ---- your project imports ----
from text2fx.sap_main import get_model
from text2fx.sap_apply import main as t2fx
from text2fx.core import (
    preprocess_audio,
    detensor_dict,
    flatten_single_item_lists,
    apply_audealize_single_word,
)
from text2fx.applyFXparams import apply_fx_to_sig

# -------------------------------------------------------------------------
# Gradio schema patch (bug workaround)
# -------------------------------------------------------------------------
_old_json_schema_to_python_type = gu._json_schema_to_python_type
_old_get_type = gu.get_type
def safe_json_schema_to_python_type(schema, defs=None):
    if isinstance(schema, bool): return "object" if schema else "never"
    try: return _old_json_schema_to_python_type(schema, defs)
    except Exception: return "object"
def safe_get_type(schema):
    if isinstance(schema, bool): return "object" if schema else "never"
    try: return _old_get_type(schema)
    except Exception: return "object"
gu._json_schema_to_python_type = safe_json_schema_to_python_type
gu.get_type = safe_get_type

# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLAP = get_model("ms_clap")
DRY_AUDIO = "/home/annie/research/text2fx/assets/guitar.wav"

EQ_WORDS = ["warm","bright","harsh","soft","cold"]

CACHE_DIR = Path("/home/annie/research/text2fx/anchors")
CACHE_DIR.mkdir(exist_ok=True, parents=True)
EMB_PATH = CACHE_DIR / "audealize_anchor_embs.npy"
FX_CACHE_PATH = CACHE_DIR / "audealize_anchor_fxparams.json"
TMP_AUDIO_DIR = Path("text2fx/tmp")
TMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# 1) Build / load Audealize-based anchor CLAP embeddings
# -------------------------------------------------------------------------
def build_or_load_anchor_embs(words):
    if EMB_PATH.exists():
        print(f"[INFO] Loading precomputed anchor embeddings from {EMB_PATH}")
        anchor_embs = np.load(EMB_PATH)
        if anchor_embs.shape[0] != len(words):
            print("[WARN] Cached embeddings count mismatch. Recomputing...")
            anchor_embs = compute_anchor_embs(words)
    else:
        anchor_embs = compute_anchor_embs(words)
    return anchor_embs

def compute_anchor_embs(words):
    print("----[INFO] Generating Audealize anchors and embeddings ...")
    embs = []
    for w in words:
        print(f"  → {w}")
        wet_sig = apply_audealize_single_word(DRY_AUDIO, w, CACHE_DIR)
        wet_sig = preprocess_audio(wet_sig).to(DEVICE)
        with torch.no_grad():
            emb = torch.nn.functional.normalize(CLAP.get_audio_embeddings(wet_sig), dim=-1)
        embs.append(emb.cpu().numpy().squeeze())
    anchor_embs = np.stack(embs)
    np.save(EMB_PATH, anchor_embs)
    print(f"---[INFO] Saved precomputed embeddings to {EMB_PATH}")
    return anchor_embs

anchor_embs = build_or_load_anchor_embs(EQ_WORDS)

# -------------------------------------------------------------------------
# 2) Fit PCA → build 2D map
# -------------------------------------------------------------------------
pca_model = PCA(n_components=2)
coords_2d_raw = pca_model.fit_transform(anchor_embs)
coords_min, coords_max = coords_2d_raw.min(0), coords_2d_raw.max(0)
coords_span = np.maximum(coords_max - coords_min, 1e-8)
coords_2d = (coords_2d_raw - coords_min) / coords_span
anchor_positions = {w: coords_2d[i] for i, w in enumerate(EQ_WORDS)}

# -------------------------------------------------------------------------
# 3) Build / load FX param cache
# -------------------------------------------------------------------------
def build_or_load_fx_cache(words):
    if FX_CACHE_PATH.exists():
        with open(FX_CACHE_PATH, "r") as f: fx_cache = json.load(f)
        if set(fx_cache.keys()) == set(words):
            print(f"[INFO] Loaded cached FX params from {FX_CACHE_PATH}")
            return fx_cache
        else:
            print("[WARN] Anchor mismatch, recomputing FX cache...")
    return compute_fx_cache(words)

def compute_fx_cache(words):
    print("[INFO] Computing anchor FX params with text2fx (EQ)...")
    fx_cache = {}
    for w in words:
        print(f"  → {w}")
        out_sig, _, out_params_dict = t2fx(
            DRY_AUDIO, ["eq"], [w],
            n_iters=150, params_init_type="curriculum",
            criterion="cosine-sim", pls_normalize=True,
        )
        fx_cache[w] = flatten_single_item_lists(detensor_dict(out_params_dict))
    with open(FX_CACHE_PATH, "w") as f: json.dump(fx_cache, f, indent=2)
    print(f"[INFO] Saved cached FX params to {FX_CACHE_PATH}")
    return fx_cache

fx_params_cache = build_or_load_fx_cache(EQ_WORDS)

# -------------------------------------------------------------------------
# 4) Helpers
# -------------------------------------------------------------------------
def interpolate_params(pos_xy, anchor_positions, fx_cache, sharpness=10.0):
    words = list(anchor_positions.keys())
    coords = np.array([anchor_positions[w] for w in words])
    dists = cdist([pos_xy], coords)[0] + 1e-6
    weights = np.exp(-sharpness * dists); weights /= weights.sum()

    interp = {}
    for i, w in enumerate(words):
        params = fx_cache[w]
        for mod, mod_params in params.items():
            interp.setdefault(mod, {})
            for k, v in mod_params.items():
                if isinstance(v, list): v = v[0] if len(v) > 0 else 0.0
                interp[mod][k] = interp[mod].get(k, 0.0) + float(v) * weights[i]
    return interp

def run_text2fx(audio_path, text_prompt):
    out_sig, _, out_params_dict = t2fx(
        audio_path, ["eq"], [text_prompt],
        n_iters=200, params_init_type="curriculum",
        criterion="cosine-sim", pls_normalize=True,
    )
    out_path = TMP_AUDIO_DIR / "t2fx_result.wav"
    out_sig.write(str(out_path))

    with torch.no_grad():
        emb = torch.nn.functional.normalize(
            CLAP.get_audio_embeddings(preprocess_audio(out_sig).to(DEVICE)), dim=-1)
    proj_2d = pca_model.transform(emb.cpu().numpy())
    proj_2d = (proj_2d - coords_min) / coords_span
    params_dict = flatten_single_item_lists(detensor_dict(out_params_dict))
    return out_sig, proj_2d.squeeze().tolist(), params_dict

# -------------------------------------------------------------------------
# 5) Plot
# -------------------------------------------------------------------------
def make_scatter(user_point=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=coords_2d[:, 0], y=coords_2d[:, 1],
        text=EQ_WORDS, mode="markers+text",
        textposition="top center",
        marker=dict(size=12, color="orange"), name="Anchors"))
    if user_point is not None:
        fig.add_trace(go.Scatter(
            x=[user_point[0]], y=[user_point[1]],
            mode="markers", marker=dict(size=16, symbol="star", color="cyan", line=dict(width=1)),
            name="Your Text2FX"))
    fig.update_layout(
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        plot_bgcolor="black", paper_bgcolor="black",
        font=dict(color="white"), height=420, showlegend=True)
    return fig

# -------------------------------------------------------------------------
# 6) Gradio functions
# -------------------------------------------------------------------------
def morph(audio_path, x, y):
    params = interpolate_params((x, y), anchor_positions, fx_params_cache)
    sig = preprocess_audio(audio_path)
    out = apply_fx_to_sig(sig, params)
    sr = out.sample_rate
    samples = out.audio_data.squeeze().cpu().numpy()
    fig = make_scatter(user_point=(x, y))
    return (sr, samples), fig, params

def handle_text2fx(audio_path, text_prompt, _state):
    if not text_prompt or text_prompt.strip() == "":
        return gr.update(), make_scatter(), _state, 0.5, 0.5, {}
    out_sig, proj_xy, params = run_text2fx(audio_path, text_prompt)
    sr = out_sig.sample_rate
    samples = out_sig.audio_data.squeeze().cpu().numpy()
    fig = make_scatter(user_point=proj_xy)
    return (sr, samples), fig, proj_xy, float(proj_xy[0]), float(proj_xy[1]), params

# -------------------------------------------------------------------------
# 7) Gradio UI
# -------------------------------------------------------------------------
with gr.Blocks(title="Semantic Audio Explorer") as demo:
    gr.Markdown(
        "## 🎧 Text2FX + Audealize Semantic FX Explorer\n"
        "1️⃣ Type a description (e.g. *warm and full*) → run Text2FX.\n"
        "2️⃣ The optimized sound appears in the 2D semantic map (cyan star).\n"
        "3️⃣ Move sliders to explore nearby sounds and see updated FX params."
    )

    with gr.Row():
        audio_in = gr.Audio(type="filepath", label="Input Audio", value=DRY_AUDIO)
        audio_out = gr.Audio(label="Output Audio")

    prompt = gr.Textbox(label="Describe your target sound", placeholder="e.g., warm and full")
    run_btn = gr.Button("Run Text2FX & Project")
    projected_state = gr.State([0.5, 0.5])

    plot = gr.Plot(label="Semantic Map", value=make_scatter())
    x_slider = gr.Slider(0, 1, 0.5, step=0.01, label="X")
    y_slider = gr.Slider(0, 1, 0.5, step=0.01, label="Y")
    params_json = gr.JSON(label="FX Params (interpolated / optimized)")

    run_btn.click(
        fn=handle_text2fx,
        inputs=[audio_in, prompt, projected_state],
        outputs=[audio_out, plot, projected_state, x_slider, y_slider, params_json],
    )

    x_slider.change(fn=morph, inputs=[audio_in, x_slider, y_slider], outputs=[audio_out, plot, params_json])
    y_slider.change(fn=morph, inputs=[audio_in, x_slider, y_slider], outputs=[audio_out, plot, params_json])

demo.launch(server_port=7888, share=True)
