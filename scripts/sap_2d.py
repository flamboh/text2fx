import text2fx
from text2fx.sap_apply import main as t2fx
import torch
from text2fx.sap_main import get_model
from text2fx.core import preprocess_audio, create_channel
from text2fx.core_plotting import load_eq_params, parametric_eq_response_biquad

import os
import gradio_client.utils as gu
os.environ["TOKENIZERS_PARALLELISM"] = "true"

"""Script currently is super rough 
It should be where the 2d explore page lives, maybe precomputed embeddings for speed?
"""

# --- Full monkey patch to make Gradio tolerant of boolean schemas ---
_old_json_schema_to_python_type = gu._json_schema_to_python_type
_old_get_type = gu.get_type

def safe_json_schema_to_python_type(schema, defs=None):
    # If schema is True/False, return a dummy string type
    if isinstance(schema, bool):
        return "object" if schema else "never"
    try:
        return _old_json_schema_to_python_type(schema, defs)
    except Exception:
        return "object"

def safe_get_type(schema):
    if isinstance(schema, bool):
        return "object" if schema else "never"
    try:
        return _old_get_type(schema)
    except Exception:
        return "object"

gu._json_schema_to_python_type = safe_json_schema_to_python_type
gu.get_type = safe_get_type
# --- End of patch ---
import os, json, numpy as np
import gradio as gr
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

import torch
from text2fx.sap_apply import main as t2fx
from text2fx.core import detensor_dict, flatten_single_item_lists, preprocess_audio
from text2fx.applyFXparams import apply_fx_to_sig

# -------------------------------------------------------------------------
# 1. Setup
# -------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FX_LIST = ["eq"]#, "reverb"]
os.makedirs("text2fx/tmp", exist_ok=True)

PROMPTS = ["bright", "dark", "warm", "harsh", "spacious", "dry", "thin", "full", "airy"]
print(f"Generating {len(PROMPTS)} semantic anchor points...")

# -------------------------------------------------------------------------
# 2. Generate anchor FX presets & embeddings
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# 2. Load or generate anchor FX presets & embeddings
# -------------------------------------------------------------------------
EMBED_PATH = "text2fx/tmp/embed_coords.npy"
FX_PRESETS_PATH = "text2fx/tmp/fx_presets.json"

if os.path.exists(EMBED_PATH) and os.path.exists(FX_PRESETS_PATH):
    print("✅ Loading precomputed embedding coordinates and FX presets...")
    coords_2d = np.load(EMBED_PATH)
    with open(FX_PRESETS_PATH, "r") as f:
        fx_params = json.load(f)
else:
    print("⚙️ Computing embeddings and FX parameters for prompts...")
    from text2fx.sap_main import get_model
    model = get_model("ms_clap")

    embeddings, fx_params = [], []
    for prompt in PROMPTS:
        print(f"→ Optimizing FX for prompt: {prompt}")
        out_sig, _, out_params_dict = t2fx(
            "/home/annie/research/text2fx/assets/guitar.wav",
            FX_LIST, [prompt],
            n_iters=400, params_init_type="curriculum",
            criterion="cosine-sim", pls_normalize=True,
        )
        params = flatten_single_item_lists(detensor_dict(out_params_dict))
        fx_params.append(params)

        emb = model.get_text_embeddings([prompt])
        embeddings.append(emb.cpu().numpy().squeeze())

    embeddings = np.stack(embeddings)
    coords_2d = PCA(n_components=2).fit_transform(embeddings)
    coords_2d = (coords_2d - coords_2d.min(0)) / (coords_2d.max(0) - coords_2d.min(0))

    np.save(EMBED_PATH, coords_2d)
    with open(FX_PRESETS_PATH, "w") as f:
        json.dump(fx_params, f, indent=2)
    print("✅ Anchors computed and saved.")

# -------------------------------------------------------------------------
# 3. Helper functions
# -------------------------------------------------------------------------
def interpolate_params(anchor_coords, fx_dicts, pos):
    """Blend FX parameters of nearby anchors by distance-weighted average."""
    dists = cdist([pos], anchor_coords)[0] + 1e-6
    weights = np.exp(-dists * 10)
    weights /= weights.sum()

    interp = {}
    for i, params in enumerate(fx_dicts):
        for mod, mod_params in params.items():
            interp.setdefault(mod, {})
            for k, v in mod_params.items():
                v = float(v[0] if isinstance(v, list) else v)
                interp[mod][k] = [interp[mod].get(k, [0.0])[0] + v * weights[i]]

    return interp
def compute_eq_response(eq_params_dict, sample_rate=44100):
    """Bridge from fx_params JSON dict to frequency response arrays."""
    # mimic the JSON format expected by load_eq_params
    json_like = {"params": {"ParametricEQ": eq_params_dict}}

    # Extract parameter arrays
    cutoffs, gains, q_factors = load_eq_params(json_like)
    freqs = np.logspace(np.log10(20), np.log10(sample_rate / 2), 512)
    response_db = parametric_eq_response_biquad(freqs, gains, cutoffs, q_factors, sample_rate)
    return freqs, response_db


def render_audio(audio_path, params_dict):
    # unwrap any [value] lists into floats for DSP compatibility
    def unwrap(v):
        if isinstance(v, list) and len(v) == 1:
            return v[0]
        if isinstance(v, dict):
            return {k: unwrap(vv) for k, vv in v.items()}
        return v

    params_dict = unwrap(params_dict)

    sig = preprocess_audio(audio_path)
    out = apply_fx_to_sig(sig, params_dict)
    out_path = "tmp/rendered.wav"
    out.write(out_path)
    return out_path


# -------------------------------------------------------------------------
# 4. Gradio logic
# -------------------------------------------------------------------------
def morph(audio_path, x, y):
    """Triggered when user clicks or drags in the map."""
    params = interpolate_params(coords_2d, fx_params, (x, y))
    out_path = render_audio(audio_path, params)

    eq_params = params.get("ParametricEQ", {})
    freqs, mag_db = compute_eq_response(eq_params)

    # return both the processed audio path and the new EQ curve
    cutoffs, _, _ = load_eq_params({"params": {"ParametricEQ": eq_params}})
    eq_fig = make_eq_plot(freqs, mag_db, cutoffs=cutoffs)
    return out_path, eq_fig

def make_eq_plot(freqs, mag_db, cutoffs=None):
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freqs, y=mag_db, mode="lines", line=dict(color="cyan"), name="EQ Curve"))
    if cutoffs is not None:
        fig.add_trace(go.Scatter(x=cutoffs, y=np.interp(cutoffs, freqs, mag_db),
                                 mode="markers", marker=dict(color="red", size=8),
                                 name="Bands"))
    fig.update_layout(xaxis_type="log", yaxis=dict(range=[-20,20]), template="plotly_dark")
    return fig

# -------------------------------------------------------------------------
# 5. Gradio UI
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# 5. Gradio UI (slider version)
# -------------------------------------------------------------------------
def make_scatter():
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=coords_2d[:, 0], y=coords_2d[:, 1],
        mode="markers+text", text=PROMPTS,
        textposition="top center", marker=dict(size=10, color="orange")
    ))
    fig.update_layout(
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        plot_bgcolor="black", paper_bgcolor="black",
        title="Semantic FX Space (adjust sliders to move through space)",
        font=dict(color="white")
    )
    return fig

with gr.Blocks(title="Semantic FX Explorer") as demo:
    gr.Markdown("## 🎧 Semantic FX Explorer\nAdjust the sliders to explore the semantic FX space.")

    with gr.Row():
        audio_in = gr.Audio(type="filepath", label="Input Audio")
        audio_out = gr.Audio(label="Processed Output")

    plot = gr.Plot(label="Semantic FX Map", value=make_scatter())
    eq_plot = gr.Plot(label="EQ Curve")

    with gr.Row():
        x_slider = gr.Slider(0, 1, 0.5, step=0.01, label="X Position")
        y_slider = gr.Slider(0, 1, 0.5, step=0.01, label="Y Position")

    # Update both audio + EQ curve
    x_slider.change(fn=morph, inputs=[audio_in, x_slider, y_slider],
                    outputs=[audio_out, eq_plot])
    y_slider.change(fn=morph, inputs=[audio_in, x_slider, y_slider],
                    outputs=[audio_out, eq_plot])

demo.launch(server_port=7889, share=True)