import torch, numpy as np
from itertools import combinations
from text2fx.sap_main import get_model
from text2fx.core import preprocess_audio, detensor_dict, flatten_single_item_lists, create_channel
from text2fx.applyFXparams import apply_fx_to_sig
from text2fx.sap_apply import main as t2fx
from audiotools import AudioSignal
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import gradio as gr
from text2fx.core import apply_audealize_single_word, get_settings_for_words
from text2fx.constants import EQ_freq_bands, EQ_GAINS_PATH
import os, json
from pathlib import Path

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

# -------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLAP = get_model("ms_clap")
DRY_AUDIO = "/home/annie/research/text2fx/assets/guitar.wav"
os.makedirs("text2fx/tmp", exist_ok=True)

# --- Audealize anchors --------------------------------------------------


# --- setup ---

SAVE_DIR = Path("/home/annie/research/text2fx/anchors")
SAVE_DIR.mkdir(exist_ok=True, parents=True)
EMB_PATH = SAVE_DIR / "audealize_anchor_embs.npy"
EQ_WORDS = ["warm","bright","harsh","soft","cold","full","airy","muffled","crisp"]
PARAMS_PATH = SAVE_DIR / "audealize_anchor_params.json"

anchor_audio, anchor_embs, fx_params_cache = {}, [], {}


# --- Try to load cached data first ---
if EMB_PATH.exists() and PARAMS_PATH.exists():
    print(f"[INFO] Loading precomputed anchor embeddings and parameters...")
    anchor_embs = np.load(EMB_PATH)
    with open(PARAMS_PATH, "r") as f:
        fx_params_cache = json.load(f)

else:
    print("[INFO] Generating Audealize anchors and embeddings ...")
    embeddings_tmp = []
    for w in EQ_WORDS:
        print(f"→ {w}")
        wet_sig = apply_audealize_single_word(DRY_AUDIO, w, SAVE_DIR)
        wet_sig = preprocess_audio(wet_sig).to(DEVICE)
        with torch.no_grad():
            emb = torch.nn.functional.normalize(CLAP.get_audio_embeddings(wet_sig), dim=-1)
        anchor_audio[w] = wet_sig.cpu()
        embeddings_tmp.append(emb.cpu().numpy().squeeze())

    # Stack and save embeddings
    anchor_embs = np.stack(embeddings_tmp)
    np.save(EMB_PATH, anchor_embs)
    print(f"[INFO] Saved precomputed embeddings to {EMB_PATH}")

    # Precompute and save FX parameter sets
    print("[INFO] Generating FX parameter caches ...")
    fx_params_cache = {}
    for w in EQ_WORDS:
        print(f"→ Computing FX params for {w}")
        out_sig, _, out_params_dict = t2fx(DRY_AUDIO, ["eq"], [w], n_iters=400)
        fx_params_cache[w] = flatten_single_item_lists(detensor_dict(out_params_dict))

    with open(PARAMS_PATH, "w") as f:
        json.dump(fx_params_cache, f, indent=2)
    print(f"[INFO] Saved FX parameter cache to {PARAMS_PATH}")
# --- Build 2D layout ---
from sklearn.decomposition import PCA

coords_2d = PCA(n_components=2).fit_transform(anchor_embs)
coords_2d = (coords_2d - coords_2d.min(0)) / (coords_2d.max(0) - coords_2d.min(0))
anchor_positions = {w: coords_2d[i] for i, w in enumerate(EQ_WORDS)}

print("[INFO] Anchors ready:")
for w in EQ_WORDS:
    print(f"  {w:>8} → coords {anchor_positions[w]}")
# SAVE_DIR = "./"
# embeddings = {}

# print("Generating Audealize anchors ...")
# anchor_audio, anchor_embs = {}, []
# for w in EQ_WORDS:
#     wet_sig = apply_audealize_single_word(DRY_AUDIO, w, "./anchors")  # your existing fn
#     wet_sig = preprocess_audio(wet_sig).to(DEVICE)
#     with torch.no_grad():
#         emb = torch.nn.functional.normalize(CLAP.get_audio_embeddings(wet_sig), dim=-1)
#     anchor_audio[w] = wet_sig.cpu()
#     anchor_embs.append(emb.cpu().numpy().squeeze())

# anchor_embs = np.stack(anchor_embs)
# coords_2d = PCA(n_components=2).fit_transform(anchor_embs)
# coords_2d = (coords_2d - coords_2d.min(0)) / (coords_2d.max(0)-coords_2d.min(0))
# anchor_positions = {w: coords_2d[i] for i, w in enumerate(EQ_WORDS)}

# #Precompute anchor FX param dicts, 
# fx_params_cache = {}
# for w in EQ_WORDS:
#     out_sig, _, out_params_dict = t2fx(DRY_AUDIO, ["eq"], [w], n_iters=200)
#     fx_params_cache[w] = flatten_single_item_lists(detensor_dict(out_params_dict))
# ## alt -- just downsampling
# # Define broad 6 bands (in Hz)

# --- Interpolation helper -----------------------------------------------
def interpolate_params(pos, anchor_positions, fx_params_cache, sharpness=10.0):
    words = list(anchor_positions.keys())
    coords = np.array([anchor_positions[w] for w in words])
    dists = cdist([pos], coords)[0] + 1e-6
    weights = np.exp(-sharpness * dists); weights /= weights.sum()
    interp = {}
    for i, w in enumerate(words):
        params = fx_params_cache[w]
        for mod, mod_params in params.items():
            interp.setdefault(mod, {})
            for k, v in mod_params.items():
                v = float(v[0] if isinstance(v, list) else v)
                interp[mod][k] = interp[mod].get(k, 0.0) + v * weights[i]
    return interp

# --- Refinement helper (Text2FX short run) -------------------------------
def refine(audio_path, text_prompt, params_dict):
    sig = preprocess_audio(audio_path)
    _, _, refined_dict = t2fx(
        audio_path,
        ["eq"],
        [text_prompt],
        n_iters=150,
        params_init_type="curriculum",
        criterion="cosine-sim",
        pls_normalize=True,
        custom_embedding_target=None
    )
    return flatten_single_item_lists(detensor_dict(refined_dict))

# --- Morph function for Gradio ------------------------------------------
def morph(audio_path, x, y):
    params = interpolate_params((x,y), anchor_positions, fx_params_cache)
    sig = preprocess_audio(audio_path)
    out = apply_fx_to_sig(sig, params)
    out_path = "text2fx/tmp/morphed.wav"
    out.write(out_path)
    return out_path

# --- Optional refine wrapper --------------------------------------------
def refine_from_map(audio_path, x, y, text_prompt):
    params = interpolate_params((x,y), anchor_positions, fx_params_cache)
    new_params = refine(audio_path, text_prompt, params)
    sig = preprocess_audio(audio_path)
    out = apply_fx_to_sig(sig, new_params)
    out_path = "text2fx/tmp/refined.wav"
    out.write(out_path)
    return out_path

# --- 2D Visualization ---------------------------------------------------
# --- 2D Visualization (interactive) -----------------------------------
import plotly.graph_objects as go

# def make_scatter():
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(
#         x=coords_2d[:, 0],
#         y=coords_2d[:, 1],
#         text=EQ_WORDS,
#         mode="markers+text",
#         textposition="top center",
#         marker=dict(size=12, color="orange")
#     ))
#     fig.update_layout(
#         xaxis=dict(visible=False),
#         yaxis=dict(visible=False),
#         plot_bgcolor="black",
#         paper_bgcolor="black",
#         font=dict(color="white"),
#         title="🌀 Audealize / Text2FX Semantic Space (click to explore)",
#         height=400
#     )
#     return fig
def make_scatter():
    return {
        "x": coords_2d[:, 0].tolist(),
        "y": coords_2d[:, 1].tolist(),
        "text": EQ_WORDS,
        "color": ["orange"] * len(EQ_WORDS),
    }


# --- Click-driven Gradio UI -------------------------------------------
with gr.Blocks(title="Semantic Audio Explorer") as demo:
    gr.Markdown("## 🎧 Text2FX + Audealize Semantic Explorer\n"
                "Click in the 2D space to explore morphs between timbral anchors.")

    with gr.Row():
        audio_in = gr.Audio(type="filepath", label="Input Audio", value=DRY_AUDIO)
        audio_out = gr.Audio(label="Processed Output")

    prompt = gr.Textbox(label="Describe your target sound", placeholder="e.g. warmer and brighter")
    run_btn = gr.Button("Run Text2FX & Load 2D Space")


    plot = gr.ScatterPlot(label="Semantic Map (click to morph)")
    run_btn.click(fn=lambda: gr.ScatterPlot.update(value=make_scatter()), outputs=[plot])

    @plot.select
    def on_click(select_data):
        if not select_data or "points" not in select_data or not select_data["points"]:
            return gr.update()
        x = float(select_data["points"][0]["x"])
        y = float(select_data["points"][0]["y"])
        print(f"Clicked at ({x:.2f}, {y:.2f})")
        out_path = morph(DRY_AUDIO, x, y)
        return out_path


    refine_btn = gr.Button("Refine (semantic Text2FX)")
    refine_btn.click(fn=refine_from_map,
                     inputs=[audio_in, gr.Number(0), gr.Number(0), prompt],
                     outputs=[audio_out])

demo.launch(server_port=7887, share=True)

