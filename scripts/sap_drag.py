import os, json, torch, numpy as np
from pathlib import Path
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
import gradio as gr
import gradio_client.utils as gu

# --- Safe patch for bool schemas (Gradio bug workaround) ---
_old_json_schema_to_python_type = gu._json_schema_to_python_type
_old_get_type = gu.get_type
def safe_json_schema_to_python_type(schema, defs=None):
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

# --- Text2FX imports ---
from text2fx.sap_main import get_model
from text2fx.core import preprocess_audio, detensor_dict, flatten_single_item_lists
from text2fx.applyFXparams import apply_fx_to_sig
from text2fx.sap_apply import main as t2fx
from text2fx.core import apply_audealize_single_word
from text2fx.constants import EQ_GAINS_PATH

# --- Setup ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLAP = get_model("ms_clap")
DRY_AUDIO = "/home/annie/research/text2fx/assets/guitar.wav"

SAVE_DIR = Path("/home/annie/research/text2fx/anchors")
SAVE_DIR.mkdir(exist_ok=True, parents=True)
EMB_PATH = SAVE_DIR / "audealize_anchor_embs.npy"
PARAMS_PATH = SAVE_DIR / "audealize_anchor_params.json"

EQ_WORDS = ["warm","bright","harsh","soft","cold",]#"full","airy","muffled","crisp"]
anchor_audio, anchor_embs, fx_params_cache = {}, [], {}

# --- Load or compute anchors ---
if EMB_PATH.exists() and PARAMS_PATH.exists():
    print("[INFO] Loading cached anchors...")
    anchor_embs = np.load(EMB_PATH)
    with open(PARAMS_PATH, "r") as f:
        fx_params_cache = json.load(f)
else:
    print("[INFO] Generating anchors...")
    embeddings_tmp = []
    for w in EQ_WORDS:
        wet_sig = apply_audealize_single_word(DRY_AUDIO, w, SAVE_DIR)
        wet_sig = preprocess_audio(wet_sig).to(DEVICE)
        with torch.no_grad():
            emb = torch.nn.functional.normalize(CLAP.get_audio_embeddings(wet_sig), dim=-1)
        anchor_audio[w] = wet_sig.cpu()
        embeddings_tmp.append(emb.cpu().numpy().squeeze())
    anchor_embs = np.stack(embeddings_tmp)
    np.save(EMB_PATH, anchor_embs)
    fx_params_cache = {}
    for w in EQ_WORDS:
        out_sig, _, out_params_dict = t2fx(DRY_AUDIO, ["eq"], [w], n_iters=400)
        fx_params_cache[w] = flatten_single_item_lists(detensor_dict(out_params_dict))
    with open(PARAMS_PATH, "w") as f:
        json.dump(fx_params_cache, f, indent=2)

# --- 2D positions ---
coords_2d = PCA(n_components=2).fit_transform(anchor_embs)
coords_2d = (coords_2d - coords_2d.min(0)) / (coords_2d.max(0) - coords_2d.min(0))
anchor_positions = {w: coords_2d[i] for i, w in enumerate(EQ_WORDS)}

# --- Interpolation ---
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

# --- Morph ---
def morph(audio_path, x, y):
    params = interpolate_params((x,y), anchor_positions, fx_params_cache)
    sig = preprocess_audio(audio_path)
    out = apply_fx_to_sig(sig, params)
    out_path = "text2fx/tmp/morphed.wav"
    out.write(out_path)
    return out_path

# --- Refine ---
def refine_from_map(audio_path, x, y, text_prompt):
    params = interpolate_params((x,y), anchor_positions, fx_params_cache)
    _, _, refined_dict = t2fx(
        audio_path, ["eq"], [text_prompt], n_iters=150,
        params_init_type="curriculum", criterion="cosine-sim",
        pls_normalize=True, custom_embedding_target=None)
    new_params = flatten_single_item_lists(detensor_dict(refined_dict))
    sig = preprocess_audio(audio_path)
    out = apply_fx_to_sig(sig, new_params)
    out_path = "text2fx/tmp/refined.wav"
    out.write(out_path)
    return out_path

# --- Plotly scatter data ---
import plotly.graph_objects as go
def make_plot():
    fig = go.Figure()
    # anchors
    fig.add_trace(go.Scatter(
        x=coords_2d[:,0], y=coords_2d[:,1],
        text=EQ_WORDS, mode="markers+text",
        textposition="top center",
        marker=dict(size=12, color="orange"), hoverinfo="text"
    ))
    # draggable cursor
    fig.add_trace(go.Scatter(
        x=[0.5], y=[0.5],
        mode="markers", marker=dict(size=16, color="cyan"), name="cursor"
    ))
    fig.update_layout(
        xaxis=dict(range=[0,1], visible=False),
        yaxis=dict(range=[0,1], visible=False),
        plot_bgcolor="black", paper_bgcolor="black",
        height=400, dragmode="pan",
        margin=dict(l=0,r=0,t=30,b=0),
        title="🌀 Audealize Semantic Space — drag cyan dot to morph"
    )
    return fig

# --- Gradio App ---
with gr.Blocks(title="Text2FX Semantic Explorer") as demo:
    gr.Markdown("## 🎧 Text2FX + Audealize Semantic Explorer\nDrag the cyan dot to morph between timbral anchors.")
    with gr.Row():
        audio_in = gr.Audio(type="filepath", label="Input Audio", value=DRY_AUDIO)
        audio_out = gr.Audio(label="Processed Output")

    prompt = gr.Textbox(label="Describe target sound (optional)")
    plot = gr.Plot(make_plot())
    x_val = gr.Number(0.5, visible=False, elem_id="x_val")
    y_val = gr.Number(0.5, visible=False, elem_id="y_val")

    # JS bridge for drag interactions
    gr.HTML("""
    <script>
    window.addEventListener('DOMContentLoaded', () => {
        const plot = document.querySelector('div.js-plotly-plot');
        if (!plot) return;
        let dragging = false;

        function moveCursor(x, y) {
            // Find hidden input fields by id
            const iframe = window.gradioApp();
            if (!iframe) return;
            const x_input = iframe.querySelector('#x_val input');
            const y_input = iframe.querySelector('#y_val input');
            if (!x_input || !y_input) return;
            // update values
            x_input.value = x;
            y_input.value = y;
            // force Gradio change events so backend updates fire
            x_input.dispatchEvent(new Event("input", { bubbles: true }));
            x_input.dispatchEvent(new Event("change", { bubbles: true }));
            y_input.dispatchEvent(new Event("input", { bubbles: true }));
            y_input.dispatchEvent(new Event("change", { bubbles: true }));
        }

        plot.on('plotly_click', (data) => {
            const p = data.points[0];
            if (!p) return;
            dragging = true;
            moveCursor(p.x, p.y);
        });

        plot.addEventListener('mousemove', evt => {
            if (!dragging) return;
            const bb = plot.getBoundingClientRect();
            const px = (evt.clientX - bb.left) / bb.width;
            const py = 1 - (evt.clientY - bb.top) / bb.height;
            moveCursor(px, py);
        });

        window.addEventListener('mouseup', () => dragging = false);
    });
    </script>
    """)


    # trigger morph when dragging
    gr.on(
        triggers=[x_val.change, y_val.change],
        fn=morph,
        inputs=[audio_in, x_val, y_val],
        outputs=[audio_out],
        show_progress=False
    )

    # refinement button
    gr.Button("Refine via Text2FX").click(
        fn=refine_from_map,
        inputs=[audio_in, x_val, y_val, prompt],
        outputs=[audio_out]
    )

demo.launch(server_name="127.0.0.1", inbrowser=True)
