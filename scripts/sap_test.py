"""
Text2FX: Unified 3-tier semantic control demo (EQ + Reverb + Presets)

Flow:
1. User uploads audio + enters text --> runs Text2FX optimization
2. User tweaks mid- and low-level sliders (both edit the same FX param dict)
3. Any change re-renders the processed output
4. User can save/recall presets with intensity scaling (α)
"""

import os, json, torch
import numpy as np
import matplotlib.pyplot as plt
import librosa, librosa.display
import gradio as gr
import gradio_client.utils as gu

from text2fx.sap_apply import main as t2fx
from text2fx.core import (
    detensor_dict,
    preprocess_audio,
    create_channel,
    flatten_single_item_lists,
)
from text2fx.applyFXparams import apply_fx_to_sig

# -------------------------------------------------------------------------
# Environment setup
# -------------------------------------------------------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "true"
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
# Globals
# -------------------------------------------------------------------------
SCRAP_DIR = "experiments/tmp/"
os.makedirs(SCRAP_DIR, exist_ok=True)

FX_LIST = ["eq", "reverb"]
CHANNEL = create_channel(FX_LIST)

# -------------------------------------------------------------------------
# Utility: spectrogram & caching
# -------------------------------------------------------------------------
from pathlib import Path

_cached_signals = {}

def get_cached_signal(path):
    key = Path(path).name  # only filename, not /tmp/gradio path
    if key not in _cached_signals:
        _cached_signals[key] = preprocess_audio(path)
    return _cached_signals[key]


def make_spectrogram(y, sr, title, out_path):
    out_path = os.path.join(SCRAP_DIR, out_path)
    S = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    fig, ax = plt.subplots(figsize=(6, 3))
    img = librosa.display.specshow(S, sr=sr, x_axis="time", y_axis="hz", cmap="magma", ax=ax)
    ax.set(title=title)
    plt.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path

# -------------------------------------------------------------------------
# Parameter utilities
# -------------------------------------------------------------------------
def build_param_spec(channel):
    specs = []
    for m in channel.modules:
        mod = type(m).__name__
        for pname, (pmin, pmax) in m.param_ranges.items():
            specs.append((mod, pname, float(pmin), float(pmax), (pmin + pmax) / 2))
    return specs

def params_to_values(params_dict, specs):
    vals = []
    for mod, pn, pmin, pmax, default in specs:
        val = params_dict.get(mod, {}).get(pn, default)
        if isinstance(val, torch.Tensor): val = val.item()
        if isinstance(val, list) and len(val)==1: val = val[0]
        vals.append(float(val))
    return vals

def values_to_params(values, specs):
    out = {}
    for (mod, pn, *_), val in zip(specs, values):
        out.setdefault(mod, {})[pn] = float(val)
    return out

# -------------------------------------------------------------------------
# Mid-level semantic mapping
# -------------------------------------------------------------------------
def apply_mid_controls(params, tone, space, channel):
    import copy
    params = copy.deepcopy(params)

    param_ranges = {type(m).__name__: m.param_ranges for m in channel.modules}

    # EQ tone
    if "ParametricEQ" in params:
        eq = params["ParametricEQ"]
        ranges = param_ranges.get("ParametricEQ", {})
        tone_delta = tone - 0.5
        for k, v in eq.items():
            v = float(v[0] if isinstance(v, list) else v)
            if "low_shelf" in k.lower() and "gain" in k.lower():
                eq[k] = v - tone_delta * 6.0
            elif "high_shelf" in k.lower() and "gain" in k.lower():
                eq[k] = v + tone_delta * 6.0
            elif "band" in k.lower() and "gain" in k.lower():
                eq[k] = v - abs(tone_delta) * 3.0
            if k in ranges:
                pmin, pmax = ranges[k]
                eq[k] = float(np.clip(eq[k], pmin, pmax))

    # Reverb space
    for mod, fx in params.items():
        if "reverb" in mod.lower():
            ranges = param_ranges.get("NoiseShapedReverb", {})
            for k, v in fx.items():
                v = float(v[0] if isinstance(v, list) else v)
                if "band" in k.lower() and "decay" in k.lower():
                    fx[k] = v * (0.5 + space * 0.5)
                elif "mix" in k.lower():
                    fx[k] = space
                if k in ranges:
                    pmin, pmax = ranges[k]
                    fx[k] = float(np.clip(fx[k], pmin, pmax))
    return params

def estimate_mid_from_params(params_dict):
    tone, space = 0.0, 0.0
    if "ParametricEQ" in params_dict:
        eq = params_dict["ParametricEQ"]
        highs = [v for k, v in eq.items() if "high_shelf" in k.lower() and "gain" in k.lower()]
        lows  = [v for k, v in eq.items() if "low_shelf"  in k.lower() and "gain" in k.lower()]
        if highs or lows:
            tone = np.tanh((np.mean(highs or [0]) - np.mean(lows or [0])) / 12)
    for mod, fx in params_dict.items():
        if "reverb" in mod.lower():
            space = float(np.clip(fx.get("mix", 0.0), 0.0, 1.0))
            break
    return float(np.clip(tone, -1, 1)), float(space)

# -------------------------------------------------------------------------
# Rendering
# -------------------------------------------------------------------------
def render_audio_from_params(audio_file, params_dict):
    # sig = get_cached_signal(audio_file)
    sig = preprocess_audio(audio_file)
    out = apply_fx_to_sig(sig, params_dict)
    sr = out.sample_rate
    y = out.samples[0, 0].cpu().numpy()
    spec = make_spectrogram(y, sr, "Rendered Output", "spec_render.png")
    return (sr, y), spec

# -------------------------------------------------------------------------
# Core operations
# -------------------------------------------------------------------------
def run_text2fx(audio_file, text_prompt, specs):
    out_sig, _, out_params_dict = t2fx(
        audio_file, FX_LIST, [text_prompt],
        n_iters=60, params_init_type="curriculum",
        criterion="cosine-sim", roll_amt=3000, pls_normalize=True,
    )
    params = flatten_single_item_lists(detensor_dict(out_params_dict))
    vals = params_to_values(params, specs)
    tone_est, space_est = estimate_mid_from_params(params)
    y = out_sig.samples[0, 0].cpu().numpy()
    sr = out_sig.sample_rate
    spec = make_spectrogram(y, sr, "Optimized Output", "spec_opt.png")
    return (sr, y), json.dumps(params, indent=2), params, spec, tone_est, space_est, *vals

def update_mid(audio_file, tone, space, params_base, specs):
    params_new = apply_mid_controls(params_base, 0.5 + tone * 0.5, space, CHANNEL)
    return *render_audio_from_params(audio_file, params_new), params_new, json.dumps(params_new, indent=2), *params_to_values(params_new, specs)

def update_low(audio_file, *args):
    *values, specs, params_base = args
    params_new = values_to_params(values, specs)
    return *render_audio_from_params(audio_file, params_new), params_new, json.dumps(params_new, indent=2), *values

def scale_params_intensity(params_dict, alpha: float):
    scaled = {}
    for fx, params in params_dict.items():
        out = {}
        for k, v in params.items():
            v = float(v[0] if isinstance(v, list) else v)
            if any(t in k.lower() for t in ["gain","mix","drive","amount","depth","level"]):
                out[k] = v * alpha
            else:
                out[k] = v
        scaled[fx] = out
    return scaled

def apply_preset(audio_file, preset_name, alpha, presets, specs):
    if preset_name not in presets:
        return None, "Preset not found.", None, None, *[0]*len(specs)
    scaled = scale_params_intensity(presets[preset_name], alpha)
    return *render_audio_from_params(audio_file, scaled), scaled, json.dumps(scaled, indent=2), *params_to_values(scaled, specs)

def save_preset(params, name, presets):
    if not name.strip():
        return presets, "⚠️ Enter preset name."
    newp = dict(presets)
    newp[name] = params
    return newp, f"💾 Saved preset '{name}'."

# -------------------------------------------------------------------------
# UI
# -------------------------------------------------------------------------
with gr.Blocks(title="Text2FX — Unified Semantic FX") as demo:
    gr.Markdown("## 🎧 Text2FX: Unified Semantic-to-Parametric FX Control (EQ + Reverb)")
    SPECS = build_param_spec(CHANNEL)
    state_params = gr.State()
    state_specs = gr.State(SPECS)
    state_presets = gr.State({})

    audio_in = gr.Audio(sources=["upload"], type="filepath", label="🎵 Input Audio")
    text_in = gr.Textbox(value="warm and spacious", label="Text Prompt")
    run_btn = gr.Button("Run Text2FX")

    output_audio = gr.Audio(label="Processed Output")
    fx_json = gr.Textbox(label="FX Parameters", lines=12)
    spec_img = gr.Image(label="Spectrogram")

    with gr.Accordion("🎛️ Semantic Controls (Mid-Level)", open=True):
        tone = gr.Slider(-1, 1, 0, 0.01, label="Tone (Dark ↔ Bright)")
        space = gr.Slider(0, 1, 0.3, 0.01, label="Space (Dry → Spacious)")

    with gr.Accordion("🔧 Low-Level FX Parameters", open=False):
        sliders = [gr.Slider(pmin, pmax, default, label=f"{mod}.{pn}") for mod, pn, pmin, pmax, default in SPECS]

    preset_name = gr.Textbox(label="Preset Name", placeholder="e.g. WarmHallVocal")
    save_btn = gr.Button("💾 Save Preset")
    preset_dropdown = gr.Dropdown(label="🎚️ Select Preset", interactive=True)
    alpha = gr.Slider(0, 1, 1.0, 0.05, label="Preset Intensity (α)")
    apply_btn = gr.Button("Apply Preset")

    # ---- WIRING ----
    run_btn.click(
        fn=run_text2fx,
        inputs=[audio_in, text_in, state_specs],
        outputs=[output_audio, fx_json, state_params, spec_img, tone, space, *sliders],
    )

    tone.change(fn=update_mid,
        inputs=[audio_in, tone, space, state_params, state_specs],
        outputs=[output_audio, spec_img, state_params, fx_json, *sliders])

    space.change(fn=update_mid,
        inputs=[audio_in, tone, space, state_params, state_specs],
        outputs=[output_audio, spec_img, state_params, fx_json, *sliders])

    for s in sliders:
        s.change(fn=update_low,
            inputs=[audio_in, *sliders, state_specs, state_params],
            outputs=[output_audio, spec_img, state_params, fx_json, *sliders])

    save_btn.click(
        fn=save_preset,
        inputs=[state_params, preset_name, state_presets],
        outputs=[state_presets, fx_json],
    ).then(fn=lambda d: list(d.keys()), inputs=[state_presets], outputs=[preset_dropdown])

    apply_btn.click(
        fn=apply_preset,
        inputs=[audio_in, preset_dropdown, alpha, state_presets, state_specs],
        outputs=[output_audio, spec_img, state_params, fx_json, *sliders],
    )

demo.launch(server_port=7869, share=True)
