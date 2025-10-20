import gradio as gr
import torch
import json
import os
import gradio_client.utils as gu

from text2fx.sap_apply import main as t2fx
from text2fx.core import detensor_dict, preprocess_audio, create_channel, flatten_single_item_lists
from text2fx.applyFXparams import apply_fx_to_sig

# -------------------------------------------------------------------------
# Environment and monkey patch for Gradio schema quirks
# -------------------------------------------------------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "true"

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

# -------------------------------------------------------------------------
# Settings
# -------------------------------------------------------------------------
FX_LIST = ["eq", "reverb"]  # whatever FX modules your channel supports

# -------------------------------------------------------------------------
# Helper: Scale only intensity-like parameters
# -------------------------------------------------------------------------
def scale_params_intensity(params_dict: dict, alpha: float) -> dict:
    """
    Scale only 'intensity-like' parameters by alpha.
    Leaves frequencies / Q / time constants unchanged.
    Handles compressor thresholds specially.
    """
    scaled = {}

    def is_intensity_key(key: str) -> bool:
        k = key.lower()
        return any(tag in k for tag in [
            "gain", "mix", "drive", "amount", "depth", "ratio", "makeup", "level"
        ])

    def is_threshold_key(key: str) -> bool:
        return "threshold" in key.lower()

    def is_structural_key(key: str) -> bool:
        k = key.lower()
        return any(tag in k for tag in [
            "freq", "frequency", "q", "attack", "release", "decay",
            "predelay", "pre_delay", "time", "delay"
        ])

    for fx, params in params_dict.items():
        fx_out = {}
        for k, v in params.items():
            # Handle tensors
            if isinstance(v, torch.Tensor):
                v = v.item()
            # Skip non-numerics
            if not isinstance(v, (int, float)):
                fx_out[k] = v
                continue

            if is_structural_key(k):
                fx_out[k] = v
            elif is_threshold_key(k):
                fx_out[k] = v * alpha  # move toward 0 as alpha → 0
            elif is_intensity_key(k):
                fx_out[k] = v * alpha  # scale intensity
            else:
                fx_out[k] = v
        scaled[fx] = fx_out
    return scaled

# -------------------------------------------------------------------------
# Core functions
# -------------------------------------------------------------------------
def run_text2fx(audio_file, text_prompt):
    """
    Runs text2fx optimization once and returns:
      - processed audio at full α=1.0
      - readable parameter summary
      - raw parameter dict (for state)
    """
    out_sig, out_params, out_params_dict = t2fx(
        audio_file,
        FX_LIST,
        [text_prompt],
        n_iters=150,
        params_init_type="curriculum",
        criterion="cosine-sim",
        roll_amt=3000,
        pls_normalize=True,
    )

    # Convert tensors → Python types
    params_dict = detensor_dict(out_params_dict)
    params_dict = flatten_single_item_lists(params_dict)
    sr = out_sig.sample_rate
    fx_summary = json.dumps(params_dict, indent=2)

    return (sr, out_sig.samples[0, 0].cpu().numpy()), fx_summary, params_dict


def update_alpha(audio_file, alpha, params_dict):
    """
    Re-applies the cached FX parameters scaled by α.
    """
    if params_dict is None:
        return None, "Run text2fx first."

    # 1. Scale parameters
    scaled = scale_params_intensity(params_dict, alpha)
    scaled = flatten_single_item_lists(scaled) 

    # 2. Apply DSP to the dry signal
    input_sig = preprocess_audio(audio_file)
    out_sig = apply_fx_to_sig(input_sig, scaled)

    # 3. Format outputs
    sr = out_sig.sample_rate
    final_np = out_sig.samples[0, 0].cpu().numpy()
    fx_summary = json.dumps(scaled, indent=2)
    return (sr, final_np), fx_summary

# -------------------------------------------------------------------------
# Gradio UI
# -------------------------------------------------------------------------
with gr.Blocks(title="Text2FX DAW-like Demo") as demo:
    gr.Markdown("## 🎧 Text2FX Interactive DAW-Like Demo\n"
                "Describe a sound and control its semantic intensity (α).")

    # State variable to store optimized FX parameters
    state_params = gr.State()

    # --- Input section ---
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                sources=["upload"], type="filepath", label="🎵 Input Audio"
            )
            dry_playback = gr.Audio(label="Original Audio (Dry)")
        with gr.Column():
            text_prompt = gr.Textbox(
                value="warm and intimate",
                label="Describe how you want it to sound",
                lines=2,
            )
            run_btn = gr.Button("Generate Plausible FX (Optimize)")

    # --- Output section ---
    with gr.Row():
        output_audio = gr.Audio(label="Processed Output")
        fx_params_box = gr.Textbox(
            label="Low-level FX Parameters", lines=12, interactive=False
        )

    alpha_slider = gr.Slider(
        0, 1, value=1.0, step=0.05, label="Effect Intensity (alpha)"
    )

    # --- Wiring ---
    # Step 1: run optimization
    run_btn.click(
        fn=run_text2fx,
        inputs=[audio_input, text_prompt],
        outputs=[output_audio, fx_params_box, state_params],
    )

    # Step 2: live α updates
    alpha_slider.change(
        fn=update_alpha,
        inputs=[audio_input, alpha_slider, state_params],
        outputs=[output_audio, fx_params_box],
    )

    # Step 3: optional dry playback
    audio_input.change(
        fn=lambda a: (a if a else None),
        inputs=audio_input,
        outputs=dry_playback,
    )

demo.launch(server_port=7869, share=True)
