import gradio as gr
from text2fx.sap_apply import main as t2fx
from text2fx.core import detensor_dict, preprocess_audio
import os
import gradio_client.utils as gu
os.environ["TOKENIZERS_PARALLELISM"] = "true"

"""
Todo:
Script currently is super rough
1. you should be able to listen to original audio too
2. it should immediately optimize per text, which returns the text-based FX params at like 100% intensity
3. then the alpha slider just scales those params down in intensity, which automatically gets applied ot the input to update the output
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

AUDIO_PATH = "/home/annie/research/text2fx/assets/salsa_piano.wav"
FX_LIST = ["eq", "reverb"]  # whatever you pass to channel/create_channel
def scale_params_intensity(params_dict: dict, alpha: float) -> dict:
    """
    Scale only 'intensity-like' parameters by alpha.
    Leaves frequencies / Q / time constants unchanged.
    Handles compressor threshold (more negative = stronger) specially.
    """
    scaled = {}

    # helper: decide if a key should be scaled as intensity
    def is_intensity_key(key: str) -> bool:
        k = key.lower()
        return any(tag in k for tag in [
            "gain", "mix", "drive", "amount", "depth", "ratio", "makeup", "level"
        ])

    def is_threshold_key(key: str) -> bool:
        return "threshold" in key.lower()

    def is_structural_key(key: str) -> bool:
        # frequencies, Q, and time constants should not move with alpha
        k = key.lower()
        return any(tag in k for tag in [
            "freq", "frequency", "q", "attack", "release", "decay", "predelay", "pre_delay", "time", "delay"
        ])

    for fx, params in params_dict.items():
        fx_out = {}
        for k, v in params.items():
            if not isinstance(v, (int, float)):
                fx_out[k] = v
                continue

            if is_structural_key(k):
                fx_out[k] = v  # do NOT scale
            elif is_threshold_key(k):
                # thresholds are usually negative; scaling 'intensity' means pushing
                # farther from 0 as alpha→1 and toward 0 as alpha→0.
                # simple safe rule: lerp toward 0 dB when alpha decreases.
                fx_out[k] = v * alpha + 0.0 * (1 - alpha)
            elif is_intensity_key(k):
                fx_out[k] = v * alpha
            else:
                # unknown numeric key: be conservative and leave it
                fx_out[k] = v

        scaled[fx] = fx_out
    return scaled

def run_text2fx_with_alpha(text_prompt: str, alpha: float):
    # 1) run text2fx normally (optimizes DSP for the text intent)
    out_sig, out_params, out_params_dict = t2fx(
        AUDIO_PATH,
        FX_LIST,
        [text_prompt],
        n_iters=100,
        params_init_type="curriculum",
        criterion="cosine-sim",
        roll_amt=3000,
        pls_normalize=True,
    )

    # out_params_dict is tensors → to python numbers
    params_dict = detensor_dict(out_params_dict)

    # 2) selectively scale intensity knobs by alpha
    scaled = scale_params_intensity(params_dict, alpha)

    # --------- audio output strategy ----------
    # (A) if you can map dict → channel tensor and re-render: do that here.
    #     channel = create_channel(FX_LIST); params_tensor = channel.params_from_dict(scaled)
    #     dry = preprocess_audio(AUDIO_PATH).to(out_sig.device)
    #     wet = channel(dry, params_tensor)
    #     final = wet  # because params already include alpha intensity
    #
    # (B) fallback: dry/wet crossfade (keeps sound responsive even without dict→tensor)
    #     This approximates “intensity” at audio level:
    dry_sig = preprocess_audio(AUDIO_PATH).to(out_sig.device)
    dry_np = dry_sig.samples[0, 0].cpu().numpy()
    wet_np = out_sig.samples[0, 0].cpu().numpy()
    final_np = (alpha * wet_np) + ((1.0 - alpha) * dry_np)

    # package results
    sr = out_sig.sample_rate
    fx_summary = "\n".join(f"{fx}: {scaled[fx]}" for fx in scaled)
    return (sr, final_np), fx_summary

# simple UI: prompt + alpha
demo = gr.Interface(
    fn=run_text2fx_with_alpha,
    inputs=[
        gr.Textbox(value="warm and intimate", label="Text Prompt"),
        gr.Slider(0, 1, value=1.0, step=0.05, label="Alpha (effect intensity)"),
    ],
    outputs=[
        gr.Audio(label="Output (dry/wet alpha)"),
        gr.Textbox(label="Scaled FX Parameters (alpha-applied)"),
    ],
    title="Text2FX α-intensity",
    description="Runs text2fx for the prompt, then scales only intensity-like parameters by α. Frequencies/Q/times are left untouched.",
)
demo.launch(server_port=7869, share=True)
