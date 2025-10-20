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

SCRAP_DIR = "experiments/tmp/"
# -------------------------------------------------------------------------
# Environment and Gradio patch
# -------------------------------------------------------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "true"
_old_json_schema_to_python_type = gu._json_schema_to_python_type
_old_get_type = gu.get_type
def safe_json_schema_to_python_type(schema, defs=None):
    if isinstance(schema, bool):
        return "object" if schema else "never"
    try: return _old_json_schema_to_python_type(schema, defs)
    except Exception: return "object"
def safe_get_type(schema):
    if isinstance(schema, bool):
        return "object" if schema else "never"
    try: return _old_get_type(schema)
    except Exception: return "object"
gu._json_schema_to_python_type = safe_json_schema_to_python_type
gu.get_type = safe_get_type

# -------------------------------------------------------------------------
# Globals and helpers
# -------------------------------------------------------------------------
FX_LIST =["eq"]# ["eq", "reverb"]
CHANNEL = create_channel(FX_LIST)

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

def scale_params_intensity(params_dict, alpha: float):
    scaled = {}
    def is_intensity_key(k): return any(tag in k.lower() for tag in
        ["gain","mix","drive","amount","depth","ratio","makeup","level"])
    def is_threshold_key(k): return "threshold" in k.lower()
    def is_structural_key(k): return any(tag in k.lower() for tag in
        ["freq","frequency","q","attack","release","decay","predelay","pre_delay","time","delay"])
    for fx, params in params_dict.items():
        fx_out = {}
        for k, v in params.items():
            if isinstance(v, torch.Tensor): v = v.item()
            if isinstance(v, list) and len(v)==1: v=v[0]
            if not isinstance(v,(int,float)): fx_out[k]=v; continue
            if is_structural_key(k): fx_out[k]=v
            elif is_threshold_key(k): fx_out[k]=v*alpha
            elif is_intensity_key(k): fx_out[k]=v*alpha
            else: fx_out[k]=v
        scaled[fx]=fx_out
    return scaled

def build_param_spec(channel):
    specs=[]
    for m in channel.modules:
        mod_name=type(m).__name__
        for pname,(pmin,pmax) in m.param_ranges.items():
            specs.append((mod_name,pname,float(pmin),float(pmax),(pmin+pmax)/2))
    return specs

def params_to_values(params_dict,specs):
    vals=[]
    for mod,pn,pmin,pmax,default in specs:
        val=params_dict.get(mod,{}).get(pn,default)
        if isinstance(val,torch.Tensor): val=val.item()
        if isinstance(val,list) and len(val)==1: val=val[0]
        vals.append(float(val))
    return vals

def values_to_params(values,specs):
    out={}
    for (mod,pn,_,_,_),val in zip(specs,values):
        out.setdefault(mod,{})[pn]=float(val)
    return out

# -------------------------------------------------------------------------
# Core functions
# -------------------------------------------------------------------------
def run_text2fx(audio_file, text_prompt, specs):
    out_sig, _, out_params_dict = t2fx(
        audio_file, FX_LIST, [text_prompt],
        n_iters=50,  # 150reduce for speed
        params_init_type="curriculum",
        criterion="cosine-sim", roll_amt=3000, pls_normalize=True,
    )
    params = flatten_single_item_lists(detensor_dict(out_params_dict))
    sr = out_sig.sample_rate
    y = out_sig.samples[0,0].cpu().numpy()
    spec_path = make_spectrogram(y,sr,"Spectrogram (α=1.0)","spec_opt.png")
    vals = params_to_values(params,specs)
    return (sr,y), json.dumps(params,indent=2), params, spec_path, *vals


def update_alpha(audio_file,alpha,params_dict,specs):
    if params_dict is None:
        return None,"Run text2fx first.",None,None,*[0]*len(specs)
    scaled = flatten_single_item_lists(scale_params_intensity(params_dict,alpha))
    input_sig=preprocess_audio(audio_file)
    out_sig=apply_fx_to_sig(input_sig,scaled)
    sr=out_sig.sample_rate
    y=out_sig.samples[0,0].cpu().numpy()
    spec_path=make_spectrogram(y,sr,f"Spectrogram (α={alpha:.2f})","spec_alpha.png")
    vals=params_to_values(scaled,specs)
    return (sr,y),json.dumps(scaled,indent=2),scaled,spec_path,*vals

def apply_manual(audio_file, *slider_values_and_specs):
    # The last value in that tuple is actually specs (your SPEC_ORDER)
    *slider_values, specs = slider_values_and_specs

    manual_params = values_to_params(slider_values, specs)
    input_sig = preprocess_audio(audio_file)
    out_sig = apply_fx_to_sig(input_sig, manual_params)
    sr = out_sig.sample_rate
    y = out_sig.samples[0, 0].cpu().numpy()
    spec_path = make_spectrogram(y, sr, "Spectrogram (Manual)", "spec_manual.png")
    return (sr, y), json.dumps(manual_params, indent=2), spec_path
# -------------------------------------------------------------------------
# Gradio UI
# -------------------------------------------------------------------------
with gr.Blocks(title="Text2FX DAW Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🎧 Text2FX — Semantic, Scalable, and Manual FX Control")

    SPECS = build_param_spec(CHANNEL)

    state_params_original = gr.State()   # baseline optimized params (frozen)
    state_params_current = gr.State()    # live changing params

    state_specs = gr.State(SPECS)

    with gr.Row():
        audio_in = gr.Audio(sources=["upload"], type="filepath", label="🎵 Input Audio")
        text_in = gr.Textbox(value="warm and intimate", label="Text Prompt")
        run_btn = gr.Button("🧠 Optimize (α=1.0)")

    with gr.Row():
        output_audio = gr.Audio(label="Processed Output")
        fx_json = gr.Textbox(label="FX Parameters (JSON)", lines=10)
        spec_img = gr.Image(label="Spectrogram")

    alpha = gr.Slider(0,1,1.0,0.05,label="Effect Intensity (α)")

    # Low-level sliders
    with gr.Accordion("🎚️ Low-Level Parameters", open=False):
        sliders=[]
        for mod,pn,pmin,pmax,default in SPECS:
            sliders.append(gr.Slider(pmin,pmax,default,label=f"{mod}.{pn}"))

    apply_btn = gr.Button("🎛️ Apply Manual Parameters")

    # Wiring
    run_btn.click(
    fn=run_text2fx,
    inputs=[audio_in, text_in, state_specs],
    outputs=[output_audio, fx_json, state_params_original, spec_img, *sliders],
    ).then(
        fn=lambda p: p,  # duplicate to current state
        inputs=state_params_original,
        outputs=state_params_current,
    )

    alpha.change(
        fn=update_alpha,
        inputs=[audio_in, alpha, state_params_original, state_specs],
        outputs=[output_audio, fx_json, state_params_current, spec_img, *sliders],
    )

    apply_btn.click(
        fn=apply_manual,
        inputs=[audio_in, *sliders, state_specs],
        outputs=[output_audio, fx_json, spec_img],
    )

demo.launch(server_port=7869, share=True)