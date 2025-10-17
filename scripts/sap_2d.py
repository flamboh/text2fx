import text2fx
from text2fx.sap_apply import main as t2fx
import torch
from text2fx.sap_main import get_model
from text2fx.core import preprocess_audio, create_channel

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

def build_semantic_space(clap, device="cuda"):
    # Define poles
    bright = clap.get_text_embeddings(["a bright sound"]).to(device)
    dark = clap.get_text_embeddings(["a dark sound"]).to(device)
    metallic = clap.get_text_embeddings(["a full sound"]).to(device)
    wooden = clap.get_text_embeddings(["a hollow sound"]).to(device)

    # Normalize embeddings
    for e in [bright, dark, metallic, wooden]:
        e /= e.norm(dim=-1, keepdim=True)

    # Compute axis vectors
    axis_x = bright - dark
    axis_y = metallic - wooden
    e_center = (bright + dark + metallic + wooden) / 4

    def z(x, y):
        """Returns embedding at coordinate (x, y)"""
        vec = e_center + x * axis_x + y * axis_y
        return vec / vec.norm(dim=-1, keepdim=True)

    return z


def transform_with_semantics(
    input_path: str,
    x: float,
    y: float,
    alpha: float = 1.0,
    use_what_fx=["eq"],
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clap = get_model("ms_clap")#, device=device)
    z = build_semantic_space(clap, device)
    channel=create_channel(use_what_fx)
    print(channel)

    # Compute input embedding
    sig = preprocess_audio(input_path).to(device)
    audio_emb = clap.get_audio_embeddings(sig).detach()
    audio_emb = torch.nn.functional.normalize(audio_emb, dim=-1)

    # Compute semantic target embedding
    embedding_target = (1 - alpha) * audio_emb + alpha * z(x, y)
    embedding_target = torch.nn.functional.normalize(embedding_target, dim=-1)

    # Run text2fx using that embedding target
    # # ====== Text2FX it! ===========
    out_sig, out_params, out_params_dict = t2fx(input_path, 
                                            use_what_fx, 
                                            ["semantic transform"],
                                             n_iters=400, #usually 400,600
                                             params_init_type="curriculum",
                                             criterion= "cosine-sim",  
                                            roll_amt = 3000,
                                            pls_normalize=True,
                                               custom_embedding_target=embedding_target)

    return out_sig, out_params, out_params_dict

   

import gradio as gr

def explore(x, y):
    out_sig, _, _ = transform_with_semantics(
        input_path="/home/annie/research/text2fx/assets/salsa_piano.wav",
        x=x,
        y=y,
        alpha=0.8
    )
    audio = out_sig.samples[0][0].cpu().numpy()
    return (out_sig.sample_rate, audio)

gr.Interface(
    fn=explore,
    inputs=[
        gr.Slider(-1, 1, step=0.1, label="X Axis: Dark ↔ Bright"),
        gr.Slider(-1, 1, step=0.1, label="Y Axis: hollow ↔ full")
    ],
    outputs=gr.Audio(label="Transformed Audio"),
).launch(server_port=7869, share=True)

