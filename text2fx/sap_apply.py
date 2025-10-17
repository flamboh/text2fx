from pathlib import Path
from typing import Union, List, Optional, Tuple
import torch

from audiotools import AudioSignal

import text2fx.core as tc
from text2fx.sap_main import text2fx, get_model
from text2fx.constants import SAMPLE_RATE, DEVICE
from text2fx.core import preprocess_audio, detensor_dict, slugify, set_seed


"""
SAP EDITION
Script to process a single audio file with a given FX chain to match a description.
Optional arguments include learning rate, number of steps, loss type, parameter initialization, and augmentation params.
The script saves:
- A dictionary of optimized effect controls as JSON (specified by export_dir).
- Exported optimized audio file (saved to export_dir).

Example Call:
python -m text2fx.apply assets/multistem_examples/10s/bass.wav eq 'warm like a hug' \
    --export_dir experiments/prod_final \
    --learning_rate 0.01 \
    --params_init_type random \
    --roll_amt 10000 \
    --n_iters 400 \
    --criterion cosine-sim \
    --model ms_clap \
    --detailed_log

    
case 1 (sparse): single audio file, single text_target
python -m text2fx.apply assets/multistem_examples/10s/guitar.wav eq reverb compression 'cold and dark' \
    --export_dir experiments/2025-01-28/guitar_multifx_2 \
    --params_init_type random \
    --n_iters 200 
"""
def build_semantic_space(model, device="cuda"):
    """Builds a 2D semantic space for sound timbre exploration using CLAP embeddings."""
    # Define poles
    bright = model.get_text_embeddings(["a bright sound"]).to(device)
    dark = model.get_text_embeddings(["a dark sound"]).to(device)
    metallic = model.get_text_embeddings(["a metallic sound"]).to(device)
    wooden = model.get_text_embeddings(["a wooden sound"]).to(device)

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
    model_name: str = "ms_clap",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load model
    clap = get_model(model_name, device)
    z = build_semantic_space(clap, device)

    # Compute input embedding
    sig = preprocess_audio(input_path).to(device)
    audio_emb = clap.get_audio_embeddings(sig).detach()
    audio_emb = torch.nn.functional.normalize(audio_emb, dim=-1)

    # Compute semantic target embedding
    embedding_target = (1 - alpha) * audio_emb + alpha * z(x, y)
    embedding_target = torch.nn.functional.normalize(embedding_target, dim=-1)


    # Run text2fx using that embedding target
    # # ====== Text2FX it! ===========
    out_sig, out_params, out_params_dict = text2fx(audio_path, 
                                            FX_chain, 
                                            target_text,
                                             n_iters=400, #usually 600
                                             params_init_type="curriculum",
                                             criterion= "cosine-sim",  
                                            roll_amt = 3000,
                                            pls_normalize=True,
                                               custom_embedding_target=embedding_target)

    return out_sig, out_params, out_params_dict

def main(audio_path: Union[str, Path, AudioSignal], 
         fx_chain: List[str], 
         text_target: str, 
         learning_rate: float = 0.003,
         params_init_type: str = 'curriculum',
         roll_amt: Optional[int] = 1000,
         n_iters: int = 600,
         criterion: str = 'cosine-sim',
         model: str = 'ms_clap',
         pls_normalize:bool = True,
         custom_embedding_target:torch.Tensor = None,) -> Tuple[AudioSignal, torch.Tensor, dict]:

    # Preprocess full audio from path, return AudioSignal
    print('text2fx on full sig')
    # in_sig = tc.preprocess_audio(audio_path).to(DEVICE)

    # print('text2fx on 3s salient_excerpt')
    in_sig = tc.preprocess_audio(audio_path, salient_excerpt_duration=3).to(DEVICE)


    # Create FX channel
    fx_channel = tc.create_channel(fx_chain)
    print(f'2. created channel from {fx_chain} ... {fx_channel.modules}')

        
    signal_effected, out_params, out_params_dict = text2fx(
        model_name=model, 
        sig_in=audio_path, 
        text=text_target, 
        channel=fx_channel,
        criterion=criterion, 
        params_init_type=params_init_type,
        lr=learning_rate,
        n_iters=n_iters,
        roll_amt=roll_amt,
        pls_normalize=pls_normalize,
        custom_embedding_target=custom_embedding_target,
    )

    return signal_effected, out_params, out_params_dict


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process an audio file with a given FX chain to match a description.")
    
    parser.add_argument("audio_path", type=str, help="Path to the audio file.")
    parser.add_argument("fx_chain", nargs="+", help="List of FX to apply.")
    parser.add_argument("text_target", type=str, default='warm', help="Text description to match.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimization.")
    parser.add_argument("--params_init_type", type=str, default='random', choices=['random', 'default'], help="Parameter initialization type.")
    parser.add_argument("--roll_amt", type=int, default=None, help="Amount to roll.")
    parser.add_argument("--n_iters", type=int, default=600, help="Number of optimization iterations.")
    parser.add_argument("--criterion", type=str, default='cosine-sim', help="Optimization criterion.")
    parser.add_argument("--model", type=str, default='ms_clap', help="Model name.")


    args = parser.parse_args()

    main(args.audio_path, 
         args.fx_chain, 
         args.text_target,
         args.learning_rate, 
         args.params_init_type, 
         args.roll_amt,
         args.n_iters, 
         args.criterion, 
         args.model,)
