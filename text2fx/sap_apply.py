from pathlib import Path
from typing import Union, List, Optional, Tuple
import torch

from audiotools import AudioSignal

import text2fx.core as tc
from text2fx.sap_main import text2fx
from text2fx.constants import SAMPLE_RATE, DEVICE

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


def main(audio_path: Union[str, Path, AudioSignal], 
         fx_chain: List[str], 
         text_target: str, 
         learning_rate: float = 0.003,
         params_init_type: str = 'curriculum',
         roll_amt: Optional[int] = 1000,
         n_iters: int = 600,
         criterion: str = 'cosine-sim',
         model: str = 'ms_clap',
         pls_normalize:bool = True) -> Tuple[AudioSignal, torch.Tensor, dict]:

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
        pls_normalize=pls_normalize
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
