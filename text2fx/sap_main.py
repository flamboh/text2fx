from pathlib import Path
from tqdm import tqdm

import torch
import numpy as np
import audiotools as at
import dasp_pytorch
from audiotools import AudioSignal

from typing import Union, List

from torch.utils.tensorboard import SummaryWriter
import json
# from msclap import CLAP

from text2fx.core import Channel, AbstractCLAPWrapper, Distortion, create_save_dir, preprocess_audio, detensor_dict, slugify, set_seed
from text2fx.constants import RUNS_DIR, SAMPLE_RATE, DEVICE


"""
new text2fx main function for building a 2D semantic space
"""
device = DEVICE #torch.device("cuda:0") if torch.cuda.is_available() else "cpu"


def get_model(model_choice: str):
    if model_choice=="laion_clap":
        from text2fx.laionclap import LAIONCLAPWrapper
        model = LAIONCLAPWrapper()
    elif model_choice == "ms_clap":
        from text2fx.msclap import MSCLAPWrapper
        model = MSCLAPWrapper()
    else:
        raise ValueError('choose a model1!!!!!!')
    return model


def rms_norm(sig, target_rms=0.1, eps=1e-8):
    s = sig.clone()
    rms = (s.samples.pow(2).mean(dim=(-1,-2), keepdim=True) + eps).sqrt()
    s.samples = s.samples * (target_rms / (rms + eps))
    return s


def clip_directional_loss(
        a1: torch.Tensor, 
        a2: torch.Tensor, 
        b1: torch.Tensor, 
        b2: torch.Tensor
    ):
        a_dir = a1 - a2
        a_dir /= a_dir.clone().norm(dim=-1, keepdim=True)

        b_dir = b1 - b2
        b_dir /= b_dir.clone().norm(dim=-1, keepdim=True)

        loss = 1 - torch.cosine_similarity(a_dir, b_dir, dim=-1)
        return loss

def get_default_channel():
    return Channel(
        dasp_pytorch.ParametricEQ(sample_rate=SAMPLE_RATE),
        # dasp_pytorch.Compressor(sample_rate=SAMPLE_RATE),
        # dasp_pytorch.Gain(sample_rate=SAMPLE_RATE),
        # dasp_pytorch.NoiseShapedReverb(sample_rate=SAMPLE_RATE),
        # Distortion(sample_rate=SAMPLE_RATE),
    )

def multi_res_stft_loss(x, y):
    """
    Multi-resolution STFT loss between two signals.
    Expects x, y shaped (B, C, T) or (B, T).
    """
    # Handle both mono/stereo
    if x.ndim == 3:  # (B, C, T)
        x = x.mean(dim=1)  # mix down channels
    if y.ndim == 3:
        y = y.mean(dim=1)
    losses = []
    for n_fft, hop in [(512, 128), (1024, 256), (2048, 512)]:
        X = torch.view_as_real(torch.stft(x, n_fft=n_fft, hop_length=hop, return_complex=True))
        Y = torch.view_as_real(torch.stft(y, n_fft=n_fft, hop_length=hop, return_complex=True))
        losses.append(torch.nn.functional.l1_loss(X, Y))
    return sum(losses) / len(losses)


def text2fx(
    model_name: str,
    sig_in: Union[torch.Tensor, str, Path, np.ndarray, AudioSignal], 
    text: Union[str, List[str]],   
    channel: Channel,
    device: str = "cuda" if torch.cuda.is_available() else "cpu", 
    optimizer_type: str = "adam",
    lr: float = 1e-2, 
    n_iters: int = 600,
    criterion: str = "standard", 
    params_init_type: str = "random",
    seed_i: int = 0,
    roll_amt: int = None,
    pls_normalize: bool = True,
    custom_embedding_target: torch.Tensor = None, #for custom embedding target
):

    ##### ============ Set up!!!!! ==========
    clap = get_model(model_name) #default to ms_clap, though laion_clap might be better....
    print(f"Criterion: {criterion}")

    sig = preprocess_audio(sig_in).to(device) #preprocessing initial sample (entire sample)
    # sig = preprocess_audio(sig_in, 5).to(device) #for fast version, taking 3s excerpt

    # FX parameter initialization
    set_seed(seed_i, deterministic=False)

    if params_init_type == 'zeros':
        params = torch.nn.parameter.Parameter(torch.zeros(sig.batch_size, channel.num_params).to(device))
    elif params_init_type == 'random':
        params = torch.nn.parameter.Parameter(torch.randn(sig.batch_size, channel.num_params).to(device))
    elif params_init_type == 'curriculum':
        params = torch.nn.parameter.Parameter(0.01 * torch.randn(sig.batch_size, channel.num_params).to(device))
    else:
        raise ValueError(f"Unknown params_init_type: {params_init_type}")
    

    params.requires_grad=True

    # setting up the optimizer
    if optimizer_type.lower() == "adam":
        optimizer = torch.optim.Adam([params], lr=lr)
    elif optimizer_type.lower() == "sgd":
        optimizer = torch.optim.SGD([params], lr=lr, momentum=0.9)
    elif optimizer_type.lower() == "cma_es": #TODO: implement CMA-ES
        raise NotImplementedError("CMA-ES optimizer not implemented yet")
    else:
        raise ValueError(f"Unknown optimizer_type: {optimizer_type}")


    # ==================== INITIAL SIGNAL ====================
    init_sig = channel(sig.clone().to(device), torch.sigmoid(params))


    # ======= TEXT PREP========
    if isinstance(text, str):
        text = [text]
    assert len(text) == sig.batch_size or len(text) == 1
    if len(text) < sig.batch_size:
        text = text * sig.batch_size

    # Computing embedding target from text if not explicitly provided (as in custom embedding target case)
    if custom_embedding_target is None:
        print("No embedding target provided, computing from text...")
        templates = ["this sound is {}", "a recording of {}", "audio that embodies {}"]
        # Collect embeddings for each text in batch, across all templates
        all_embeds = []
        for t in text:  # Loop over each batch element
            embeds = []
            for temp in templates:
                sentence = temp.format(t)
                embeds.append(clap.get_text_embeddings([sentence]))
            embeds = torch.stack(embeds).mean(0)  # average over templates
            all_embeds.append(embeds)
        embedding_target = torch.cat(all_embeds, dim=0).detach()     # Shape: (batch_size, embedding_dim)
        embedding_target = torch.nn.functional.normalize(embedding_target, dim=-1)
    else:
        print("Using provided custom embedding target...")
        embedding_target = torch.nn.functional.normalize(custom_embedding_target.to(device), dim=-1)


    ### ==== INPUT ADUIO EMB =====
    audio_in_emb = clap.get_audio_embeddings(sig.to(device)).detach()
    audio_in_emb = torch.nn.functional.normalize(audio_in_emb, dim=-1)

    # ====== NEGATIVE ANCHOR EMBED ======
    if criterion == "cosine-sim":
        neg_templates = ["not {}", "the opposite of {}", "definitely not {}"]
        neg_embeds = []
        for t in text:
            e = []
            for temp in neg_templates:
                e.append(clap.get_text_embeddings([temp.format(t)]))
            e = torch.stack(e).mean(0)
            neg_embeds.append(e)
        embedding_neg = torch.cat(neg_embeds, dim=0).detach()
        embedding_neg = torch.nn.functional.normalize(embedding_neg, dim=-1)
    else:
        embedding_neg = None

    print(f"Starting optimization with {sig.batch_size} samples...")
    print(f"pls_normalize: {pls_normalize}")

    # ==================== OPTIMIZATION LOOP ====================
    # Single-Instance Optimization: Optimize our parameters by matching effected audio against the target text embedding
    pbar = tqdm(range(n_iters), total=n_iters)
    for n in pbar:
        # rolling
        sig_roll = sig.clone()
        if roll_amt or roll_amt == 0:
            roll_amount = torch.randint(-roll_amt, roll_amt + 1, (sig_roll.batch_size,))
        else:
            roll_amount = torch.randint(0, sig_roll.signal_length, (sig_roll.batch_size,))

 
        for i in range(sig_roll.batch_size):
            rolled = torch.roll(sig_roll.samples[i], shifts=roll_amount[i].item(), dims=-1)
            sig_roll.samples[i:i+1] = rolled

        # Curriculum annealing (always apply, but factor=1.0 if not curriculum)
        if params_init_type == 'curriculum':
            warmup_iters = 50  # number of steps before full range is open
            # print(f"Curriculum annealing over {warmup_iters} iters")
            anneal_factor = min(1.0, n / warmup_iters)
        else:
            anneal_factor = 1.0

        scaled_params = torch.sigmoid(params.to(device)) * anneal_factor
        signal_effected = channel(sig_roll.to(device), scaled_params)

        # Get CLAP embedding for effected audio
        embedding_effected = clap.get_audio_embeddings(signal_effected) #.get_audio_embeddings takes in preprocessed audio
        embedding_effected = torch.nn.functional.normalize(embedding_effected, dim=-1)

        # Calculating Loss
        if criterion == "directional_loss":
            text_neg_processed = [f"not {t}" for t in text]
            text_anchor_emb = clap.get_text_embeddings(text_neg_processed).detach()
            batch_loss = clip_directional_loss(embedding_effected, audio_in_emb, embedding_target, text_anchor_emb)

        elif criterion == "standard": #is neg dot product loss aims to minimize the dot prod b/w dissimilar items, no direction intake
            batch_loss = - (embedding_effected * embedding_target).sum(dim=-1)

        elif criterion == "cosine-sim":
            pos_sim = torch.cosine_similarity(embedding_effected, embedding_target, dim=-1)
            neg_sim = torch.cosine_similarity(embedding_effected, embedding_neg, dim=-1)
            margin = 0.2
            batch_loss = (1 - pos_sim) + torch.relu(neg_sim - margin)

        else:
            raise ValueError(f"Criterion {criterion} not recognized")
        
        # === PERCEPTUAL LOSS?? AS A REGULARIZER?? ===
        lambda_spec = 0.05
        spec_loss = multi_res_stft_loss(signal_effected.samples, sig_roll.samples)
        loss = batch_loss.mean() + lambda_spec * spec_loss
        # loss = batch_loss.mean()


        # Optimize
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([params], max_norm=5.0)
        optimizer.step()

        pbar.set_description(f"step: {n+1}/{n_iters}, loss: {loss.item():.3f}")

        # Initialize variable to store the initial loss
        if n == 0:
            initial_loss = batch_loss.detach().cpu().numpy()

        # Saving last batch_loss and computing total loss change
        if n == n_iters - 1:
            print("Direcitonal vector from source:")
            final_loss = batch_loss.detach().cpu().numpy()
            loss_change = final_loss - initial_loss  # Compute change in loss
            print(f"Initial loss: {initial_loss} // Final loss: {final_loss}")
            print(f"Change in loss from iteration 0 to {n_iters - 1}: {loss_change}")
    
  
   
    # min_loss_index = int(np.argmin(final_losses)) # used for comparing across multiple runs

    # Play final signal with optimized effects parameters
    clean_sig = preprocess_audio(sig_in).to(device) #taking full input sample 
    out_sig = channel(clean_sig.clone(), torch.sigmoid(params)).clone().detach().cpu()
    out_sig = preprocess_audio(out_sig)  
    out_params = params.detach().cpu() #optimized output FXparams
    out_params_dict = channel.save_params_to_dict(out_params) #mapping back to FX ranges


    return out_sig, out_params, out_params_dict
