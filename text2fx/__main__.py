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

from text2fx.core import Channel, AbstractCLAPWrapper, Distortion, create_save_dir, preprocess_audio, detensor_dict, slugify
from text2fx.constants import RUNS_DIR, SAMPLE_RATE, DEVICE

"""
EX CLI USAGE
python -m text2fx --input_audio "assets/speech_examples/VCTK_p225_001_mic1.flac"\
                 --text "this sound is happy" \
                 --criterion "cosine-sim" \
                 --n_iters 600 \
                 --lr 0.01 
                 --params_init_type "zeros"
                 --
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
    log_audio_every_n: int = 25, 
    lr: float = 1e-2, 
    n_iters: int = 600,
    criterion: str = "standard", 
    save_dir: str = None, # figure out a save path automatically,
    params_init_type: str = "random",
    # seed_i: int = 0,
    roll_amt: int = None,
    detailed_log: bool = False,
    export_audio: bool = False,
    log_tensorboard: bool = False,
    pls_normalize: bool = True,
):

    ##### ============ Set up!!!!! ==========
    clap = get_model(model_name) #default to ms_clap, though laion_clap might be better....
    print(f"Criterion: {criterion}")

    sig = preprocess_audio(sig_in).to(device) #preprocessing initial sample (entire sample)
    # sig = preprocess_audio(sig_in, 3).to(device) #for fast version, taking 3s excerpt

    # a save dir for our goods
    if log_tensorboard or export_audio or detailed_log:
        if not save_dir:
            save_dir = create_save_dir(f'{text}_{lr}_{criterion}', RUNS_DIR)
        else:
            save_dir = Path(save_dir)
            # save_dir = create_save_dir(f'{text}', Path(save_dir))
            save_dir.mkdir(exist_ok=True, parents=True)

    # Tensorboard writer
    if log_tensorboard:
        writer_dir = save_dir / "logs"
        writer_dir.mkdir(exist_ok=True)
        writer = SummaryWriter(writer_dir) #SummaryWriter is tensorboard writer
    else:
        writer = False

    # FX parameter initialization
    if params_init_type == 'zeros':
        params = torch.nn.parameter.Parameter(torch.zeros(sig.batch_size, channel.num_params).to(device))
    elif params_init_type == 'random':
        params = torch.nn.parameter.Parameter(torch.randn(sig.batch_size, channel.num_params).to(device))
    elif params_init_type == 'super_random':
        params = torch.nn.parameter.Parameter((torch.randn(sig.batch_size, channel.num_params).to(device) * 8))
    elif params_init_type == 'curriculum':
        params = torch.nn.parameter.Parameter(0.01 * torch.randn(sig.batch_size, channel.num_params).to(device))
    else:
        raise ValueError(f"Unknown params_init_type: {params_init_type}")
    

    params.requires_grad=True


    # =====LOGGING=====
    if log_tensorboard or export_audio or detailed_log:
        log_file = save_dir / f"experiment_log.txt"
        with open(log_file, "w") as log:
            log.write(f"Model: {model_name}\n")
            log.write(f"Channel: {channel.modules}\n")
            log.write(f"Learning Rate: {lr}\n")
            log.write(f"Number of Iterations: {n_iters}\n")
            log.write(f"Criterion: {criterion}\n")
            log.write(f"Params Initialization Type: {params_init_type}\n")
            log.write(f"Starting Params Values: {params.data.cpu().numpy()}\n")
            log.write(f"Starting Params Values (post sigmoid): {torch.sigmoid(params).data.cpu().numpy()}\n")
            log.write(f"Custom roll?: {roll_amt}\n")
            log.write("="*40 + "\n")

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

    if writer:
        writer.add_audio("input", sig.samples[0][0], 0, sample_rate=sig.sample_rate)
        writer.add_audio("effected", init_sig.samples[0][0], 0, sample_rate=init_sig.sample_rate)
    # sig_in.clone().cpu().write(save_dir / 'input.wav')
    if export_audio: #starting audio
        if sig.batch_size == 1:
            init_sig_path = Path(init_sig.path_to_file)
            sig.clone().detach().cpu().write(save_dir / f'{init_sig_path.stem}_input.wav')
            init_sig.detach().cpu().write(save_dir / f'{init_sig_path.stem}_starting.wav')
        else:
            for i, s in enumerate(init_sig):
                sig[i].clone().detach().cpu().write(save_dir / f'{init_sig.path_to_file[i].stem}_input.wav')
                init_sig[i].detach().cpu().write(save_dir / f'{init_sig.path_to_file[i].stem}_starting.wav')

    # ======= TEXT PREP========
    if isinstance(text, str):
        text = [text]
    assert len(text) == sig.batch_size or len(text) == 1
    if len(text) < sig.batch_size:
        text = text * sig.batch_size

    # OLD Preprocess text
    # # text_processed = [f"this sound is {t}" for t in text]
    # embedding_target = clap.get_text_embeddings(text_processed).detach()
    # 10/3/2025: Text is already a list with length = batch_size
    templates = ["this sound is {}", "a recording of {}", "audio that embodies {}"]

    # Collect embeddings for each text in batch, across all templates
    all_embeds = []
    for t in text:  # loop over each batch element
        embeds = []
        for temp in templates:
            sentence = temp.format(t)
            embeds.append(clap.get_text_embeddings([sentence]))
        embeds = torch.stack(embeds).mean(0)  # average over templates
        all_embeds.append(embeds)
    text_emb = torch.cat(all_embeds, dim=0).detach()     # Shape: (batch_size, embedding_dim)
    text_emb = torch.nn.functional.normalize(text_emb, dim=-1)

    ### ==== INPUT ADUIO EMB =====
    audio_in_emb = clap.get_audio_embeddings(sig.to(device)).detach()
    audio_in_emb = torch.nn.functional.normalize(audio_in_emb, dim=-1)

    # ====== TARGET EMBED blending ======
    alpha = 0.8  # slider: how strong is the text influence
    print(f"text: {text}, alpha = {alpha}")
    embedding_target = (1 - alpha) * audio_in_emb + alpha * text_emb


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
        # embedding_neg = torch.nn.functional.normalize(embedding_neg, dim=-1)
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

        if log_tensorboard or export_audio or detailed_log:
            with open(log_file, "a") as log:
                log.write(f"Iteration {n}: roll_amount: {roll_amount.cpu().numpy()}\n")

        for i in range(sig_roll.batch_size):
            rolled = torch.roll(sig_roll.samples[i], shifts=roll_amount[i].item(), dims=-1)
            sig_roll.samples[i:i+1] = rolled

        # Curriculum annealing (always apply, but factor=1.0 if not curriculum)
        if params_init_type == 'curriculum':
            warmup_iters = 200  # number of steps before full range is open
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

        # elif criterion == "cosine-sim": # cosine_sim loss aims to maximize the cosine similarity between similar items, normalized
        #     batch_loss = 1 - torch.cosine_similarity(embedding_effected, embedding_target, dim=-1)
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

        if writer: 
            writer.add_scalar("loss", loss.item(), n)

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
            final_loss = batch_loss.detach().cpu().numpy()
            loss_change = final_loss - initial_loss  # Compute change in loss
            print(f"Initial loss: {initial_loss} // Final loss: {final_loss}")
            print(f"Change in loss from iteration 0 to {n_iters - 1}: {loss_change}")
    
        if log_tensorboard:
            if n % log_audio_every_n == 0:
                # Save audio
                signal_effected.detach().cpu().ensure_max_of_audio().write_audio_to_tb("effected", writer, n)
                if writer:
                    writer.add_audio("effected", signal_effected.clone().ensure_max_of_audio().samples[0][0], n, sample_rate=signal_effected.sample_rate)
        
        # detailed logging, log params + signal every 100 iters
        if detailed_log:
            init_sig_path = Path(init_sig.path_to_file)
            detailed_dir = Path(save_dir) / 'detailed_logs'
            detailed_dir.mkdir(parents=True, exist_ok=True)
            json_log_path = detailed_dir / "params_log.json"  # Path to save the JSON file
            sig.clone().detach().cpu().write(detailed_dir / f'{init_sig_path.stem}__ref.wav')
            if n % 100 == 0 or n==n_iters-1:
                params_i = params.detach().cpu()
                out_params_dict = channel.save_params_to_dict(params.detach().cpu())
                print(out_params_dict)
                with open(json_log_path, "a") as json_log_file:
                    json.dump({"iteration": n, "params": detensor_dict(out_params_dict)}, json_log_file)
                    json_log_file.write("\n")  # For better readability in the file
                    json.dump({"iteration": n, "raw_params": params_i.tolist()}, json_log_file)
                    json_log_file.write("\n")  # For better readability in the file
                
                signal_effected.detach().cpu().ensure_max_of_audio().write(detailed_dir / f'{init_sig_path.stem}_{n}.wav')
                # signal_effected_original.detach().cpu().ensure_max_of_audio().write(detailed_dir / f'{init_sig_path.stem}_{n}.wav')

    if log_tensorboard or export_audio or detailed_log:
        with open(log_file, "a") as log:
            log.write(f"ENDING Params Values: {params.data.cpu().numpy()}\n")
    
    # min_loss_index = int(np.argmin(final_losses)) # used for comparing across multiple runs

    # Play final signal with optimized effects parameters
    clean_sig = preprocess_audio(sig_in).to(device) #taking full input sample 
    out_sig = channel(clean_sig.clone(), torch.sigmoid(params)).clone().detach().cpu()
    out_sig = preprocess_audio(out_sig)  
    out_params = params.detach().cpu() #optimized output FXparams
    out_params_dict = channel.save_params_to_dict(out_params) #mapping back to FX ranges

    if export_audio:
        if sig.batch_size == 1:
            out_sig.detach().cpu().write(save_dir / f'{init_sig_path.stem}_final.wav')
            # out_sig.clone().detach().cpu().write(save_dir / f'{init_sig_path.stem}_final.wav')
        else:
            for i, s in enumerate(out_sig):
                i_init_sig_path = Path(init_sig.path_to_file[i])
                out_sig[i].detach().cpu().write(save_dir / f'{i_init_sig_path.stem}_final.wav')

    # out_sig.write(save_dir / "final.wav")

    if writer:
        writer.add_audio("final", out_sig.samples[0][0], n_iters, sample_rate=out_sig.sample_rate)
        writer.close()

    return out_sig, out_params, out_params_dict


# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()

#     # parser.add_argument("--input_audio", type=int, default=5, help="index of example audio file")
#     parser.add_argument("--model_name", type=str, help="choose either 'laion_clap' or 'ms_clap'")
#     parser.add_argument("--input_audio", type=str, help="path to input audio file")
#     parser.add_argument("--text", type=str, help="text prompt for the effect")
#     parser.add_argument("--criterion", type=str, default="cosine-sim", help="criterion to use for optimization")
#     parser.add_argument("--n_iters", type=int, default=600, help="number of iterations to optimize for")
#     parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for optimization")
#     parser.add_argument("--save_dir", type=str, default=None, help="path to export audio file")
#     parser.add_argument("--params_init_type", type=str, default='zeros', help="enter params init type")
#     parser.add_argument("--roll_amt", type=int, default=None, help="range of # of samples for rolling action")
#     parser.add_argument("--export_audio", type=bool, default=False, help="export audio?")
#     parser.add_argument("--log_tensorboard", type=bool, default=False, help="log tensorboard?")


#     args = parser.parse_args()

#     # channel = Channel(dasp_pytorch.ParametricEQ(sample_rate=SAMPLE_RATE))

#     text2fx(
#         model_name=args.model_name, 
#         sig=AudioSignal(args.input_audio), 
#         text=args.text, 
#         channel=args.channel,
#         criterion=args.criterion, 
#         save_dir=args.save_dir,
#         params_init_type=args.params_init_type,
#         roll_amt=args.roll_amt,
#         export_audio=args.export_audio,
#         log_tensorboard=args.log_tensorboard
#     )
