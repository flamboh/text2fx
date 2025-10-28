import os
from pathlib import Path
from typing import Union, List

import torch
import laion_clap
from audiotools import AudioSignal

from text2fx.core import AbstractCLAPWrapper, download_file
from text2fx.constants import PRETRAINED_DIR, DEVICE


class LAIONCLAPWrapper(AbstractCLAPWrapper):
    def __init__(self):
        CLAP_MODELS = [
            '630k-best.pt',                                 # Best non-fusion checkpoint, good for general audio < 10s
            '630k-audioset-best.pt',                        # Best non-fusion checkpoint, good for general audio < 10s
            '630k-fusion-best.pt',                          # Best fusion checkpoint, good for general audio of variable lengths > 10s
            '630k-audioset-fusion-best.pt',                 # Best fusion checkpoint, good for general audio of variable lengths > 10s
            'music_audioset_epoch_15_esc_90.14.pt',         # Specialized for music, best music-tagging performance
            'music_speech_epoch_15_esc_89.25.pt',           # Specialized for music and speech, near-best music-tagging performance
            'music_speech_audioset_epoch_15_esc_89.98.pt',  # For music / speech / general audio, lower music-tagging performance
        ]

        CLAP_AUDIO_MODELS = [
            'HTSAT-base',
            'HTSAT-large',
            'HTSAT-tiny',                                    # Default
            'HTSAT-tiny-win-1536',
            'PANN-6',
            'PANN-10',
            'PANN-14',
            'PANN-14-fmax-8k-20s',
            'PANN-14-fmax-18k',
            'PANN-14-tiny-transformer',
            'PANN-14-win-1536'
        ]

        self.CLAP_SAMPLE_RATE = 48_000
        CLAP_PRETRAINED_DIR = PRETRAINED_DIR / "clap"
        CLAP_DOWNLOAD_LINK = 'https://huggingface.co/lukewys/laion_clap/resolve/main/'
        CLAP_MODEL_IDX = 1
        CLAP_AUDIO_MODEL_IDX = 2
        ENABLE_FUSION = False  # Fusion currently has issues

        # Ensure that weights are downloaded
        ckpt = CLAP_MODELS[CLAP_MODEL_IDX]
        ckpt_pth = CLAP_PRETRAINED_DIR / ckpt

        if not os.path.exists(ckpt_pth):
            CLAP_PRETRAINED_DIR.mkdir(exist_ok=True, parents=True)
            print(f"Downloading weights for checkpoint {ckpt}")
            ckpt_pth = download_file(CLAP_DOWNLOAD_LINK + ckpt, CLAP_PRETRAINED_DIR)

        # Initialize model
        self.model = laion_clap.CLAP_Module(
            enable_fusion=ENABLE_FUSION, 
            amodel=CLAP_AUDIO_MODELS[CLAP_AUDIO_MODEL_IDX]
        )

        # Load checkpoint with DataParallel prefix handling
        ckpt_data = torch.load(ckpt_pth, map_location=DEVICE)
        
        # Remove 'module.' prefix from DataParallel checkpoints
        state_dict = ckpt_data['state_dict']
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('module.', '')
            new_state_dict[new_key] = value
        
        missing_keys, unexpected_keys = self.model.model.load_state_dict(new_state_dict, strict=False)
        
        # Only print if there are significant issues
        if missing_keys:
            print(f"Warning: {len(missing_keys)} missing keys")
        if unexpected_keys and unexpected_keys != ['text_branch.embeddings.position_ids']:
            print(f"Warning: Unexpected keys: {unexpected_keys}")

        self.model = self.model.to(DEVICE)

        # Ensure model does not track parameter gradients (wastes memory)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def preprocess_audio(self, signal: AudioSignal, quantize: bool = False) -> AudioSignal: 
        signal = signal.resample(self.CLAP_SAMPLE_RATE)
        x = signal.samples.mean(1, keepdim=False)  # Convert to mono if needed

        # Quantize audio
        if quantize:
            quant = (x.clone().clamp(min=-1, max=1) * 32767.).to(torch.int16)
            quant = (quant / 32767.).to(torch.float32)
            # Straight-through estimator: no-op on forward pass, preserves gradient on backward pass
            x = x + (quant - x).detach()
        
        signal.samples = x.unsqueeze(1)
        return signal
    
    def get_audio_embeddings(self, signal: AudioSignal) -> torch.Tensor:
        x = self.preprocess_audio(signal).samples.squeeze(1)  #  shape: (batch, samples)
        return self.model.get_audio_embedding_from_data(x=x, use_tensor=True)
    
    def get_text_embeddings(self, text: Union[str, List[str]]) -> torch.Tensor:
        if isinstance(text, str):
            text = [text]

        # Account for known batch_size==1 issue
        text_padded = text + ["<null>"]
        return self.model.get_text_embedding(text_padded, use_tensor=True)[:-1]
    
    def compute_similarity(self, audio_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between audio and text embeddings.
        
        Args:
            audio_emb: shape (batch_audio, embedding_dim)
            text_emb: shape (batch_text, embedding_dim)
        
        Returns:
            similarity: shape (batch_audio, batch_text)
        """
        # Normalize embeddings
        audio_emb = audio_emb / audio_emb.norm(dim=-1, keepdim=True)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        
        # Compute cosine similarity (dot product of normalized vectors)
        return audio_emb @ text_emb.T
    
    # def compute_similarity(self, audio_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
    #     audio_emb = audio_emb / audio_emb.norm(dim=-1, keepdim=True)
    #     text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        
    #     logit_scale = self.model.model.logit_scale_a.exp()  # or logit_scale_t
    #     similarity = logit_scale * (audio_emb @ text_emb.T)
    #     return similarity
    
    @property
    def sample_rate(self):
        return self.CLAP_SAMPLE_RATE