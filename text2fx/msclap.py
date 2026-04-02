from pathlib import Path
from tqdm import tqdm
import datetime
import unicodedata
import re
from typing import Optional

import torch
import torchaudio.transforms as T
import numpy as np
import audiotools as at
import dasp_pytorch
from audiotools import AudioSignal
from typing import Iterable
import random
from torch.utils.tensorboard import SummaryWriter

from msclap import CLAP

import matplotlib.pyplot as plt
from text2fx.core import AbstractCLAPWrapper, download_file 
from text2fx.constants import PRETRAINED_DIR, DEVICE


#REQUIRES TRANSFORMERS >= 4.34.0
""" utility wrapper around MS CLAP model! """
class MSCLAPWrapper(AbstractCLAPWrapper):

    def __init__(self, use_cuda: Optional[bool] = None, crop_mode: str = "random"):
        if use_cuda is None:
            use_cuda = torch.cuda.is_available()
        if crop_mode not in {"center", "random"}:
            raise ValueError(f"Unsupported crop_mode: {crop_mode}")

        self.crop_mode = crop_mode
        self.clap_model = CLAP(version='2023', use_cuda=use_cuda)

    #testing just the clap_model.load_audio() !!
    def resample(self, signal: AudioSignal, resample=True):
        """
        trying to see if resampling step in read_audio (step 1) is the issue, but it seems its from the very beginning w/ reading in the file
        """
        audio_time_series = signal.samples
        resample_rate = self.clap_model.args.sampling_rate
        sample_rate = signal.sample_rate #check

        if resample and resample_rate != sample_rate:
            resampler = T.Resample(sample_rate, resample_rate)
            resampler.to(signal.device)
            audio_time_series = resampler(audio_time_series)
            signal.samples = audio_time_series
            signal.sample_rate = resample_rate
            
        return signal

    def audio_trim(self, audio_time_series, audio_duration, sample_rate):
        audio_time_series = audio_time_series.squeeze(0).squeeze(0)
        target_num_samples = audio_duration * sample_rate

        #if audio duration is shorter than 7 seconds, repeat samples
        if target_num_samples >= audio_time_series.shape[0]:
            repeat_factor = int(np.ceil(target_num_samples / audio_time_series.shape[0]))
            # Repeat audio_time_series by repeat_factor to match audio_duration
            audio_time_series = audio_time_series.repeat(repeat_factor)
            # remove excess part of audio_time_series
            audio_time_series = audio_time_series[0:target_num_samples]
        else:
            max_start = audio_time_series.shape[0] - target_num_samples
            if self.crop_mode == "random":
                start_index = random.randrange(max_start)
            else:
                start_index = max_start // 2
            audio_time_series = audio_time_series[start_index:start_index + target_num_samples]
        return audio_time_series.unsqueeze(0).unsqueeze(0).float()

    def preprocess_audio(self, signal: AudioSignal) -> AudioSignal: #returns an AudioSignal
        sig_resamp = []
        for i in range(signal.shape[0]):
            _sig = self.resample(signal[i]) #uses CLAP function on AudioSignal.samples
            _sig.to_mono()
            _sig.samples = self.audio_trim(
                _sig.samples, 
                self.clap_model.args.duration, 
                self.clap_model.args.sampling_rate
            ) #should work for batches, NOTE: might be good to vectorize
            sig_resamp.append(_sig)

        return AudioSignal.batch(sig_resamp)

    def _get_audio_embed(self, preprocessed_audio: AudioSignal) -> torch.Tensor: 
        preprocessed_audio = preprocessed_audio.reshape(preprocessed_audio.shape[0], preprocessed_audio.shape[2])
        return self.clap_model.clap.audio_encoder(preprocessed_audio)[0]

    def get_audio_embeddings(self, signal: AudioSignal) -> torch.Tensor: 
        return self._get_audio_embed(self.preprocess_audio(signal).samples)

    def get_text_embeddings(self, texts) -> torch.Tensor:
        return self.clap_model.get_text_embeddings(texts)
    
    def compute_similarities(self, audio_emb, text_emb) -> torch.Tensor:
        return self.clap_model.compute_similarities(audio_emb, text_emb)

    @property
    def sample_rate(self):
        return self.clap_model.args.sampling_rate
