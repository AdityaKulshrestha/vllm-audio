"""Audio token processing for TTS Model."""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch


@dataclass
class AudioCodecConfig:
    """Configuration for audio codec (SNAC, EnCoddec etc.)"""
    codec_type: str = "snac"
    model_name: str = "hubertsiuzdak/snac_24khz"
    sample_rate: int = 24000

    start_token: int = 128257
    end_token: int = 128258
    token_offset: int = 128266
    num_codebooks: int = 7
    codebook_size: int = 4096


class AudioTokenProcessor:
    """
    Proceses audio tokens from TTS models into audio waveforms.

    Handles different codec models 
    """

    def __init__(self, config: AudioCodecConfig, device: str = "cpu"):
        self.config = config 
        self.device = device
        self._codec = None 

    @property
    def codec(self):
        """Lazy load the codec model"""
        if self._codec is None:
            if self.config.codec_type == "snac":
                from snac import SNAC
                self._codec = SNAC.from_pretrained(self.config.model_name)
                self._codec = self._codec.eval()
            else:
                raise ValueError(f"Unknown codec: {self.config.codec_type}")
            
        return self._codec
    
    def extract_audio_tokens(self, token_ids: list[int]) -> list[int]:
        """
        Extract and normalize audio tokens from model output.
        Removes special tokens and applies offset correction.
        """

        cfg = self.config

        # Filter out end tokens
        end_index = token_ids.index(cfg.end_token) if cfg.end_token in token_ids else len(token_ids)
        filtered = token_ids[:end_index]

        # Trim to multiple of num_codebooks (7 for SNAC)
        trim_len = (len(filtered) // cfg.num_codebooks) * cfg.num_codebooks
        trimmed = filtered[:trim_len]

        # Apply offset correction 
        audio_tokens = [t - cfg.token_offset for t in trimmed]

        return audio_tokens
    
    def redistribute_codes(self, audio_tokens: list[int]) -> tuple[torch.Tensor, ...]:
        """
        Redistribute flat token sequence into SNAC's 3-layer codebook format.

        SNAC uses 3 codebooks with different temporal resolutions.
        - Layer 1: 1 token per frame
        - Layer 2: 2 tokens per frame
        - Layer 3: 4 tokens per frame

        Input pattern (7 tokens per frame):
        [L1_0, L2_0, L3_0, L3_1, L2_1, L3_2, L3_3, ...]
        """
        cfg = self.config
        num_frames = (len(audio_tokens) + 1) // cfg.num_codebooks

        layer_1, layer_2, layer_3 = [], [], []

        for i in range(num_frames):
            base = cfg.num_codebooks * i

            layer_1.append(audio_tokens[base])

            layer_2.append(audio_tokens[base + 1] - cfg.codebook_size)
            layer_2.append(audio_tokens[base + 4] - 4 * cfg.codebook_size)

            # Layer 3: 4 tokens (with different offsets)
            layer_3.append(audio_tokens[base + 2] - 2 * cfg.codebook_size)
            layer_3.append(audio_tokens[base + 3] - 3 * cfg.codebook_size)
            layer_3.append(audio_tokens[base + 5] - 5 * cfg.codebook_size)
            layer_3.append(audio_tokens[base + 6] - 6 * cfg.codebook_size)
        
        codes = [
            torch.tensor(layer_1).unsqueeze(0),
            torch.tensor(layer_2).unsqueeze(0),
            torch.tensor(layer_3).unsqueeze(0),
        ]
        return codes
    
    def decode_to_audio(self, audio_tokens: list[int]) -> np.ndarray:
        """
        Full pipeline: token_ids -> audio waveform

        Returns:
            numpy array of float32 audio samples
        """
        # Extract audio tokens
        audio_tokens = self.extract_audio_tokens(audio_tokens)
        print("These are the tokens: ", audio_tokens)
        if len(audio_tokens) < self.config.num_codebooks:
            # Not enough tokens for even one frame
            return np.array([], dtype=np.float32)
        
        # Redistribute into codebook format
        codes = self.redistribute_codes(audio_tokens)

        # Decode with SNAC
        with torch.inference_mode():
            audio_hat = self.codec.decode(codes)

        audio_np = audio_hat.numpy()

        if audio_np.ndim == 3:
            audio_np = audio_np[0, 0, :]

        return audio_np.astype(np.float32)
    
    def validate_codes(self, codes: list[int]) -> np.ndarray:
        """
        Check that all code values are within valid range
        """
        for code in codes:
            if torch.any(code < 0) or torch.any(code >= self.config.codebook_size):
                return False
        return True