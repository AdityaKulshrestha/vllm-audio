"""Audio codec configuration for TTS Model"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class AudioCodecConfig:
    """
    Configuration for audio codec used in TTS models.
    """
    codec_type: str = "snac"
    model_name: str = "hubertsiuzdak/snac_24khz"
    sample_rate: int = 24000
    start_token: int = 128257
    end_token: int = 128258
    token_offset: int = 128266
    num_codebooks: int = 7
    codebook_size: int = 4096
    bitrate: Optional[int] = None  # in kbps

    @classmethod
    def from_model_config(cls, model_config) -> "AudioCodecConfig | None":
        """Creates AudioCodecConfig from ModelConfig if audio_encodec is set"""
        if model_config.audio_encodec is None:
            return None
        
        parts = model_config.audio_encodec.split("/")
        codec_type = parts[0]

        if codec_type == "snac":
            model_name = parts[1] if len(parts) > 1 else "hubertsiuzdak/snac_24khz"
            sample_rate = int(parts[2]) if len(parts) > 2 else 24000
            return cls(
                codec_type="snac",
                model_name=model_name,
                sample_rate=sample_rate,
                num_codebooks=7,
                codebook_size=4096,
            )
        elif codec_type == "encodec":
            model_name = parts[1] if len(parts) > 1 else "facebook/encodec_24khz"
            sample_rate = int(parts[2]) if len(parts) > 2 else 24000
            bitrate = int(parts[3]) if len(parts) > 3 else None
            return cls(
                codec_type="encodec",
                model_name=model_name,
                sample_rate=sample_rate,
                bitrate=bitrate,
                num_codebooks=8,
                codebook_size=1024,
            )
        else:
            raise ValueError(f"Unsupported codec type: {codec_type}")