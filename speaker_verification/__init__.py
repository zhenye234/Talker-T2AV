"""Talker-T2AV: vendored speaker_verification package.

Sourced from the WavLM-Large speaker-verification downstream of
Microsoft UniSpeech:

    https://github.com/microsoft/UniSpeech/tree/main/downstreams/speaker_verification

Trimmed to just what we need at inference time — the `wavlm_large` ECAPA-TDNN
configuration. The CLI (`fire`-based verification) and the `unispeech_sat`
`UpstreamExpert` config path are dropped.
"""
from .verification import init_model

__all__ = ["init_model"]
