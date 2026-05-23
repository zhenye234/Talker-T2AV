"""init_model() factory for the WavLM-Large ECAPA-TDNN speaker encoder.

Vendored (and trimmed to the wavlm_large path) from
https://github.com/microsoft/UniSpeech/tree/main/downstreams/speaker_verification
"""
import torch

from .models2.ecapa_tdnn import ECAPA_TDNN_SMALL

MODEL_LIST = ["ecapa_tdnn", "wavlm_base_plus", "wavlm_large"]


def init_model(model_name: str, checkpoint: str = None):
    """Return an ECAPA-TDNN speaker encoder.
    For Talker-T2AV we always call with `model_name='wavlm_large'`.
    """
    if model_name == "wavlm_base_plus":
        model = ECAPA_TDNN_SMALL(feat_dim=768, feat_type="wavlm_base_plus", config_path=None)
    elif model_name == "wavlm_large":
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large", config_path=None)
    elif model_name == "ecapa_tdnn":
        model = ECAPA_TDNN_SMALL(feat_dim=40, feat_type="fbank")
    else:
        raise ValueError(
            f"Unsupported model_name {model_name!r}; expected one of {MODEL_LIST}"
        )

    if checkpoint is not None:
        state_dict = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict["model"], strict=False)
    return model
