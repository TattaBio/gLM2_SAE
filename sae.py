"""JumpReLU SAE encoder for gLM2 layer-24 residuals."""

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer

GLM2_MODEL = "tattabio/gLM2_650M"
SAE_REPO = "tattabio/gLM2_650M_sae"
SAE_ID = "glm2.encoder.layers.24"
SAE_LAYER = 24
CONTEXT_SIZE = 4096


class JumpReluSae:
    def __init__(self, repo: str = SAE_REPO, sae_id: str = SAE_ID, device: str = "cpu"):
        weights = load_file(hf_hub_download(repo, f"{sae_id}/sae_weights.safetensors"), device=device)
        self.W_enc = weights["W_enc"]  # [d_in, d_sae]
        self.b_enc = weights["b_enc"]  # [d_sae]
        self.b_dec = weights["b_dec"]  # [d_in]
        self.threshold = weights["threshold"]  # [d_sae]

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(n, d_in) residuals -> (n, d_sae) latent activations."""
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc
        return pre * (pre > self.threshold)


def load_glm2(device: str):
    """gLM2 encoder truncated after SAE_LAYER, so `last_hidden_state` is the SAE's input."""
    tokenizer = AutoTokenizer.from_pretrained(GLM2_MODEL, trust_remote_code=True)
    model = AutoModel.from_pretrained(GLM2_MODEL, trust_remote_code=True)
    model.encoder.layers = model.encoder.layers[: SAE_LAYER + 1]
    return model.eval().requires_grad_(False).to(device), tokenizer
