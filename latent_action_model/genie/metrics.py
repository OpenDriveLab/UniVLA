"""Small, device-aware metrics used while training the latent action model."""

import torch
from torch import Tensor


def codebook_usage(indices: Tensor, num_codes: int) -> Tensor:
    """Return the fraction of codebook entries used by ``indices``.

    The result is kept on the same device as ``indices`` so the metric can be
    computed from every distributed worker without an implicit device copy.
    ``num_codes`` is supplied by the corresponding vector-quantizer rather than
    being inferred from the observed batch (which would make the denominator
    change from step to step).
    """
    if num_codes <= 0:
        raise ValueError(f"num_codes must be positive, got {num_codes}")

    flat_indices = indices.reshape(-1).to(dtype=torch.long)
    used = torch.zeros(num_codes, device=indices.device, dtype=torch.bool)
    if flat_indices.numel() > 0:
        used[torch.unique(flat_indices)] = True
    return used.to(dtype=torch.float32).mean()
