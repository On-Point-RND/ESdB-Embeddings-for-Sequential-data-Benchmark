import torch


def get_pad_mask_from_lengths(
    lengths: torch.Tensor,
    seq_len: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    if device is None:
        device = lengths.device
    return torch.arange(seq_len, device=device)[None, :] < lengths[:, None]
