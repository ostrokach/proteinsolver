import torch


def str_to_tensor(s: str) -> torch.tensor:
    t = torch.tensor([int(i) for i in s], dtype=torch.long)
    return t
