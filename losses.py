import torch
import torch.nn.functional as F


def loss_bce(input, target):
    BCE = F.binary_cross_entropy(input, target, reduction="sum") / target.size(0)
    return BCE
