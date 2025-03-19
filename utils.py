import torch


def pytorch_cos_sim(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=1)