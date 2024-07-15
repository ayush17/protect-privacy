import torch
from torch.nn import functional as F
from torch import nn
from transformers import PreTrainedModel, GPTJForCausalLM
from torch import Tensor
import numpy as np

class ICVLayer(nn.Module):
    def __init__(self, icv, alpha):
        super(ICVLayer, self).__init__()
        self.icv = icv
        self.alpha = alpha

    def forward(self, x):
        return x + self.alpha * self.icv

def find_module(layer, keywords):
    for name, module in layer.named_children():
        if any(keyword in name.lower() for keyword in keywords):
            return module
    return None


def find_longest_modulelist(model, path=""):
    """
    Recursively find the longest nn.ModuleList in a PyTorch model.
    Args:
        model: PyTorch model.
        path: Current path in the model (used for recursion).
    Returns:
        Tuple with path and length of the longest nn.ModuleList found.
    """
    longest_path = path
    longest_len = 0

    for name, child in model.named_children():
        if isinstance(child, nn.ModuleList) and len(child) > longest_len:
            longest_len = len(child)
            longest_path = f"{path}.{name}" if path else name

        # Recursively check the child's children
        child_path, child_len = find_longest_modulelist(child, f"{path}.{name}" if path else name)
        if child_len > longest_len:
            longest_len = child_len
            longest_path = child_path

    return longest_path, longest_len


def get_layers_path(model: PreTrainedModel):
    longest_path, longest_len = find_longest_modulelist(model)
    return longest_path

def get_nested_attr(obj, attr_path):
    attrs = attr_path.split(".")
    for attr in attrs:
        obj = getattr(obj, attr)
    return obj

def get_layers(model: PreTrainedModel):
    longest_path = get_layers_path(model)
    return get_nested_attr(model, longest_path)

def add_icv_layers(model: PreTrainedModel, icv: Tensor, alpha: list):
    layers = get_layers(model)
    print(layers)
    mlp_keywords = ["mlp", "feedforward", "ffn"]
    print("LEN ICV",len(icv))
    print("LEN layers",len(layers))
    assert len(icv) == len(layers)
    for i, layer in enumerate(layers):
        original_mlp = find_module(layer, mlp_keywords)
        layer.mlp = nn.Sequential(original_mlp, ICVLayer(icv[i], alpha)) 

def remove_icv_layers(model: nn.Module):
    layers = get_layers(model)
    for layer in layers:
        if hasattr(layer, 'icv_layer'):
            original_mlp = layer.icv_layer[0]
            layer.add_module('mlp', original_mlp)
            del layer.icv_layer
