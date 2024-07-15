import torch
import torch.nn as nn

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

def get_layers(model: nn.Module):
    layers = [module for module in model.modules() if isinstance(module, nn.Transformer)]
    return layers

def add_icv_layers(model: nn.Module, icv: torch.Tensor, alpha: list):
    layers = get_layers(model)
    mlp_keywords = ["mlp", "feedforward", "ffn"]
    
    for i, layer in enumerate(layers):
        original_mlp = find_module(layer, mlp_keywords)
        if original_mlp:
            icv_layer = ICVLayer(icv[i], alpha[i])
            layer.add_module("icv_layer", nn.Sequential(original_mlp, icv_layer))

def remove_icv_layers(model: nn.Module):
    layers = get_layers(model)
    for layer in layers:
        if hasattr(layer, 'icv_layer'):
            original_mlp = layer.icv_layer[0]
            layer.add_module('mlp', original_mlp)
            del layer.icv_layer
