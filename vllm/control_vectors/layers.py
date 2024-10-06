from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ControlVectorMapping:
    layer_mapping: Dict[int, torch.Tensor]


class BaseLayerWithControlVector(nn.Module):
    pass


class MLPWithControlVector(BaseLayerWithControlVector):

    def __init__(self, base_layer) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.control_vectors = {}
        self.normalize = True
        self.active_vector: torch.Tensor = None
        self.scale_factor = None

    def set_normalization(self, normalize: bool) -> None:
        self.normalize = normalize

    def set_layer_id(self, layer_id: int) -> None:
        """assign the layer id of this MLP layer"""
        self.layer_id = layer_id

    def set_control_vector(self, index: int, cv_vector: torch.Tensor, scale_factor: float):
        """Set a control vector at a specific index."""
        self.control_vectors[index] = cv_vector
        self.scale_factor = scale_factor

    def get_control_vector(self, index: int) -> Optional[torch.Tensor]:
        """Get a control vector by index."""
        return self.control_vectors.get(index)

    def reset_control_vector(self, index: int):
        """Reset a control vector to zero at a specific index."""
        if index in self.control_vectors:
            self.control_vectors[index] = 0

    def set_active_tensor(self, index: int):
        """Sets the active vector"""
        if index is not None and index in self.control_vectors:
            self.active_vector = self.control_vectors[index]
        else:
            self.active_vector = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional application of control vectors."""
        hidden_states = self.base_layer(hidden_states)
        norm_pre = torch.norm(hidden_states, dim=-1, keepdim=True)
        cv = self.active_vector

        if cv is not None and cv.numel() > 0:
            y = 0 
            lambda_sim = 1.0 + torch.max(torch.tensor([0.]).to(hidden_states.device), F.cosine_similarity(hidden_states, -cv[None, :], dim=-1)).unsqueeze(-1)
            y += self.scale_factor * lambda_sim * F.normalize(cv, dim=-1).unsqueeze(0).repeat(hidden_states.size(0), 1)
            hidden_states = F.normalize(F.normalize(hidden_states, p=2, dim=-1) + y, p=2, dim=-1) * norm_pre
            hidden_states = hidden_states.to(torch.bfloat16)
        
        return hidden_states
