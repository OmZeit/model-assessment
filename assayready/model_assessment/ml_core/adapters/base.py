import abc
import torch
import torch.nn as nn

class BaseModelAdapter(nn.Module, abc.ABC):
    """
    Base interface for all foundation models (Native or HuggingFace).
    Provides a unified API for extracting sequence embeddings.
    """
    @abc.abstractmethod
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass to get sequence embeddings.
        Returns tensor of shape [batch, seq_len, hidden_size]
        """
        pass
    
    @abc.abstractmethod
    def get_hidden_size(self) -> int:
        """Returns the hidden dimension size of the model."""
        pass
    
    def freeze_backbone(self):
        """Freezes all backbone weights for PEFT/LoRA fine-tuning."""
        for param in self.parameters():
            param.requires_grad = False
