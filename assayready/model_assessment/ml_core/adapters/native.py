import torch
from .base import BaseModelAdapter
from ..model import DnaModel
from ...data_core.config import DnaConfig

class NativeAdapter(BaseModelAdapter):
    """Expose the native :class:`DnaModel` through the adapter interface.

    ``tokenizer`` is accepted for compatibility with older callers, but model
    construction only requires the model configuration.
    """

    def __init__(self, config: DnaConfig, tokenizer=None):
        super().__init__()
        del tokenizer
        self.model = DnaModel(config)
        self.config = config
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            kmer_ids=kwargs.get("kmer_ids"),
            **{key: value for key, value in kwargs.items() if key != "kmer_ids"},
        )
        return outputs["last_hidden_state"]

    def get_hidden_size(self) -> int:
        return self.config.hidden_size
