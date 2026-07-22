import torch
from .base import BaseModelAdapter

class HuggingFaceAdapter(BaseModelAdapter):
    def __init__(self, model_name: str, trust_remote_code: bool = False):
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError(
                "HuggingFaceAdapter requires the optional 'transformers' package."
            ) from exc
        # e.g., 'InstaDeepAI/nucleotide-transformer-2.5b-multi-species'
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # Typically last_hidden_state is what we need for sequence tasks
        if hasattr(outputs, 'last_hidden_state'):
            return outputs.last_hidden_state
        elif len(outputs) > 0:
            return outputs[0]
        else:
            raise ValueError("Unexpected HuggingFace model output format.")

    def get_hidden_size(self) -> int:
        hidden_size = getattr(self.model.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(self.model.config, "d_model", None)
        if hidden_size is None:
            raise AttributeError("The Hugging Face model config has no hidden_size or d_model.")
        return int(hidden_size)
