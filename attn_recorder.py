import torch
import torch.nn as nn
from typing import Dict, List

class CrossAttnCaptureProcessor(nn.Module):
    """
    Drop-in AttnProcessor that captures cross-attention maps (attn2) from the UNet.
    Maps are stored in store[name] with shape [B, heads, Q, K], where Q is HxW
    image latent tokens and K is the text token length (typically 77 for SD2.1).
    """
    def __init__(self, store: Dict[str, List[torch.Tensor]], name: str):
        super().__init__()
        self.store = store
        self.name = name

    def __call__(
        self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None
    ):
        # Cross-attention (attn2) only when encoder_hidden_states is provided
        is_cross = encoder_hidden_states is not None
        residual = hidden_states

        # 1) Normalization (matches diffusers Attention)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states)

        # 2) Project to q, k, v
        batch_size, sequence_length, _ = hidden_states.shape
        if is_cross:
            key_value_states = encoder_hidden_states
        else:
            key_value_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(key_value_states)
        value = attn.to_v(key_value_states)

        # 3) Split into attention heads
        dim = query.shape[-1]
        head_dim = dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)       # [B, H, Q, d]
        key   = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)         # [B, H, K, d]
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)       # [B, H, K, d]

        # 4) Scores & softmax -> attention probabilities
        #    Note: attn.scale is typically 1/sqrt(d)
        scores = torch.matmul(query, key.transpose(-2, -1)) * attn.scale                # [B, H, Q, K]
        if attention_mask is not None:
            # attention_mask is usually [B, 1, Q, K] or [B, 1, 1, K], consistent with diffusers
            scores = scores + attention_mask

        attn_probs = scores.softmax(dim=-1)                                            # [B, H, Q, K]


        # 5) Record (cross-attn only; self-attn is usually unnecessary)
        # if is_cross:
        #     # Move to CPU to save VRAM; remove .detach().to("cpu") if gradients are needed
        #     self.store.setdefault(self.name, []).append(attn_probs)

        if encoder_hidden_states is not None:
            self.store[self.name] = attn_probs 

        # 6) Output
        hidden_states = torch.matmul(attn_probs, value)                                 # [B, H, Q, d]
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, sequence_length, dim)
        hidden_states = attn.to_out[0](hidden_states)
        if attn.to_out[1] is not None:
            hidden_states = attn.to_out[1](hidden_states)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        return hidden_states
