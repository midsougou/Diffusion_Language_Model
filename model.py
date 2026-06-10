import torch
import torch.nn as nn

class DiffusionTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.time_emb = nn.Embedding(cfg["diffusion_steps"] + 1, cfg["emb_dim"])
        self.drop = nn.Dropout(cfg["dropout"])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg["emb_dim"],
            nhead=cfg["n_heads"],
            dim_feedforward=cfg["d_ff"],
            dropout=cfg["dropout"],
            activation="gelu",
            batch_first=True,
            norm_first=False
        )

        self.encoder = nn.TransformerEncoder(encoder_layer , num_layers=cfg["n_layers"])
        self.final_norm = nn.LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

        # Tie weights (optional; common in LMs)
        self.out_head.weight = self.tok_emb.weight

    def forward(self, input_ids, timesteps, attention_mask=None):
        # input_ids of size [batch_size, seq_length]
        # timesteps of size [batch_size] integer diffusion step in {1, 2, ...T}
        batch_size, seq_length = input_ids.shape

        tok_embeds = self.tok_emb(input_ids)
        pos_embeds = self.pos_emb(torch.arange(seq_length, device=input_ids.device))
        x = tok_embeds + pos_embeds # [batch_size, seq_length, emb_dim] emb_dim = d_model

        # NEW
        t_emb = self.time_emb(timesteps) # [batch_size, emb_dim]
        t_emb = t_emb.unsqueeze(1) # [batch_size, 1, emb_dim] to allow broadcasting later
        x = x + t_emb # [batch_size, seq_length, emb_dim]

        x = self.drop(x)

        if attention_mask is None:
            src_key_padding_mask = None
        else:
            src_key_padding_mask = ~attention_mask  # invert: True = PAD tokens which will be ignored

        x = self.encoder(x, is_causal=False, src_key_padding_mask=src_key_padding_mask) # this is by default False the causal attention mask
        x = self.final_norm(x)
        logits = self.out_head(x) # [batch_size, seq_length, vocab_size]

        return logits