"""Shared encoder, BIO head, target attention and gated local/global fusion."""
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel


class JointABSA(nn.Module):
    def __init__(self, encoder, local_radius=16, fusion=True, dropout=0.15):
        super().__init__()
        self.encoder = encoder
        self.local_radius = local_radius
        self.fusion = fusion
        self.dropout_rate = dropout
        hidden = encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.extractor = nn.Linear(hidden, 3)
        self.query = nn.Linear(hidden, hidden)
        self.global_key = nn.Linear(hidden, hidden)
        self.local_key = nn.Linear(hidden, hidden)
        self.gate = nn.Linear(hidden * 3, hidden)
        self.sentiment = nn.Sequential(nn.LayerNorm(hidden * 3), nn.Dropout(dropout),
                                       nn.Linear(hidden * 3, 256), nn.GELU(),
                                       nn.Dropout(dropout), nn.Linear(256, 3))

    @classmethod
    def from_pretrained(cls, name, **kwargs):
        return cls(AutoModel.from_pretrained(name), **kwargs)

    def encode(self, inputs):
        hidden = self.encoder(**inputs, return_dict=True).last_hidden_state
        return hidden, self.extractor(self.dropout(hidden))

    def classify(self, hidden, valid, spans):
        if not len(spans):
            return hidden.new_empty((0, 3))
        batch, starts, ends = spans.unbind(-1)
        positions = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
        target_mask = (positions >= starts[:, None]) & (positions < ends[:, None])
        selected = hidden[batch]
        aspect = (selected * target_mask.unsqueeze(-1)).sum(1) / target_mask.sum(1, keepdim=True).clamp_min(1)
        query = self.query(aspect)
        global_scores = (self.global_key(selected) * query.unsqueeze(1)).sum(-1) / math.sqrt(hidden.shape[-1])
        context_mask = valid[batch]
        global_attention = global_scores.float().masked_fill(~context_mask, -1e4).softmax(-1).to(hidden.dtype)
        global_context = (global_attention.unsqueeze(-1) * selected).sum(1)
        if self.fusion:
            local_mask = context_mask & (positions >= starts[:, None] - self.local_radius) & (positions < ends[:, None] + self.local_radius)
            local_scores = (self.local_key(selected) * query.unsqueeze(1)).sum(-1) / math.sqrt(hidden.shape[-1])
            local_attention = local_scores.float().masked_fill(~local_mask, -1e4).softmax(-1).to(hidden.dtype)
            local_context = (local_attention.unsqueeze(-1) * selected).sum(1)
            gate = self.gate(torch.cat((aspect, local_context, global_context), -1)).sigmoid()
            context = gate * local_context + (1 - gate) * global_context
        else:
            context = global_context
        features = torch.cat((hidden[batch, 0], aspect, context), -1)
        return self.sentiment(features)

    def forward(self, inputs, valid, spans):
        hidden, bio_logits = self.encode(inputs)
        return bio_logits, self.classify(hidden, valid, spans)

    def save_checkpoint(self, directory, tokenizer, metadata):
        import json
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.encoder.config.save_pretrained(directory / "encoder_config")
        tokenizer.save_pretrained(directory / "tokenizer")
        # Own checkpoint only; torch.load below disables arbitrary pickle objects.
        torch.save({k: v.detach().cpu() for k, v in self.state_dict().items()}, directory / "model.pt")
        settings = {**metadata, "local_radius": self.local_radius, "fusion": self.fusion,
                    "dropout": self.dropout_rate, "labels": ["negative", "neutral", "positive"]}
        (directory / "settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load_checkpoint(cls, directory):
        import json
        from transformers import AutoTokenizer
        directory = Path(directory)
        settings = json.loads((directory / "settings.json").read_text(encoding="utf-8"))
        encoder = AutoModel.from_config(AutoConfig.from_pretrained(directory / "encoder_config"))
        model = cls(encoder, local_radius=settings["local_radius"], fusion=settings["fusion"], dropout=settings["dropout"])
        state = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        tokenizer = AutoTokenizer.from_pretrained(directory / "tokenizer", use_fast=True)
        return model, tokenizer, settings


def joint_loss(bio_logits, sentiment_logits, bio_labels, sentiment_labels,
               class_weights=None, extraction_weight=0.5, smoothing=0.05):
    active = bio_labels != -100
    if active.any():
        bio_weights = bio_logits.new_tensor([0.3, 1.0, 1.0], dtype=torch.float32)
        extraction = F.cross_entropy(bio_logits[active].float(), bio_labels[active], weight=bio_weights)
    else:
        extraction = bio_logits.sum() * 0
    if len(sentiment_labels):
        sentiment = F.cross_entropy(sentiment_logits.float(), sentiment_labels,
                                   weight=class_weights, label_smoothing=smoothing)
    else:
        sentiment = bio_logits.sum() * 0
    return sentiment + extraction_weight * extraction, sentiment, extraction
