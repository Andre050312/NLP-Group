"""Aspect-conditioned sentence pairs with target-preserving context cropping."""
import torch
from torch import nn
from torch.utils.data import Dataset

from .data import SENTIMENTS, align_span, trim_span
from .model import JointABSA


class PairDataset(Dataset):
    """One training row per gold aspect; no implicit extraction supervision."""
    def __init__(self, documents, tokenizer, max_length=192, stride=None):
        if not tokenizer.is_fast:
            raise ValueError("PairDataset requires a fast tokenizer")
        self.documents, self.rows = documents, []
        non_exact, cropped, targets = [], 0, 0
        for doc_index, document in enumerate(documents):
            encoded = tokenizer(document.text, add_special_tokens=False, return_offsets_mapping=True)
            full_ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
            for aspect in document.aspects:
                if aspect.label not in SENTIMENTS:
                    continue
                targets += 1
                text = document.text[aspect.start:aspect.end]
                aspect_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
                positions = align_span(offsets, aspect.start, aspect.end)
                if not positions or not aspect_ids:
                    raise ValueError(f"Cannot align aspect in {document.uid}: {text!r}")
                first, end = positions[0], positions[-1] + 1
                budget = max_length - len(aspect_ids) - tokenizer.num_special_tokens_to_add(pair=True)
                if budget < end - first:
                    raise ValueError("max-length too small to retain target and aspect query")
                left = max(0, first - (budget - (end - first)) // 2)
                left = min(left, max(0, len(full_ids) - budget))
                context_ids = full_ids[left:left + budget]
                cropped += int(len(context_ids) < len(full_ids))
                input_ids = tokenizer.build_inputs_with_special_tokens(context_ids, aspect_ids)
                # The encoder's own pair template inserts CLS/SEP tokens. Negative
                # sentinels locate the two sequences without hardcoding a BERT layout.
                markers = tokenizer.build_inputs_with_special_tokens([-1] * len(context_ids), [-2] * len(aspect_ids))
                context_positions = [i for i, marker in enumerate(markers) if marker == -1]
                query_mask = [marker == -2 for marker in markers]
                start_token = context_positions[first - left]
                end_token = context_positions[end - left - 1] + 1
                inputs = {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}
                if "token_type_ids" in tokenizer.model_input_names:
                    inputs["token_type_ids"] = tokenizer.create_token_type_ids_from_sequences(context_ids, aspect_ids)
                covering = trim_span(document.text, offsets[first][0], offsets[end - 1][1])
                if covering != (aspect.start, aspect.end):
                    non_exact.append({"sentence_id": document.uid, "aspect": text,
                                      "start": aspect.start, "end": aspect.end,
                                      "covering_token_text": document.text[covering[0]:covering[1]]})
                label = SENTIMENTS.index(aspect.label)
                self.rows.append({"inputs": inputs, "valid": query_mask, "bio": [-100] * len(input_ids),
                                  "targets": [(start_token, end_token, label, aspect.start, aspect.end)],
                                  "doc_index": doc_index})
        self.alignment_stats = {"gold_aspects": targets, "non_exact_token_spans": len(non_exact),
                                "non_exact_examples": non_exact[:10], "cropped_contexts": cropped}
        if not self.rows:
            raise ValueError("Pair dataset has no three-class aspect targets")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class PairABSA(JointABSA):
    """The target participates in every encoder layer through the second sequence."""
    model_kind = "pair"

    def __init__(self, encoder, dropout=0.15):
        nn.Module.__init__(self)
        self.encoder = encoder
        self.local_radius, self.fusion, self.dropout_rate = 0, False, dropout
        hidden = encoder.config.hidden_size
        self.sentiment = nn.Sequential(nn.LayerNorm(hidden * 3), nn.Dropout(dropout),
                                       nn.Linear(hidden * 3, 256), nn.GELU(),
                                       nn.Dropout(dropout), nn.Linear(256, 3))

    def encode(self, inputs):
        hidden = self.encoder(**inputs, return_dict=True).last_hidden_state
        # Reuse the training interface. All pair BIO labels are ignored; no BIO head.
        return hidden, hidden[..., :3] * 0

    def classify(self, hidden, query_mask, spans):
        if not len(spans):
            return hidden.new_empty((0, 3))
        batch, starts, ends = spans.unbind(-1)
        positions = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
        target_mask = (positions >= starts[:, None]) & (positions < ends[:, None])
        selected = hidden[batch]
        target = (selected * target_mask.unsqueeze(-1)).sum(1) / target_mask.sum(1, keepdim=True).clamp_min(1)
        query = query_mask[batch]
        query_vector = (selected * query.unsqueeze(-1)).sum(1) / query.sum(1, keepdim=True).clamp_min(1)
        return self.sentiment(torch.cat((hidden[batch, 0], target, query_vector), -1))
